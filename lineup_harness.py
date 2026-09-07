"""Hierarchical player-hero playtime-share model (q_ih) + validation.

Target: q_ih = E[playtime share of hero h for player i | history + ban mask].
(No map/opponent conditioning yet: player history shrunk to team/league,
renormalized over the live legal pool. Do not claim more.)
Hierarchy: player -> team -> league, count-space shrinkage on decayed shares.
Protocol: chronological pass; hyperparams tuned on INNER window (last 120
pre-holdout maps); the last-45-series holdout is scored ONCE with the chosen
config (freeze policy).
"""
import json, glob, math, os, time
from collections import defaultdict
import numpy as np

DRAFT_ROOT = os.environ.get("DRAFT_ROOT", "/content")
roles = json.load(open(DRAFT_ROOT + "/data/roles/heroes.json"))
HEROES = sorted(roles); HIDX = {h: i for i, h in enumerate(HEROES)}; NH = len(HEROES)

recs = [json.load(open(f)) for f in glob.glob(DRAFT_ROOT + "/data/processed/maps/*.json")]
recs.sort(key=lambda r: (r["played_at"], r["map_uid"]))
series_order, seen = [], set()
for r in recs:
    if r["match_id"] not in seen:
        seen.add(r["match_id"]); series_order.append(r["match_id"])
HOLD = set(series_order[-45:])
hold_maps = [r["map_uid"] for r in recs if r["match_id"] in HOLD]
pre = [r["map_uid"] for r in recs if r["match_id"] not in HOLD]
INNER = set(pre[-120:])
HOLDSET = set(hold_maps)

def collect(dp, dt, dg):
    """One chronological pass; returns per-(map,side) team observations with
    as-of ingredient snapshots. dp/dt/dg: per-appearance decay for player/team/league."""
    p_time = defaultdict(lambda: np.zeros(NH))
    t_time = defaultdict(lambda: np.zeros(NH))
    g_time = np.zeros(NH)
    map_u = defaultdict(lambda: np.zeros(NH)); map_n = defaultdict(float)
    obs = []
    for r in recs:
        tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
        bans_by = {"blue": [], "red": []}; prots_by = {"blue": [], "red": []}
        for a in r["actions"]:
            if a["kind"] == "ban": bans_by[a["side"]].append(a["hero"])
            else: prots_by[a["side"]].append(a["hero"])
        mp = r.get("map_name")
        gs2 = g_time.sum()
        if mp and map_n.get(mp, 0) >= 3 and gs2 > 0:
            gref = (g_time + 0.1) / (gs2 + 0.1 * NH)
            gm = (map_u[mp] + 15.0 * gref * 6.0) / (map_n[mp] * 6.0 + 15.0 * 6.0)
            moff = np.clip(np.log(np.maximum(gm / np.maximum(gref, 1e-9), 1e-6)), -1.2, 1.2)
        else:
            moff = np.zeros(NH)
        phase = "hold" if r["map_uid"] in HOLDSET else ("inner" if r["map_uid"] in INNER else "train")
        gs = g_time.sum()
        g_norm = (g_time + 0.1) / (gs + 0.1 * NH)
        for side in ("blue", "red"):
            opp = "red" if side == "blue" else "blue"
            banned = np.zeros(NH, bool)
            for h in bans_by[opp]:
                if h in HIDX: banned[HIDX[h]] = True
            protv = np.zeros(NH)
            for h in prots_by[side]:
                if h in HIDX: protv[HIDX[h]] = 1.0
            team = tn[side]
            players = []
            for p in r["lineups"][side]:
                pid = p["player_id"]
                shares = r["hero_time"].get(pid, {})
                tot = sum(shares.values())
                if tot <= 0: continue
                a_vec = np.zeros(NH)
                for h, s in shares.items():
                    if h in HIDX: a_vec[HIDX[h]] = s / tot
                if a_vec.sum() <= 0: continue
                players.append({"pid": pid, "a": a_vec,
                                "p_vec": p_time[pid].copy(), "p_sum": float(p_time[pid].sum()),
                                "last": None})
            if players:
                obs.append({"map": r["map_uid"], "side": side, "team": team, "phase": phase,
                            "mapname": r.get("map_name"), "banned": banned,
                            "prot": protv, "moff": moff,
                            "t_vec": t_time[team].copy(), "t_sum": float(t_time[team].sum()),
                            "g": g_norm, "players": players})
        # post-map update
        g_time *= dg
        if mp:
            map_u[mp] *= 0.995; map_n[mp] *= 0.995
        for side in ("blue", "red"):
            team = tn[side]
            t_time[team] *= dt
            for p in r["lineups"][side]:
                pid = p["player_id"]
                shares = r["hero_time"].get(pid, {})
                tot = sum(shares.values())
                if tot <= 0: continue
                vec = np.zeros(NH)
                for h, s in shares.items():
                    if h in HIDX: vec[HIDX[h]] = s / tot
                p_time[pid] = dp * p_time[pid] + vec
                t_time[team] += vec
                g_time += vec
                if mp:
                    map_u[mp] += vec
        if mp: map_n[mp] += 1
    return obs

BP = 0.0   # own-protect boost (log-scale)
LM = 0.0   # map-offset loading

def q_for(pl, ob, kp, kt):
    g = ob["g"]
    s_team = (ob["t_vec"] + kt * g) / (ob["t_sum"] + kt)
    q = (pl["p_vec"] + kp * s_team) / (pl["p_sum"] + kp)
    if BP or LM:
        q = q * np.exp(BP * ob["prot"] + LM * ob["moff"])
    q = np.where(ob["banned"], 0.0, q)
    s = q.sum()
    return q / s if s > 0 else np.full(NH, 1.0 / NH)

from scipy.optimize import linear_sum_assignment

def assign_six(qs):
    """Exact max-sum-log-q assignment over the full player x hero matrix (Hungarian)."""
    C = np.stack([-np.log(np.maximum(q, 1e-9)) for q in qs])
    rows, cols = linear_sum_assignment(C)
    out = [0] * len(qs)
    for r_, c_ in zip(rows, cols): out[int(r_)] = int(c_)
    return out

def score(obs, phase, kp, kt):
    top1 = top2 = nobs = 0; ce = 0.0; jac = 0.0; nteam = 0
    boxacc = boxtop2 = 0; cov2 = 0.0
    for ob in obs:
        if ob["phase"] != phase: continue
        qs = []
        for pl in ob["players"]:
            q = q_for(pl, ob, kp, kt); qs.append(q)
            a = pl["a"]
            am = int(np.argmax(a)); order = np.argsort(-q)
            top1 += (int(order[0]) == am); top2 += (am in set(int(x) for x in order[:2]))
            cov2 += float(a[int(order[0])] + (a[int(order[1])] if len(order) > 1 else 0.0))
            ce += float(-(a * np.log(q + 1e-9)).sum()); nobs += 1
        # exact joint assignment; score the boxes the UI would draw
        pick = assign_six(qs)
        for i, pl in enumerate(ob["players"]):
            am_ = int(np.argmax(pl["a"]))
            alt = next((int(x) for x in np.argsort(-qs[i]) if int(x) != pick[i]), None)
            boxacc += (pick[i] == am_); boxtop2 += (am_ in (pick[i], alt))
        pred = set(pick)
        atot = np.zeros(NH)
        for pl in ob["players"]: atot += pl["a"]
        act = set(int(i) for i in np.argsort(-atot)[:len(ob["players"])] if atot[i] > 0)
        if act:
            jac += len(pred & act) / len(pred | act); nteam += 1
    return {"n": nobs, "top1": top1 / max(nobs, 1), "top2": top2 / max(nobs, 1),
            "ce": ce / max(nobs, 1), "jaccard": jac / max(nteam, 1),
            "boxacc": boxacc / max(nobs, 1), "boxtop2": boxtop2 / max(nobs, 1),
            "cov2": cov2 / max(nobs, 1)}

t0 = time.time()
results = {}
for dp in (0.88, 0.95, 0.99):
    obs = collect(dp, 0.88, 0.995)
    for kp in (2.0, 5.0, 10.0, 20.0):
        for kt in (5.0, 20.0, 50.0):
            r = score(obs, "inner", kp, kt)
            results[(dp, kp, kt)] = (r, obs)
            print(f"inner dp={dp} kp={kp} kt={kt} | top1 {r['top1']*100:.1f}% top2 {r['top2']*100:.1f}% ce {r['ce']:.4f} box {r['boxacc']*100:.1f}%/{r['boxtop2']*100:.1f}% cov2 {r['cov2']*100:.1f}% n={r['n']}", flush=True)
best_key = min(results, key=lambda k: results[k][0]["ce"])
print(f"\nBEST on inner: dp={best_key[0]} kp={best_key[1]} kt={best_key[2]}  [{time.time()-t0:.0f}s]", flush=True)

# conditioning grid (protect boost x map loading) on INNER, base config fixed
r_obs = results[best_key][1]
bestc = ((0.0, 0.0), results[best_key][0]["ce"])
for bp in (0.0, 1.0, 2.0, 3.0):
    for lm in (0.0, 0.5, 1.0):
        if bp == 0.0 and lm == 0.0: continue
        globals()["BP"], globals()["LM"] = bp, lm
        rc = score(r_obs, "inner", best_key[1], best_key[2])
        print(f"inner cond bp={bp} lm={lm} | top1 {rc['top1']*100:.1f}% ce {rc['ce']:.4f} box {rc['boxacc']*100:.1f}%/{rc['boxtop2']*100:.1f}% cov2 {rc['cov2']*100:.1f}%", flush=True)
        if rc["ce"] < bestc[1]: bestc = ((bp, lm), rc["ce"])
globals()["BP"], globals()["LM"] = bestc[0]
print(f"BEST conditioning: bp={bestc[0][0]} lm={bestc[0][1]}", flush=True)
condf = score(r_obs, "hold", best_key[1], best_key[2])
print(f"HOLDOUT q+conditioning | top1 {condf['top1']*100:.1f}% top2 {condf['top2']*100:.1f}% ce {condf['ce']:.4f} box {condf['boxacc']*100:.1f}%/{condf['boxtop2']*100:.1f}% cov2 {condf['cov2']*100:.1f}% n={condf['n']}", flush=True)
globals()["BP"], globals()["LM"] = 0.0, 0.0

final = score(r_obs, "hold", best_key[1], best_key[2])
print(f"HOLDOUT q-model      | top1 {final['top1']*100:.1f}% top2 {final['top2']*100:.1f}% ce {final['ce']:.4f} box {final['boxacc']*100:.1f}%/{final['boxtop2']*100:.1f}% cov2 {final['cov2']*100:.1f}% n={final['n']}", flush=True)
naive = score(r_obs, "hold", 1e-6, 1e9)   # kp~0: raw player career share (league fill-in for empty)
prior = score(r_obs, "hold", 1e9, 1e-6)   # kp huge: pure team-rate prior
print(f"HOLDOUT raw-career   | top1 {naive['top1']*100:.1f}% top2 {naive['top2']*100:.1f}% ce {naive['ce']:.4f} box {naive['boxacc']*100:.1f}%/{naive['boxtop2']*100:.1f}% cov2 {naive['cov2']*100:.1f}%", flush=True)
print(f"HOLDOUT team-prior   | top1 {prior['top1']*100:.1f}% top2 {prior['top2']*100:.1f}% ce {prior['ce']:.4f} box {prior['boxacc']*100:.1f}%/{prior['boxtop2']*100:.1f}% cov2 {prior['cov2']*100:.1f}%", flush=True)

# ---- export as-of-END shrunk q per player (latest lineup per team) ----
dp_b, kp_b, kt_b = best_key
p_time = defaultdict(lambda: np.zeros(NH)); t_time = defaultdict(lambda: np.zeros(NH))
g_time = np.zeros(NH); latest = {}
for r in recs:
    tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
    g_time *= 0.995
    for side in ("blue", "red"):
        team = tn[side]; t_time[team] *= 0.88
        names = []
        for p in r["lineups"][side]:
            pid = p["player_id"]
            shares = r["hero_time"].get(pid, {}); tot = sum(shares.values())
            if tot <= 0: continue
            vec = np.zeros(NH)
            for h, s in shares.items():
                if h in HIDX: vec[HIDX[h]] = s / tot
            p_time[pid] = dp_b * p_time[pid] + vec
            t_time[team] += vec; g_time += vec
            names.append((pid, p["name"]))
        if names: latest[team] = names
export = {}
gs = g_time.sum(); g_norm = (g_time + 0.1) / (gs + 0.1 * NH)
for team, names in latest.items():
    s_team = (t_time[team] + kt_b * g_norm) / (t_time[team].sum() + kt_b)
    rows = []
    for pid, nm in names:
        q = (p_time[pid] + kp_b * s_team) / (p_time[pid].sum() + kp_b)
        idx = np.argsort(-q)[:8]
        rows.append({"name": nm,
                     "q": [[HEROES[int(i)], round(float(q[i]), 4)] for i in idx if q[i] > 0.005]})
    export[team] = rows
json.dump({"config": {"dp": dp_b, "kp": kp_b, "kt": kt_b},
           "holdout": final, "player_q": export},
          open("/content/player_q.json" if os.path.isdir("/content") else "player_q.json", "w"))
print("Q_EXPORT_DONE", flush=True)
print("LINEUP_DONE", flush=True)
