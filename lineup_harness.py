"""Hierarchical player-hero playtime-share model (q_ih) + validation.

Target: q_ih = E[playtime share of hero h for player i | draft, map, history].
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
    obs = []
    for r in recs:
        tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
        bans_by = {"blue": [], "red": []}
        for a in r["actions"]:
            if a["kind"] == "ban": bans_by[a["side"]].append(a["hero"])
        phase = "hold" if r["map_uid"] in HOLDSET else ("inner" if r["map_uid"] in INNER else "train")
        gs = g_time.sum()
        g_norm = (g_time + 0.1) / (gs + 0.1 * NH)
        for side in ("blue", "red"):
            opp = "red" if side == "blue" else "blue"
            banned = np.zeros(NH, bool)
            for h in bans_by[opp]:
                if h in HIDX: banned[HIDX[h]] = True
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
                            "t_vec": t_time[team].copy(), "t_sum": float(t_time[team].sum()),
                            "g": g_norm, "players": players})
        # post-map update
        g_time *= dg
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
    return obs

def q_for(pl, ob, kp, kt):
    g = ob["g"]
    s_team = (ob["t_vec"] + kt * g) / (ob["t_sum"] + kt)
    q = (pl["p_vec"] + kp * s_team) / (pl["p_sum"] + kp)
    q = np.where(ob["banned"], 0.0, q)
    s = q.sum()
    return q / s if s > 0 else np.full(NH, 1.0 / NH)

def assign_six(qs):
    """Max sum log q assignment, distinct heroes, DFS over top-8 candidates."""
    cands = []
    for q in qs:
        idx = np.argsort(-q)[:8]
        cands.append([(int(i), math.log(max(q[i], 1e-9))) for i in idx])
    best = [-1e18, None]
    n = len(cands)
    def dfs(i, used, sc, pick):
        if sc + (n - i) * 0.0 < best[0] - 50: return
        if i == n:
            if sc > best[0]: best[0], best[1] = sc, list(pick)
            return
        for h, lq in cands[i]:
            if h in used: continue
            used.add(h); pick.append(h)
            dfs(i + 1, used, sc + lq, pick)
            used.discard(h); pick.pop()
    dfs(0, set(), 0.0, [])
    return best[1] or []

def score(obs, phase, kp, kt):
    top1 = top2 = nobs = 0; ce = 0.0; jac = 0.0; nteam = 0
    for ob in obs:
        if ob["phase"] != phase: continue
        qs = []
        for pl in ob["players"]:
            q = q_for(pl, ob, kp, kt); qs.append(q)
            a = pl["a"]
            am = int(np.argmax(a)); order = np.argsort(-q)
            top1 += (int(order[0]) == am); top2 += (am in set(int(x) for x in order[:2]))
            ce += float(-(a * np.log(q + 1e-9)).sum()); nobs += 1
        # six-set jaccard: predicted assignment vs actual top-6 team heroes by time
        pred = set(assign_six(qs))
        atot = np.zeros(NH)
        for pl in ob["players"]: atot += pl["a"]
        act = set(int(i) for i in np.argsort(-atot)[:len(ob["players"])] if atot[i] > 0)
        if act:
            jac += len(pred & act) / len(pred | act); nteam += 1
    return {"n": nobs, "top1": top1 / max(nobs, 1), "top2": top2 / max(nobs, 1),
            "ce": ce / max(nobs, 1), "jaccard": jac / max(nteam, 1)}

t0 = time.time()
results = {}
for dp in (0.88, 0.95, 0.99):
    obs = collect(dp, 0.88, 0.995)
    for kp in (2.0, 5.0, 10.0, 20.0):
        for kt in (5.0, 20.0, 50.0):
            r = score(obs, "inner", kp, kt)
            results[(dp, kp, kt)] = (r, obs)
            print(f"inner dp={dp} kp={kp} kt={kt} | top1 {r['top1']*100:.1f}% top2 {r['top2']*100:.1f}% ce {r['ce']:.4f} jac {r['jaccard']:.3f} n={r['n']}", flush=True)
best_key = min(results, key=lambda k: results[k][0]["ce"])
print(f"\nBEST on inner: dp={best_key[0]} kp={best_key[1]} kt={best_key[2]}  [{time.time()-t0:.0f}s]", flush=True)

# baselines on holdout + chosen model, scored ONCE
r_obs = results[best_key][1]
final = score(r_obs, "hold", best_key[1], best_key[2])
print(f"HOLDOUT q-model      | top1 {final['top1']*100:.1f}% top2 {final['top2']*100:.1f}% ce {final['ce']:.4f} jac {final['jaccard']:.3f} n={final['n']}", flush=True)
naive = score(r_obs, "hold", 1e-6, 1e9)   # kp~0: raw player career share (league fill-in for empty)
prior = score(r_obs, "hold", 1e9, 1e-6)   # kp huge: pure team-rate prior
print(f"HOLDOUT raw-career   | top1 {naive['top1']*100:.1f}% top2 {naive['top2']*100:.1f}% ce {naive['ce']:.4f} jac {naive['jaccard']:.3f}", flush=True)
print(f"HOLDOUT team-prior   | top1 {prior['top1']*100:.1f}% top2 {prior['top2']*100:.1f}% ce {prior['ce']:.4f} jac {prior['jaccard']:.3f}", flush=True)

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
