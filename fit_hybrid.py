import os
DRAFT_ROOT = os.environ.get("DRAFT_ROOT", "/Users/dlivdan/projects/marvel-draft-model")
"""Chronological variant evaluation for the tracker model.
One as-of feature pass over all decisions, then per-variant weighted conditional
logit fits with block refits; holdout = last 45 series (July-Aug incl. MSF)."""
import json, glob, math, time
import numpy as np
from collections import defaultdict

t0 = time.time()
roles = json.load(open(DRAFT_ROOT + "/data/roles/heroes.json"))
HEROES = sorted(roles); HIDX = {h: i for i, h in enumerate(HEROES)}
NH = len(HEROES)
ROLE_ORDER = ["Vanguard", "Duelist", "Strategist"]
PATTERNS = [((2,2,2), 0.0), ((2,1,3), -0.60), ((3,1,2), -3.21), ((3,0,3), -3.86)]
NMAX = {r: max(c[i] for c,_ in PATTERNS) for i, r in enumerate(ROLE_ORDER)}
TAU = 12.0
DAY = 86400.0
H_ALPHA = 5.0       # habit half-life (validated)
H_MAP = 30.0
DELTA = 0.88        # per-team-map decay (spec)

recs = []
for f in glob.glob(DRAFT_ROOT + "/data/processed/maps/*.json"):
    r = json.load(open(f))
    if r.get("qa", {}).get("status") == "pass":
        recs.append(r)
recs.sort(key=lambda r: (r["played_at"], r["map_uid"]))
series_order = []
seen = set()
for r in recs:
    if r["match_id"] not in seen:
        seen.add(r["match_id"]); series_order.append(r["match_id"])

# ---------- rolling state ----------
slot_c = defaultdict(lambda: np.zeros(NH)); slot_n = defaultdict(float)
kind_c = {"ban": np.zeros(NH), "protect": np.zeros(NH)}
kind_n = {"ban": 0.0, "protect": 0.0}
g_num = np.zeros(NH); g_den = 0.0
last_t = None
team_use = defaultdict(lambda: np.zeros(NH))   # delta-decayed appearance
team_n = defaultdict(float)
team_time = defaultdict(lambda: np.zeros(NH))  # delta-decayed playtime share basis
team_win = defaultdict(lambda: np.zeros(NH))   # delta-decayed wins-with
map_c = defaultdict(lambda: np.zeros(NH)); map_n = defaultdict(float)
H_SLOW = 60.0
RHOS = (0.90, 0.97, 0.99)
slot_cS = defaultdict(lambda: np.zeros(NH)); slot_nS = defaultdict(float)
kind_cS = {"ban": np.zeros(NH), "protect": np.zeros(NH)}
kind_nS = {"ban": 0.0, "protect": 0.0}
EVID = {rho: {"ban": np.zeros(2), "protect": np.zeros(2)} for rho in RHOS}
team_banc = defaultdict(lambda: np.zeros(NH))
team_protc = defaultdict(lambda: np.zeros(NH))
team_banc = defaultdict(lambda: np.zeros(NH))
team_protc = defaultdict(lambda: np.zeros(NH))
prev_in_series = {}                            # match_id -> last map record

def time_decay(dt_days, H): return 0.5 ** (dt_days / H)

def alpha_asof(slot):
    kind = "ban" if slot.startswith("B") else "protect"
    tot = kind_n[kind] + 0.25 * NH
    pa = (kind_c[kind] + 0.25) / tot
    p = (slot_c[slot] + 4.0 * pa) / (slot_n[slot] + 4.0)
    return np.log(p)

def alpha_asof_slow(slot):
    kind = "ban" if slot.startswith("B") else "protect"
    tot = kind_nS[kind] + 0.25 * NH
    pa = (kind_cS[kind] + 0.25) / tot
    p = (slot_cS[slot] + 4.0 * pa) / (slot_nS[slot] + 4.0)
    return np.log(p)

def g_asof():
    return (g_num + 0.5) / (g_den + 0.5 * NH)

def cap_asof(team, g):
    n = team_n[team]
    return (team_use[team] + 3.0 * g) / (n + 3.0)

def threat_asof(team, g):
    # decayed winrate-with-hero, shrunk to 0.5, times sqrt of exposure
    u = team_use[team]; w = team_win[team]
    wr = (w + 1.0) / (u + 2.0)
    return (wr - 0.5) * np.sqrt(np.minimum(u, 6.0) / 6.0)

def ls_asof(team):
    t = team_time[team]; s = t.sum()
    return t / s if s > 0 else np.zeros(NH)

def mapoff_asof(mapname):
    tot_c = sum(map_c.values(), np.zeros(NH)); tot_n = sum(map_n.values())
    if tot_n < 1 or map_n.get(mapname, 0) < 1: return np.zeros(NH)
    gref = (tot_c + 0.5) / (tot_n + 0.5 * NH)
    gm = (map_c[mapname] + 25.0 * gref) / (map_n[mapname] + 25.0)
    return np.clip(np.log(gm / gref), -1.5, 1.5)

# denial over as-of team v
def soft_val(vv, mask):
    dp = {}
    for ro in ROLE_ORDER:
        arr = np.full(NMAX[ro] + 1, -np.inf); arr[0] = 0.0
        for i in range(NH):
            if mask[i] and roles[HEROES[i]] == ro:
                for k in range(NMAX[ro], 0, -1):
                    arr[k] = np.logaddexp(arr[k], arr[k-1] + vv[i] / TAU)
        dp[ro] = arr
    zs = []
    for c, rho in PATTERNS:
        z = rho / TAU; ok = True
        for i, ro in enumerate(ROLE_ORDER):
            if not np.isfinite(dp[ro][c[i]]): ok = False; break
            z += dp[ro][c[i]]
        if ok and np.isfinite(z): zs.append(z)
    if not zs: return -np.inf
    a = np.array(zs); m = a.max()
    return TAU * (m + math.log(np.exp(a - m).sum()))

def denial_asof(vv, banned_by_actor):
    mask = np.ones(NH, bool)
    for h in banned_by_actor: mask[HIDX[h]] = False
    full = soft_val(vv, mask)
    out = np.zeros(NH)
    if not np.isfinite(full): return out
    mx = 0.0
    for i in range(NH):
        if not mask[i]: continue
        m2 = mask.copy(); m2[i] = False
        d = full - soft_val(vv, m2)
        out[i] = d if np.isfinite(d) else np.inf
        if np.isfinite(d) and d > mx: mx = d
    out[np.isinf(out)] = mx
    return np.clip(out, 0, None)

# ---------- feature pass ----------
decisions = []   # dict: kind, slot, series, t, legal(idx), y(idx in legal), feats {name: vec over legal}
dcache = {}
for ri, r in enumerate(recs):
    t = r["played_at"]
    if last_t is not None and t > last_t:
        f = time_decay((t - last_t) / DAY, H_ALPHA)
        for k in slot_c: slot_c[k] *= f
        for k in list(slot_n): slot_n[k] *= f
        for k in kind_c: kind_c[k] *= f
        for k in kind_n: kind_n[k] *= f
        fS = time_decay((t - last_t) / DAY, H_SLOW)
        for k in slot_cS: slot_cS[k] *= fS
        for k in list(slot_nS): slot_nS[k] *= fS
        for k in kind_cS: kind_cS[k] *= fS
        for k in kind_nS: kind_nS[k] *= fS
        g_num *= f; g_den *= f
        fm = time_decay((t - last_t) / DAY, H_MAP)
        for k in map_c: map_c[k] *= fm
        for k in list(map_n): map_n[k] *= fm
    last_t = t
    g = g_asof()
    tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
    prev = prev_in_series.get(r["match_id"])
    rev_used = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    rev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    own_prev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    if prev is not None:
        for side in ("blue", "red"):
            opp = tn["red" if side == "blue" else "blue"]
            me = tn[side]
            for who, tgt_used, tgt_won, tgt_own in ((opp, True, True, False), (me, False, False, True)):
                ps = next((s for s in ("blue","red") if prev["teams"][s]["name"] == who), None)
                if ps is None: continue
                used = set()
                for pid in [p["player_id"] for p in prev["lineups"][ps]]:
                    used |= set(prev["hero_time"].get(pid, {}))
                won = prev.get("winner_side") == ps
                for h in used:
                    if h not in HIDX: continue
                    if tgt_own:
                        if won: own_prev_won[side][HIDX[h]] = 1.0
                    else:
                        rev_used[side][HIDX[h]] = 1.0
                        if won: rev_won[side][HIDX[h]] = 1.0
    mo = mapoff_asof(r.get("map_name"))
    caps = {s: cap_asof(tn[s], g) for s in ("blue", "red")}
    selfb = {s: team_banc[tn[s]] / (team_n[tn[s]] + 1.0) for s in ("blue", "red")}
    selfp = {s: team_protc[tn[s]] / (team_n[tn[s]] + 1.0) for s in ("blue", "red")}
    selfb = {s: team_banc[tn[s]] / (team_n[tn[s]] + 1.0) for s in ("blue", "red")}
    selfp = {s: team_protc[tn[s]] / (team_n[tn[s]] + 1.0) for s in ("blue", "red")}
    thr = {s: threat_asof(tn[s], g) for s in ("blue", "red")}
    lss = {s: ls_asof(tn[s]) for s in ("blue", "red")}
    logg = np.log(g)
    vteam = {s: logg + 0.76 * caps[s] for s in ("blue", "red")}   # v proxy, fixed shape
    bans = {"blue": [], "red": []}; prots = {"blue": [], "red": []}
    by_phase = defaultdict(list)
    for a in r["actions"]: by_phase[a["phase"]].append(a)
    for ph in sorted(by_phase):
        acts = by_phase[ph]
        for a in acts:
            side, kind, slot, hero = a["side"], a["kind"], a["slot"], a["hero"]
            oside = "red" if side == "blue" else "blue"
            if kind == "ban":
                legal = [i for i in range(NH)
                         if HEROES[i] not in bans[side] and HEROES[i] not in prots[oside]]
                key = (tn[oside], tuple(sorted(bans[side])), round(float(vteam[oside].sum()), 3))
                if key not in dcache:
                    dcache[key] = denial_asof(vteam[oside], bans[side])
                D = dcache[key]
                feats = {"cap": caps[oside], "thr": thr[oside],
                         "revu": rev_used[side], "revw": rev_won[side],
                         "map": mo, "den": D, "selfban": selfb[side]}
            else:
                legal = [i for i in range(NH)
                         if HEROES[i] not in bans[oside] and HEROES[i] not in prots[side]]
                feats = {"cap": caps[side], "ls": lss[side],
                         "ownw": own_prev_won[side], "map": mo,
                         "selfprot": selfp[side]}
            if HIDX[hero] not in legal: continue
            # lookahead windows for anticipation (schedule-aware)
            HAZ_WIN = {("red","P1"): ("blue", ["B2","B3"]), ("blue","P1"): ("red", ["B2","B3"]),
                       ("blue","P2"): ("red", ["B4"]),      ("red","P2"): ("blue", ["B4"])}
            URG_WIN = {("blue","B1"): ("red", ["P1"]), ("red","B1"): ("blue", ["P1"]),
                       ("blue","B3"): ("red", ["P2"]), ("red","B3"): ("blue", ["P2"])}
            look = None
            if kind == "protect" and (side, slot) in HAZ_WIN:
                oo, oslots = HAZ_WIN[(side, slot)]
                banmask = np.array([HEROES[i] not in bans[oo] and HEROES[i] not in prots[side]
                                    for i in range(NH)])
                look = {"slots": oslots, "mask": banmask,
                        "alpha": {osl: alpha_asof(osl) for osl in oslots},
                        "cap": caps[side], "thr": thr[side],
                        "revu": rev_used[oo], "revw": rev_won[oo],
                        "selfban": selfb[oo], "map": mo}
            if kind == "ban" and (side, slot) in URG_WIN:
                oo, oslots = URG_WIN[(side, slot)]
                pmask = np.array([HEROES[i] not in bans[side] and HEROES[i] not in prots[oo]
                                  for i in range(NH)])
                look = {"slots": oslots, "mask": pmask,
                        "alpha": {osl: alpha_asof(osl) for osl in oslots},
                        "cap": caps[oo], "ls": lss[oo],
                        "ownw": own_prev_won[oo], "selfprot": selfp[oo], "map": mo}
            aFull = alpha_asof(slot); aSFull = alpha_asof_slow(slot)
            yy0 = legal.index(HIDX[hero])
            wmap = {}
            for rho in RHOS:
                e = EVID[rho][kind]; m0 = e.max()
                wmap[rho] = float(np.exp(e[0]-m0)/(np.exp(e[0]-m0)+np.exp(e[1]-m0)))
            def _ll(a):
                al = a[legal]; m0 = al.max()
                return float(al[yy0] - (m0 + math.log(np.exp(al-m0).sum())))
            llF, llS = _ll(aFull), _ll(aSFull)
            for rho in RHOS:
                EVID[rho][kind] = rho * EVID[rho][kind] + np.array([llF, llS])
            decisions.append({
                "kind": kind, "slot": slot, "series": r["match_id"], "t": t,
                "alpha_slow": aSFull[legal], "w": wmap,
                "alpha": alpha_asof(slot)[legal],
                "feats": {k: v[legal] for k, v in feats.items()},
                "y": legal.index(HIDX[hero]),
                "legal": np.array(legal), "look": look,
            })
        for a in acts:
            (bans if a["kind"] == "ban" else prots)[a["side"]].append(a["hero"])
    prev_in_series[r["match_id"]] = r
    # ---- post-update rolling profiles with this map ----
    for a in r["actions"]:
        i = HIDX[a["hero"]]
        slot_c[a["slot"]][i] += 1; slot_n[a["slot"]] += 1
        kind_c[a["kind"]][i] += 1; kind_n[a["kind"]] += 1
        tm = tn[a["side"]]
        if a["kind"] == "ban": team_banc[tm][i] += 1
        else: team_protc[tm][i] += 1
        slot_cS[a["slot"]][i] += 1; slot_nS[a["slot"]] += 1
        kind_cS[a["kind"]][i] += 1; kind_nS[a["kind"]] += 1
        tm = tn[a["side"]]
        if a["kind"] == "ban": team_banc[tm][i] += 1
        else: team_protc[tm][i] += 1
        slot_cS[a["slot"]][i] += 1; slot_nS[a["slot"]] += 1
        kind_cS[a["kind"]][i] += 1; kind_nS[a["kind"]] += 1
    mp = r.get("map_name")
    for side in ("blue", "red"):
        team = tn[side]
        won = r.get("winner_side") == side
        team_use[team] *= DELTA; team_win[team] *= DELTA
        team_banc[team] *= DELTA; team_protc[team] *= DELTA
        team_banc[team] *= DELTA; team_protc[team] *= DELTA
        team_time[team] *= DELTA; team_n[team] = team_n[team] * DELTA + 1
        credit = {}
        for pid in [p["player_id"] for p in r["lineups"][side]]:
            shares = r["hero_time"].get(pid, {})
            ptot = sum(shares.values())
            for h, s in shares.items():
                if h in HIDX:
                    team_time[team][HIDX[h]] += s
                    if ptot > 0:
                        credit[h] = max(credit.get(h, 0.0), min(1.0, s / ptot))
        for h, cr in credit.items():
            team_use[team][HIDX[h]] += cr
            if won: team_win[team][HIDX[h]] += cr
        g_den += 1
        for h, cr in credit.items(): g_num[HIDX[h]] += cr
        if mp:
            map_n[mp] += 1
            for h, cr in credit.items(): map_c[mp][HIDX[h]] += cr
print(f"feature pass done: {len(decisions)} decisions, {time.time()-t0:.0f}s", flush=True)




# ---------- production fit: hybrid winner (bans: mix rho=0.90; protects: hazard) ----------
from scipy.optimize import minimize
END = max(d["t"] for d in decisions)
H_OBS = 30.0
BAN_F = ["cap","thr","revu","revw","map","den","selfban"]
PROT_F = ["cap","ls","ownw","map","selfprot","haz"]
SLOTS = {"ban": ["B1","B2","B3","B4"], "protect": ["P1","P2"]}
RHO_BAN = 0.90

def habit(d, rho):
    if rho is None: return d["alpha"]
    w = d["w"][rho]
    return np.log(w * np.exp(d["alpha"]) + (1.0 - w) * np.exp(d["alpha_slow"]))

def fit_kind(kind, names, rows, rho):
    slots = SLOTS[kind]; sidx = {sl:k for k,sl in enumerate(slots)}
    nf, nT = len(names), len(slots)
    def nll(th):
        w, logT = th[:nf], th[nf:]
        tot = 0.0
        for d in rows:
            wt = 0.5 ** ((END - d["t"]) / DAY / H_OBS)
            T = math.exp(logT[sidx[d["slot"]]])
            u = habit(d, rho).copy()
            for j, nm in enumerate(names):
                u = u + w[j] * (d["feats"][nm] if nm in d["feats"] else d["_x"][nm])
            u = u / T
            m = u.max()
            tot += wt * (m + math.log(np.exp(u - m).sum()) - u[d["y"]])
        tot += 0.5 * float((w ** 2).sum())   # ridge: stabilize collinear features
        return tot
    th = minimize(nll, np.zeros(nf + nT), method="L-BFGS-B").x
    return th[:nf], th[nf:], sidx

def lookahead_probs(d, names, w, logT, sidx, rho):
    lk = d["look"]; surv = np.ones(NH)
    for osl in lk["slots"]:
        u = lk["alpha"][osl].copy()
        # NOTE: lookahead habit uses the FAST alpha only (page mirrors this)
        for j, nm in enumerate(names):
            if nm in ("den",): continue
            if nm in lk: u = u + w[j] * lk[nm]
        T = math.exp(logT[sidx[osl]])
        u = np.where(lk["mask"], u / T, -np.inf)
        m = u[lk["mask"]].max() if lk["mask"].any() else 0.0
        e = np.exp(u - m); p = e / e.sum()
        surv = surv * (1.0 - p)
    return 1.0 - surv

for d in decisions: d["_cut"] = END; d["_x"] = {}
btr = [d for d in decisions if d["kind"] == "ban"]
ptr = [d for d in decisions if d["kind"] == "protect"]
bw, blT, bsx = fit_kind("ban", BAN_F, btr, RHO_BAN)
for d in ptr:
    hz = lookahead_probs(d, BAN_F, bw, blT, bsx, RHO_BAN)[d["legal"]] if d["look"] else np.zeros(len(d["legal"]))
    d["_x"]["haz"] = hz
pw, plT, psx = fit_kind("protect", PROT_F, ptr, None)
print("ban coefs:", {nm: round(float(bw[j]),4) for j, nm in enumerate(BAN_F)},
      "T:", {sl: round(math.exp(blT[k]),3) for sl,k in bsx.items()}, flush=True)
print("protect coefs:", {nm: round(float(pw[j]),4) for j, nm in enumerate(PROT_F)},
      "T:", {sl: round(math.exp(plT[k]),3) for sl,k in psx.items()}, flush=True)

# export: as-of-END state (fast + slow alphas, current mixture weight)
g = g_asof(); logg = np.log(g)
teams_out = {}
for team in sorted(team_n):
    if team_n[team] <= 0: continue
    cap = cap_asof(team, g); thr = threat_asof(team, g); ls = ls_asof(team)
    sb = team_banc[team] / (team_n[team] + 1.0)
    sp = team_protc[team] / (team_n[team] + 1.0)
    teams_out[team] = {"cap": [round(float(x),5) for x in cap],
        "thr": [round(float(x),4) for x in thr], "ls": [round(float(x),5) for x in ls],
        "sb": [round(float(x),5) for x in sb], "sp": [round(float(x),5) for x in sp],
        "v": [round(float(x),4) for x in (logg + 0.76 * cap)]}
def w_now(kind):
    e = EVID[RHO_BAN][kind]; m0 = e.max()
    return float(np.exp(e[0]-m0)/(np.exp(e[0]-m0)+np.exp(e[1]-m0)))
alphas, alphas_slow = {}, {}
for kind, slots in SLOTS.items():
    for sl in slots:
        alphas[sl] = [round(float(x),4) for x in alpha_asof(sl)]
        alphas_slow[sl] = [round(float(x),4) for x in alpha_asof_slow(sl)]
coef = {"ban": {sl: {nm: round(float(bw[j]),4) for j, nm in enumerate(BAN_F)} for sl in SLOTS["ban"]},
        "protect": {sl: {nm: round(float(pw[j]),4) for j, nm in enumerate(PROT_F)} for sl in SLOTS["protect"]}}
out = {"fitted_on": "868 maps through 2026-08-01; hybrid: time-credited usage, ban habit two-clock mix (rho .90), anticipatory protects (hazard), slot temperatures",
    "tau": TAU, "patterns": [[list(c), r] for c, r in PATTERNS],
    "coef": coef,
    "temps": {"ban": {sl: round(math.exp(blT[k]),3) for sl,k in bsx.items()},
               "protect": {sl: round(math.exp(plT[k]),3) for sl,k in psx.items()}},
    "mix": {"rho": RHO_BAN, "w_ban": round(w_now("ban"), 4)},
    "heroes": HEROES, "roles": {h: roles[h] for h in HEROES},
    "alpha": alphas, "alpha_slow": alphas_slow,
    "g": [round(float(x),6) for x in g],
    "map_offsets": {mp: [round(float(x),3) for x in mapoff_asof(mp)] for mp in map_n},
    "teams": teams_out, "default_v": [round(float(x),4) for x in logg]}
json.dump(out, open("/content/model_params_hybrid.json", "w"))
print("EXPORT_DONE bytes:", len(json.dumps(out)), flush=True)
