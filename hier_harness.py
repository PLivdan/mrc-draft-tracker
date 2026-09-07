"""Hierarchical sequential draft-policy model (the 'big rewrite').

Per kind (ban/protect), utility for hero h at slot s by team i:
    U = mu_h + delta_{s,h} + beta . X   (X includes the team-deviation state A)
    P(y=h|L) = softmax over the LEGAL set of U/T_s      (T at reference slot = 1)
- mu: persistent hero priority, estimated in-likelihood (recency-weighted obs).
- delta: hero x slot deviations, ridge-shrunk  (== N(0, sigma_s^2) prior).
- A_ih = log((N_ih + k_team p_h)/(N_i + k_team)) - log p_h  (empirical-Bayes team dev).
- Ban X: [cap, thr, map, A].  Protect X: [ownw, map, A, hazard(exact recursion)].
- Revenge dropped (re-ablation leg available via BAN_REV).
Fitting inside the conditional likelihood over actual legal sets handles the
endogenous-availability censoring exactly (the Mantis problem) by construction.
Protocol: inner tuning (last 15 pre-holdout series) for lambda_delta x k_team;
last-45-series holdout scored once, 9 chronological block refits.
Modes: eval (default) | export (full-data fit -> /content/model_params_hier.json)
"""
import json, glob, math, os, sys, time
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize

DRAFT_ROOT = os.environ.get("DRAFT_ROOT", "/content")
t0 = time.time()
roles = json.load(open(DRAFT_ROOT + "/data/roles/heroes.json"))
HEROES = sorted(roles); HIDX = {h: i for i, h in enumerate(HEROES)}; NH = len(HEROES)
DAY = 86400.0
H_ALPHA = 5.0
H_MAP = 30.0
DELTA = 0.88
TAU = 12.0
PATTERNS = [((2,2,2), 0.0), ((2,1,3), -0.60), ((3,1,2), -3.21), ((3,0,3), -3.86)]
H_OBS = 30.0
LAM_MU, LAM_BETA = 0.05, 0.5

recs = []
for f in glob.glob(DRAFT_ROOT + "/data/processed/maps/*.json"):
    r = json.load(open(f))
    if r.get("qa", {}).get("status") == "pass":
        recs.append(r)
recs.sort(key=lambda r: (r["played_at"], r["map_uid"]))
series_order, seen = [], set()
for r in recs:
    if r["match_id"] not in seen:
        seen.add(r["match_id"]); series_order.append(r["match_id"])
HOLD = set(series_order[-45:])
INNER = set(series_order[-60:-45])

SLOTS = {"ban": ["B1", "B2", "B3", "B4"], "protect": ["P1", "P2"]}
SIDX = {k: {sl: i for i, sl in enumerate(v)} for k, v in SLOTS.items()}
HAZ_WIN = {("red","P1"): ("blue", ["B2","B3"]), ("blue","P1"): ("red", ["B2","B3"]),
           ("blue","P2"): ("red", ["B4"]),      ("red","P2"): ("blue", ["B4"])}

# ---------- rolling state ----------
slot_c = defaultdict(lambda: np.zeros(NH)); slot_n = defaultdict(float)
kind_c = {"ban": np.zeros(NH), "protect": np.zeros(NH)}
kind_n = {"ban": 0.0, "protect": 0.0}
g_num = np.zeros(NH); g_den = 0.0
team_use = defaultdict(lambda: np.zeros(NH)); team_n = defaultdict(float)
team_time = defaultdict(lambda: np.zeros(NH)); team_win = defaultdict(lambda: np.zeros(NH))
team_banc = defaultdict(lambda: np.zeros(NH)); team_protc = defaultdict(lambda: np.zeros(NH))
map_c = defaultdict(lambda: np.zeros(NH)); map_n = defaultdict(float)
slot_avail = defaultdict(lambda: np.zeros(NH))
prev_in_series = {}
last_t = None

def time_decay(dt, H): return 0.5 ** (dt / H)
def g_asof(): return (g_num + 0.5) / (g_den + 0.5 * NH)
def cap_asof(team, g): return (team_use[team] + 3.0 * g) / (team_n[team] + 3.0)
def threat_asof(team, g):
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
def prio(kind):
    return (kind_c[kind] + 0.25) / (kind_n[kind] + 0.25 * NH)
def hab_asof(slot):
    kind = "ban" if slot.startswith("B") else "protect"
    pa = prio(kind)
    if slot.startswith("B"):
        return np.log((slot_c[slot] + 4.0 * pa) / (slot_avail[slot] + 4.0))
    return np.log((slot_c[slot] + 4.0 * pa) / (slot_n[slot] + 4.0))

# ---------- feature pass ----------
decisions = []
for r in recs:
    t = r["played_at"]
    if last_t is not None and t > last_t:
        f = time_decay((t - last_t) / DAY, H_ALPHA)
        for k in slot_c: slot_c[k] *= f
        for k in list(slot_n): slot_n[k] *= f
        for k in kind_c: kind_c[k] *= f
        for k in kind_n: kind_n[k] *= f
        g_num *= f; g_den *= f
        for k in slot_avail: slot_avail[k] *= f
        fm = time_decay((t - last_t) / DAY, H_MAP)
        for k in map_c: map_c[k] *= fm
        for k in list(map_n): map_n[k] *= fm
    last_t = t
    g = g_asof()
    tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
    expo_updates = []
    prev = prev_in_series.get(r["match_id"])
    rev_used = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    rev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    own_prev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
    if prev is not None:
        for side in ("blue", "red"):
            opp_nm = tn["red" if side == "blue" else "blue"]; me_nm = tn[side]
            for who, tgt_own in ((opp_nm, False), (me_nm, True)):
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
    thr = {s: threat_asof(tn[s], g) for s in ("blue", "red")}
    pk = {"ban": prio("ban"), "protect": prio("protect")}
    bans = {"blue": [], "red": []}; prots = {"blue": [], "red": []}
    by_phase = defaultdict(list)
    for a in r["actions"]: by_phase[a["phase"]].append(a)
    for ph in sorted(by_phase):
        acts = by_phase[ph]
        for a in acts:
            side, kind, slot, hero = a["side"], a["kind"], a["slot"], a["hero"]
            oside = "red" if side == "blue" else "blue"
            if kind == "ban":
                mask = np.array([HEROES[i] not in bans[side] and HEROES[i] not in prots[oside]
                                 for i in range(NH)])
                F = {"cap": caps[oside].copy(), "thr": thr[oside].copy(), "map": mo.copy(),
                     "revu": rev_used[side].copy(), "revw": rev_won[side].copy()}
                tbc, tbn = team_banc[tn[side]].copy(), float(team_n[tn[side]])
            else:
                mask = np.array([HEROES[i] not in bans[oside] and HEROES[i] not in prots[side]
                                 for i in range(NH)])
                F = {"ownw": own_prev_won[side].copy(), "map": mo.copy()}
                tbc, tbn = team_protc[tn[side]].copy(), float(team_n[tn[side]])
            if not mask[HIDX[hero]]: continue
            look = None
            if kind == "protect" and (side, slot) in HAZ_WIN:
                oo, oslots = HAZ_WIN[(side, slot)]
                banmask = np.array([HEROES[i] not in bans[oo] and HEROES[i] not in prots[side]
                                    for i in range(NH)])
                look = {"slots": oslots, "mask": banmask,
                        "F": {"cap": caps[side].copy(), "thr": thr[side].copy(), "map": mo.copy(),
                              "revu": rev_used[oo].copy(), "revw": rev_won[oo].copy()},
                        "tbc": team_banc[tn[oo]].copy(), "tbn": float(team_n[tn[oo]]),
                        "pk": pk["ban"].copy()}
            if look is not None:
                look["habs"] = {osl: hab_asof(osl) for osl in look["slots"]}
            top_meta = int(np.argmax(kind_c["ban"]))
            surprise = bool(mask[top_meta]) and not (kind == "ban" and slot == "B1")
            decisions.append({"kind": kind, "slot": slot, "series": r["match_id"], "t": t,
                              "mask": mask, "yg": HIDX[hero], "F": F, "hab": hab_asof(slot),
                              "tbc": tbc, "tbn": tbn, "pk": pk[kind].copy(),
                              "surprise": surprise, "look": look})
            expo_updates.append((slot, np.where(mask)[0]))
        for a in acts:
            (bans if a["kind"] == "ban" else prots)[a["side"]].append(a["hero"])
    prev_in_series[r["match_id"]] = r
    for a in r["actions"]:
        i = HIDX[a["hero"]]
        slot_c[a["slot"]][i] += 1; slot_n[a["slot"]] += 1
        kind_c[a["kind"]][i] += 1; kind_n[a["kind"]] += 1
        tm = tn[a["side"]]
        if a["kind"] == "ban": team_banc[tm][i] += 1
        else: team_protc[tm][i] += 1
    for _sl, _lg in expo_updates: slot_avail[_sl][_lg] += 1
    mp = r.get("map_name")
    for side in ("blue", "red"):
        team = tn[side]
        won = r.get("winner_side") == side
        team_use[team] *= DELTA; team_win[team] *= DELTA
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

BAN_F = ["cap", "thr", "map"]          # + A appended as final column
BAN_REV = ["cap", "thr", "revu", "revw", "map"]
PROT_F = ["ownw", "map"]               # + A, + haz appended

def a_state(tbc, tbn, pkv, kt):
    return np.log((tbc + kt * pkv) / (tbn + kt)) - np.log(pkv)

def build(rows, names, kt, cut, extra=None):
    n = len(rows); S = len(SLOTS[rows[0]["kind"]])
    nf = len(names) + 2 + (1 if extra is not None else 0)
    X = np.zeros((n, NH, nf)); MASK = np.zeros((n, NH), bool)
    Y = np.zeros(n, int); SL = np.zeros(n, int); W = np.zeros(n)
    sidx = SIDX[rows[0]["kind"]]
    for k, d in enumerate(rows):
        for j, nm in enumerate(names): X[k, :, j] = d["F"][nm]
        X[k, :, len(names)] = d["hab"]
        X[k, :, len(names) + 1] = a_state(d["tbc"], d["tbn"], d["pk"], kt)
        if extra is not None: X[k, :, -1] = extra[k]
        MASK[k] = d["mask"]; Y[k] = d["yg"]; SL[k] = sidx[d["slot"]]
        W[k] = 0.5 ** ((cut - d["t"]) / DAY / H_OBS)
    return X, MASK, Y, SL, W, S, nf

def fit(X, MASK, Y, SL, W, S, nf, lam_d):
    """Two-stage: (1) convex fit of mu/delta/beta at T=1; (2) temps alone, frozen rest."""
    n = X.shape[0]
    n_mu, n_d = NH, S * NH
    onehot = np.zeros((n, NH)); onehot[np.arange(n), Y] = 1.0
    def core(mu, dl, be, lT):
        T = np.exp(lT)[SL]
        U = mu[None, :] + dl[SL] + X @ be
        Us = np.where(MASK, U / T[:, None], -np.inf)
        m = Us.max(1, keepdims=True)
        E = np.exp(Us - m); Z = E.sum(1)
        P = E / Z[:, None]
        nll = float((W * (np.log(Z) + m[:, 0] - Us[np.arange(n), Y])).sum())
        return nll, P, U, T
    def obj1(th):
        mu = th[:n_mu]; dl = th[n_mu:n_mu+n_d].reshape(S, NH); be = th[n_mu+n_d:]
        nll, P, U, T = core(mu, dl, be, np.zeros(S))
        G = (P - onehot) * W[:, None]
        g_mu = G.sum(0)
        g_dl = np.zeros((S, NH)); np.add.at(g_dl, SL, G)
        g_be = np.einsum("nh,nhf->f", G, X)
        nll += LAM_MU * float((mu**2).sum()) + lam_d * float((dl**2).sum()) + LAM_BETA * float((be**2).sum())
        g_mu += 2*LAM_MU*mu; g_dl += 2*lam_d*dl; g_be += 2*LAM_BETA*be
        return nll, np.concatenate([g_mu, g_dl.ravel(), g_be])
    r1 = minimize(obj1, np.zeros(n_mu + n_d + nf), jac=True, method="L-BFGS-B",
                  options={"maxiter": 3000, "maxfun": 6000})
    mu = r1.x[:n_mu]; dl = r1.x[n_mu:n_mu+n_d].reshape(S, NH); be = r1.x[n_mu+n_d:]
    def obj2(thT):
        lT = np.concatenate([[0.0], thT])
        nll, P, U, T = core(mu, dl, be, lT)
        uy = U[np.arange(n), Y]
        pu = (P * np.where(MASK, U, 0.0)).sum(1)
        g_lT = np.zeros(S); np.add.at(g_lT, SL, W * (uy - pu) / T)
        nll += 1.0 * float((lT**2).sum())
        return nll, g_lT[1:] + 2.0 * lT[1:]
    r2 = minimize(obj2, np.zeros(S - 1), jac=True, method="L-BFGS-B",
                  options={"maxiter": 200})
    lT = np.concatenate([[0.0], r2.x])
    return mu, dl, be, lT

def haz_exact(d, mu, dl, be, lT, kt, names):
    lk = d["look"]
    A = a_state(lk["tbc"], lk["tbn"], lk["pk"], kt)
    xb = np.zeros(NH)
    for j, nm in enumerate(names): xb += be[j] * lk["F"].get(nm, np.zeros(NH))
    xb += be[len(names) + 1] * A
    sidx = SIDX["ban"]; mask = lk["mask"]
    def pvec(osl):
        T = math.exp(lT[sidx[osl]])
        u = np.where(mask, (mu + dl[sidx[osl]] + be[len(names)] * lk["habs"][osl] + xb) / T, -np.inf)
        m = u[mask].max() if mask.any() else 0.0
        e = np.exp(u - m); return e / e.sum(), e
    slots = lk["slots"]
    p1, _ = pvec(slots[0])
    if len(slots) == 1: return p1
    _, e2 = pvec(slots[1])
    Z2 = e2.sum()
    den = np.maximum(Z2 - e2, 1e-12)
    Ssum = float((p1 / den).sum())
    H = p1 + e2 * (Ssum - p1 / den)
    return np.clip(H, 0.0, 1.0)

def eval_block(rows_tr_b, rows_te_b, rows_tr_p, rows_te_p, cut, lam_d, kt, names_b):
    Xb, Mb, Yb, Sb, Wb, S, nf = build(rows_tr_b, names_b, kt, cut)
    mu, dl, be, lT = fit(Xb, Mb, Yb, Sb, Wb, S, nf, lam_d)
    hz_tr = [haz_exact(d, mu, dl, be, lT, kt, names_b) if d["look"] else np.zeros(NH) for d in rows_tr_p]
    hz_te = [haz_exact(d, mu, dl, be, lT, kt, names_b) if d["look"] else np.zeros(NH) for d in rows_te_p]
    Xp, Mp, Yp, Sp, Wp, S2, nf2 = build(rows_tr_p, PROT_F, kt, cut, extra=hz_tr)
    mu2, dl2, be2, lT2 = fit(Xp, Mp, Yp, Sp, Wp, S2, nf2, lam_d)
    out = []
    for rows_te, prm, names, extra in ((rows_te_b, (mu, dl, be, lT), names_b, None),
                                       (rows_te_p, (mu2, dl2, be2, lT2), PROT_F, hz_te)):
        m_, d_, b_, t_ = prm; sidx = SIDX[rows_te[0]["kind"]] if rows_te else None
        for k, d in enumerate(rows_te):
            xb = np.zeros(NH)
            for j, nm in enumerate(names): xb += b_[j] * d["F"][nm]
            xb += b_[len(names)] * d["hab"]
            xb += b_[len(names) + 1] * a_state(d["tbc"], d["tbn"], d["pk"], kt)
            if extra is not None: xb += b_[-1] * extra[k]
            T = math.exp(t_[sidx[d["slot"]]])
            u = np.where(d["mask"], (m_ + d_[sidx[d["slot"]]] + xb) / T, -np.inf)
            mm = u[d["mask"]].max()
            e = np.exp(u - mm); p = e / e.sum()
            rank = int((p > p[d["yg"]]).sum()) + 1
            out.append((d["kind"], d["slot"], d["surprise"],
                        -math.log(max(p[d["yg"]], 1e-12)), rank))
    return out

def run_eval(block_sets, label, lam_d, kt, names_b=BAN_F):
    res = {sc: {k: [0.0, 0, 0, 0] for k in ("ban", "protect")} for sc in ("all", "surprise")}
    per_slot = defaultdict(lambda: [0.0, 0])
    t_run = time.time()
    for bi, blk in enumerate(block_sets):
        cut = min(d["t"] for d in decisions if d["series"] in blk)
        tr_b = [d for d in decisions if d["kind"]=="ban" and d["t"] < cut]
        tr_p = [d for d in decisions if d["kind"]=="protect" and d["t"] < cut]
        te_b = [d for d in decisions if d["kind"]=="ban" and d["series"] in blk]
        te_p = [d for d in decisions if d["kind"]=="protect" and d["series"] in blk]
        for kind, slot, sup, nll, rank in eval_block(tr_b, te_b, tr_p, te_p, cut, lam_d, kt, names_b):
            for sc in (["all", "surprise"] if sup else ["all"]):
                a = res[sc][kind]; a[0] += nll; a[1] += 1; a[2] += rank == 1; a[3] += rank <= 3
            if kind == "ban":
                ps = per_slot[slot]; ps[0] += nll; ps[1] += 1
        eta = (time.time()-t_run)/(bi+1)*(len(block_sets)-bi-1)
        print(f"   [{label}] block {bi+1}/{len(block_sets)} ETA {eta:.0f}s", flush=True)
    for sc in ("all", "surprise"):
        line = (label + " " + sc).ljust(30)
        for kind in ("ban", "protect"):
            nl, n, t1, t3 = res[sc][kind]
            if n: line += f" | {kind}: ll {nl/n:.4f} top1 {100*t1/n:.1f}% top3 {100*t3/n:.1f}% n={n}"
        print(line, flush=True)
    print("   per-slot ban ll: " + " ".join(f"{sl} {v[0]/v[1]:.3f}(n={v[1]})" for sl, v in sorted(per_slot.items())), flush=True)
    return res

mode = sys.argv[1] if len(sys.argv) > 1 else "eval"
hold_series = [s for s in series_order if s in HOLD]
BLOCKS = [set(hold_series[i:i+5]) for i in range(0, len(hold_series), 5)]

if mode == "eval":
    print("--- inner tuning (last 15 pre-holdout series, one shot each) ---", flush=True)
    inner_blk = [set(s for s in series_order if s in INNER)]
    best = (None, 1e18)
    for hob in (10.0, 30.0):
        for lam_d in (2.0, 8.0):
            for kt in (8.0, 32.0):
                globals()["H_OBS"] = hob
                r = run_eval(inner_blk, f"inner H={hob:.0f} ld={lam_d} kt={kt}", lam_d, kt)
                comp = r["all"]["ban"][0]/max(r["all"]["ban"][1],1) + r["all"]["protect"][0]/max(r["all"]["protect"][1],1)
                if comp < best[1]: best = ((hob, lam_d, kt), comp)
    hob, lam_d, kt = best[0]
    globals()["H_OBS"] = hob
    print(f"\nBEST inner: H_OBS={hob} lam_delta={lam_d} k_team={kt}", flush=True)
    run_eval(BLOCKS, f"HIER holdout H={hob:.0f} ld={lam_d} kt={kt}", lam_d, kt)
    run_eval(BLOCKS, f"HIER+revenge holdout", lam_d, kt, names_b=BAN_REV)
    print("HIER_DONE", flush=True)
elif mode == "export":
    lam_d = float(sys.argv[2]); kt = float(sys.argv[3])
    if len(sys.argv) > 4: globals()["H_OBS"] = float(sys.argv[4])
    END = max(d["t"] for d in decisions)
    tr_b = [d for d in decisions if d["kind"] == "ban"]
    tr_p = [d for d in decisions if d["kind"] == "protect"]
    Xb, Mb, Yb, Sb, Wb, S, nf = build(tr_b, BAN_F, kt, END)
    mu, dl, be, lT = fit(Xb, Mb, Yb, Sb, Wb, S, nf, lam_d)
    hz = [haz_exact(d, mu, dl, be, lT, kt, BAN_F) if d["look"] else np.zeros(NH) for d in tr_p]
    Xp, Mp, Yp, Sp, Wp, S2, nf2 = build(tr_p, PROT_F, kt, END, extra=hz)
    mu2, dl2, be2, lT2 = fit(Xp, Mp, Yp, Sp, Wp, S2, nf2, lam_d)
    print("ban beta [cap,thr,map,hab,A]:", [round(float(x),4) for x in be],
          "T:", [round(math.exp(x),3) for x in lT], flush=True)
    print("prot beta [ownw,map,hab,A,haz]:", [round(float(x),4) for x in be2],
          "T:", [round(math.exp(x),3) for x in lT2], flush=True)
    g = g_asof(); logg = np.log(g)
    teams_out = {}
    pkb, pkp = prio("ban"), prio("protect")
    for team in sorted(team_n):
        if team_n[team] <= 0: continue
        cap = cap_asof(team, g); th_ = threat_asof(team, g); ls = ls_asof(team)
        Ab = a_state(team_banc[team], team_n[team], pkb, kt)
        Ap = a_state(team_protc[team], team_n[team], pkp, kt)
        teams_out[team] = {"cap": [round(float(x),5) for x in cap],
            "thr": [round(float(x),4) for x in th_], "ls": [round(float(x),5) for x in ls],
            "sb": [round(float(x),5) for x in Ab], "sp": [round(float(x),5) for x in Ap],
            "v": [round(float(x),4) for x in (logg + 0.76 * cap)]}
    alphas = {}
    for sl in SLOTS["ban"]:
        alphas[sl] = [round(float(x),4) for x in (mu + dl[SIDX["ban"][sl]] + be[len(BAN_F)] * hab_asof(sl))]
    for sl in SLOTS["protect"]:
        alphas[sl] = [round(float(x),4) for x in (mu2 + dl2[SIDX["protect"][sl]] + be2[len(PROT_F)] * hab_asof(sl))]
    coef = {"ban": {sl: {"cap": round(float(be[0]),4), "thr": round(float(be[1]),4),
                          "revu": 0.0, "revw": 0.0, "map": round(float(be[2]),4),
                          "selfban": round(float(be[4]),4), "den": 0.0} for sl in SLOTS["ban"]},
            "protect": {sl: {"cap": 0.0, "ls": 0.0, "ownw": round(float(be2[0]),4),
                              "map": round(float(be2[1]),4), "selfprot": round(float(be2[3]),4),
                              "haz": round(float(be2[4]),4)} for sl in SLOTS["protect"]}}
    out = {"fitted_on": "868 maps; hierarchical v3: in-likelihood hero priorities + shrunk hero-x-slot deviations + team-deviation states, exact hazard recursion, revenge/denial removed",
        "tau": TAU, "patterns": [[list(c), r] for c, r in PATTERNS],
        "coef": coef,
        "temps": {"ban": {sl: round(math.exp(lT[SIDX["ban"][sl]]),3) for sl in SLOTS["ban"]},
                   "protect": {sl: round(math.exp(lT2[SIDX["protect"][sl]]),3) for sl in SLOTS["protect"]}},
        "mix": {"rho": 0.9, "w_ban": 1.0},
        "heroes": HEROES, "roles": {h: roles[h] for h in HEROES},
        "alpha": alphas, "alpha_slow": alphas,
        "g": [round(float(x),6) for x in g],
        "map_offsets": {mp: [round(float(x),3) for x in mapoff_asof(mp)] for mp in map_n},
        "teams": teams_out, "default_v": [round(float(x),4) for x in logg]}
    path = "/content/model_params_hier.json" if os.path.isdir("/content") else "model_params_hier.json"
    json.dump(out, open(path, "w"))
    print("EXPORT_DONE bytes:", len(json.dumps(out)), flush=True)
