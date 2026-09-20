"""Ban/protect model for the tracker: as-of feature pass, hybrid fit, holdout, export.

Numerics follow the hybrid v4 harness (fit_hybrid.py): decayed slot habits with
exposure adjustment, a fast/slow habit mixture for bans, shared feature weights
with per-slot temperatures, and hazard lookahead for protects. Beyond v4 it
handles the five-ban format (slot B5; a P2 hazard window covers B5 only on
five-ban maps), filters heroes by release date, drops the denial DP whose
coefficient was pinned at zero, and can embed its export into index.html.

    python fit_draft.py fit --phi 0.25 --out params.json [--embed index.html]
    python fit_draft.py holdout --phi 0.25 --start 2026-09-17 --variants live,own,share
"""
import argparse, calendar, glob, json, math, os, re, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

DRAFT_ROOT = os.environ.get("DRAFT_ROOT", "/Users/dlivdan/projects/marvel-draft-model")
DAY = 86400.0
H_ALPHA = 5.0       # habit half-life (validated)
H_MAP = 30.0
H_SLOW = 60.0
DELTA = 0.88        # per-team-map decay (spec)
RHOS = (0.90, 0.97, 0.99)
RHO_BAN = 0.90
H_OBS = 30.0
TAU = 12.0
PATTERNS = [((2, 2, 2), 0.0), ((2, 1, 3), -0.60), ((3, 1, 2), -3.21), ((3, 0, 3), -3.86)]
BAN_F = ["cap", "thr", "revu", "revw", "map", "selfban", "bb"]
PROT_F = ["cap", "ls", "ownw", "map", "selfprot", "haz"]
SLOTS = {"ban": ["B1", "B2", "B3", "B4", "B5"], "protect": ["P1", "P2"]}


def _pt(m, d, y=2026):
    return calendar.timegm((y, m, d, 9, 0, 0))


# Major balance patches: habits are discounted by phi when the clock crosses one.
MAJOR_BREAKS = [_pt(5, 13), _pt(7, 2), _pt(7, 11), _pt(8, 6), _pt(8, 19), _pt(9, 11)]


# ---------- draft format ----------
def n_bans_of(rec):
    return max(int(a["slot"][1:]) for a in rec["actions"] if a["kind"] == "ban")


def hazard_window(side, slot, n_bans):
    """Opponent ban slots that fall between this protect and the side's next one."""
    opp = "red" if side == "blue" else "blue"
    if slot == "P1":
        return (opp, ["B2", "B3"])
    if slot == "P2":
        return (opp, ["B4", "B5"] if n_bans == 5 else ["B4"])
    return None


def urgency_window(side, slot):
    """Opponent protect slots that follow this ban before the side bans again."""
    opp = "red" if side == "blue" else "blue"
    if slot == "B1":
        return (opp, ["P1"])
    if slot == "B3":
        return (opp, ["P2"])
    return None


def heroes_as_of(roles, releases, cutoff):
    if cutoff is None:
        return sorted(roles)
    return sorted(h for h in roles if releases.get(h) is None or releases[h] <= cutoff)


def _end_of_day(date):
    y, m, d = (int(x) for x in date.split("-"))
    return calendar.timegm((y, m, d, 0, 0, 0)) + DAY


def load_maps(data_root=DRAFT_ROOT, as_of=None):
    recs, excluded = [], 0
    for f in glob.glob(os.path.join(data_root, "data/processed/maps/*.json")):
        r = json.load(open(f))
        if r.get("qa", {}).get("status") != "pass":
            excluded += 1
            continue
        if as_of is not None and r["played_at"] >= _end_of_day(as_of):
            continue
        recs.append(r)
    recs.sort(key=lambda r: (r["played_at"], r["map_uid"]))
    return recs, excluded


def load_roles(data_root=DRAFT_ROOT):
    return json.load(open(os.path.join(data_root, "data/roles/heroes.json")))


def load_releases(data_root=DRAFT_ROOT):
    p = os.path.join(data_root, "data/roles/releases.json")
    return json.load(open(p)).get("released", {}) if os.path.exists(p) else {}


def time_decay(dt_days, H):
    return 0.5 ** (dt_days / H)


# ---------- rolling as-of state ----------
class AsOfState:
    def __init__(self, heroes):
        self.heroes = list(heroes)
        self.hidx = {h: i for i, h in enumerate(self.heroes)}
        NH = self.NH = len(self.heroes)
        z = lambda: np.zeros(NH)
        self.slot_c = defaultdict(z); self.slot_n = defaultdict(float)
        self.kind_c = {"ban": z(), "protect": z()}
        self.kind_n = {"ban": 0.0, "protect": 0.0}
        self.g_num = z(); self.g_den = 0.0
        self.last_t = None
        self.team_use = defaultdict(z); self.team_n = defaultdict(float)
        self.team_time = defaultdict(z); self.team_win = defaultdict(z)
        self.map_c = defaultdict(z); self.map_n = defaultdict(float)
        self.slot_cS = defaultdict(z); self.slot_nS = defaultdict(float)
        self.kind_cS = {"ban": z(), "protect": z()}
        self.kind_nS = {"ban": 0.0, "protect": 0.0}
        self.EVID = {rho: {"ban": np.zeros(2), "protect": np.zeros(2)} for rho in RHOS}
        self.team_banc = defaultdict(z); self.team_protc = defaultdict(z)
        self.prev_in_series = {}
        self.slot_avail = defaultdict(z)
        self.series_order = []

    def alpha_asof(self, slot):
        kind = "ban" if slot.startswith("B") else "protect"
        tot = self.kind_n[kind] + 0.25 * self.NH
        pa = (self.kind_c[kind] + 0.25) / tot
        p = (self.slot_c[slot] + 4.0 * pa) / (self.slot_avail[slot] + 4.0)
        return np.log(p)

    def alpha_asof_slow(self, slot):
        kind = "ban" if slot.startswith("B") else "protect"
        tot = self.kind_nS[kind] + 0.25 * self.NH
        pa = (self.kind_cS[kind] + 0.25) / tot
        p = (self.slot_cS[slot] + 4.0 * pa) / (self.slot_nS[slot] + 4.0)
        return np.log(p)

    def g_asof(self):
        return (self.g_num + 0.5) / (self.g_den + 0.5 * self.NH)

    def cap_asof(self, team, g):
        return (self.team_use[team] + 3.0 * g) / (self.team_n[team] + 3.0)

    def threat_asof(self, team, g):
        u = self.team_use[team]; w = self.team_win[team]
        wr = (w + 1.0) / (u + 2.0)
        return (wr - 0.5) * np.sqrt(np.minimum(u, 6.0) / 6.0)

    def ls_asof(self, team):
        t = self.team_time[team]; s = t.sum()
        return t / s if s > 0 else np.zeros(self.NH)

    def mapoff_asof(self, mapname):
        tot_c = sum(self.map_c.values(), np.zeros(self.NH)); tot_n = sum(self.map_n.values())
        if tot_n < 1 or self.map_n.get(mapname, 0) < 1:
            return np.zeros(self.NH)
        gref = (tot_c + 0.5) / (tot_n + 0.5 * self.NH)
        gm = (self.map_c[mapname] + 25.0 * gref) / (self.map_n[mapname] + 25.0)
        return np.clip(np.log(gm / gref), -1.5, 1.5)

    def w_now(self, kind, rho=RHO_BAN):
        e = self.EVID[rho][kind]; m0 = e.max()
        return float(np.exp(e[0] - m0) / (np.exp(e[0] - m0) + np.exp(e[1] - m0)))


def feature_pass(recs, heroes, roles, breaks=MAJOR_BREAKS, phi=0.25, slot_map=None,
                 temp_map=None, p2_sees_b5=True, log=False):
    """One chronological pass: emits a decision per draft action with its as-of
    features, then updates the rolling state with the map. `slot_map` re-keys
    slots for habit/exposure (e.g. {"B5": "B4"}); temperatures follow the same
    map unless `temp_map` is given (an empty dict keeps every slot's own
    temperature). `p2_sees_b5=False` reproduces the four-ban hazard windows
    regardless of format."""
    t0 = time.time()
    st = AsOfState(heroes)
    HIDX, NH = st.hidx, st.NH
    slot_map = slot_map or {}
    skey_of = lambda sl: slot_map.get(sl, sl)
    tkey_of = (lambda sl: temp_map.get(sl, sl)) if temp_map is not None else skey_of
    decisions = []
    seen = set()
    for r in recs:
        if r["match_id"] not in seen:
            seen.add(r["match_id"]); st.series_order.append(r["match_id"])
    for r in recs:
        t = r["played_at"]
        if st.last_t is not None and t > st.last_t:
            f = time_decay((t - st.last_t) / DAY, H_ALPHA)
            for k in st.slot_c: st.slot_c[k] *= f
            for k in list(st.slot_n): st.slot_n[k] *= f
            for k in st.kind_c: st.kind_c[k] *= f
            for k in st.kind_n: st.kind_n[k] *= f
            for k in st.slot_avail: st.slot_avail[k] *= f
            fS = time_decay((t - st.last_t) / DAY, H_SLOW)
            for k in st.slot_cS: st.slot_cS[k] *= fS
            for k in list(st.slot_nS): st.slot_nS[k] *= fS
            for k in st.kind_cS: st.kind_cS[k] *= fS
            for k in st.kind_nS: st.kind_nS[k] *= fS
            st.g_num *= f; st.g_den *= f
            for bts in breaks:
                if st.last_t < bts <= t:
                    for k in st.slot_c: st.slot_c[k] *= phi
                    for k in list(st.slot_n): st.slot_n[k] *= phi
                    for k in st.kind_c: st.kind_c[k] *= phi
                    for k in st.kind_n: st.kind_n[k] *= phi
                    for k in st.slot_avail: st.slot_avail[k] *= phi
            fm = time_decay((t - st.last_t) / DAY, H_MAP)
            for k in st.map_c: st.map_c[k] *= fm
            for k in list(st.map_n): st.map_n[k] *= fm
        st.last_t = t
        g = st.g_asof()
        tn = {s: r["teams"][s]["name"] for s in ("blue", "red")}
        nb = n_bans_of(r) if p2_sees_b5 else 4
        expo_updates = []
        prev = st.prev_in_series.get(r["match_id"])
        rev_used = {"blue": np.zeros(NH), "red": np.zeros(NH)}
        rev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
        own_prev_won = {"blue": np.zeros(NH), "red": np.zeros(NH)}
        if prev is not None:
            for side in ("blue", "red"):
                opp = tn["red" if side == "blue" else "blue"]
                me = tn[side]
                for who, tgt_own in ((opp, False), (me, True)):
                    ps = next((s for s in ("blue", "red") if prev["teams"][s]["name"] == who), None)
                    if ps is None:
                        continue
                    used = set()
                    for pid in [p["player_id"] for p in prev["lineups"][ps]]:
                        used |= set(prev["hero_time"].get(pid, {}))
                    won = prev.get("winner_side") == ps
                    for h in used:
                        if h not in HIDX:
                            continue
                        if tgt_own:
                            if won: own_prev_won[side][HIDX[h]] = 1.0
                        else:
                            rev_used[side][HIDX[h]] = 1.0
                            if won: rev_won[side][HIDX[h]] = 1.0
        mo = st.mapoff_asof(r.get("map_name"))
        caps = {s: st.cap_asof(tn[s], g) for s in ("blue", "red")}
        selfb = {s: st.team_banc[tn[s]] / (st.team_n[tn[s]] + 1.0) for s in ("blue", "red")}
        selfp = {s: st.team_protc[tn[s]] / (st.team_n[tn[s]] + 1.0) for s in ("blue", "red")}
        thr = {s: st.threat_asof(tn[s], g) for s in ("blue", "red")}
        lss = {s: st.ls_asof(tn[s]) for s in ("blue", "red")}
        bans = {"blue": [], "red": []}; prots = {"blue": [], "red": []}
        by_phase = defaultdict(list)
        for a in r["actions"]:
            by_phase[a["phase"]].append(a)
        for ph in sorted(by_phase):
            acts = by_phase[ph]
            for a in acts:
                side, kind, slot, hero = a["side"], a["kind"], a["slot"], a["hero"]
                if hero not in HIDX:
                    continue
                oside = "red" if side == "blue" else "blue"
                if kind == "ban":
                    legal = [i for i in range(NH)
                             if heroes[i] not in bans[side] and heroes[i] not in prots[oside]]
                    bbv = np.zeros(NH)
                    for _h in bans[oside]:
                        bbv[HIDX[_h]] = 1.0
                    feats = {"cap": caps[oside], "thr": thr[oside],
                             "revu": rev_used[side], "revw": rev_won[side],
                             "map": mo, "selfban": selfb[side], "bb": bbv}
                else:
                    legal = [i for i in range(NH)
                             if heroes[i] not in bans[oside] and heroes[i] not in prots[side]]
                    feats = {"cap": caps[side], "ls": lss[side],
                             "ownw": own_prev_won[side], "map": mo,
                             "selfprot": selfp[side]}
                if HIDX[hero] not in legal:
                    continue
                look = None
                hw = hazard_window(side, slot, nb) if kind == "protect" else None
                if hw is not None:
                    oo, oslots = hw
                    banmask = np.array([heroes[i] not in bans[oo] and heroes[i] not in prots[side]
                                        for i in range(NH)])
                    _bb2 = np.zeros(NH)
                    for _h in bans[side]:
                        _bb2[HIDX[_h]] = 1.0
                    look = {"slots": oslots, "skeys": [skey_of(s) for s in oslots],
                            "tkeys": [tkey_of(s) for s in oslots], "mask": banmask,
                            "alpha": {osl: st.alpha_asof(skey_of(osl)) for osl in oslots},
                            "cap": caps[side], "thr": thr[side],
                            "revu": rev_used[oo], "revw": rev_won[oo],
                            "selfban": selfb[oo], "map": mo, "bb": _bb2}
                uw = urgency_window(side, slot) if kind == "ban" else None
                if uw is not None:
                    oo, oslots = uw
                    pmask = np.array([heroes[i] not in bans[side] and heroes[i] not in prots[oo]
                                      for i in range(NH)])
                    look = {"slots": oslots, "skeys": list(oslots), "tkeys": list(oslots),
                            "mask": pmask,
                            "alpha": {osl: st.alpha_asof(osl) for osl in oslots},
                            "cap": caps[oo], "ls": lss[oo],
                            "ownw": own_prev_won[oo], "selfprot": selfp[oo], "map": mo}
                skey = skey_of(slot)
                aFull = st.alpha_asof(skey); aSFull = st.alpha_asof_slow(skey)
                yy0 = legal.index(HIDX[hero])
                wmap = {}
                for rho in RHOS:
                    e = st.EVID[rho][kind]; m0 = e.max()
                    wmap[rho] = float(np.exp(e[0] - m0) / (np.exp(e[0] - m0) + np.exp(e[1] - m0)))

                def _ll(a):
                    al = a[legal]; m0 = al.max()
                    return float(al[yy0] - (m0 + math.log(np.exp(al - m0).sum())))
                llF, llS = _ll(aFull), _ll(aSFull)
                for rho in RHOS:
                    st.EVID[rho][kind] = rho * st.EVID[rho][kind] + np.array([llF, llS])
                decisions.append({
                    "kind": kind, "slot": slot, "skey": skey, "tkey": tkey_of(slot), "side": side,
                    "series": r["match_id"], "t": t, "n_bans": nb,
                    "alpha_slow": aSFull[legal], "w": wmap,
                    "alpha": aFull[legal],
                    "feats": {k: v[legal] for k, v in feats.items()},
                    "y": yy0, "legal": np.array(legal), "look": look, "_x": {},
                })
                expo_updates.append((skey, np.array(legal)))
            for a in acts:
                (bans if a["kind"] == "ban" else prots)[a["side"]].append(a["hero"])
        st.prev_in_series[r["match_id"]] = r
        # ---- post-update rolling profiles with this map ----
        pend_ban = defaultdict(lambda: np.zeros(NH)); pend_prot = defaultdict(lambda: np.zeros(NH))
        for a in r["actions"]:
            if a["hero"] not in HIDX:
                continue
            i = HIDX[a["hero"]]
            sk = skey_of(a["slot"])
            st.slot_c[sk][i] += 1; st.slot_n[sk] += 1
            st.kind_c[a["kind"]][i] += 1; st.kind_n[a["kind"]] += 1
            tm = tn[a["side"]]
            if a["kind"] == "ban": pend_ban[tm][i] += 1
            else: pend_prot[tm][i] += 1
            st.slot_cS[sk][i] += 1; st.slot_nS[sk] += 1
            st.kind_cS[a["kind"]][i] += 1; st.kind_nS[a["kind"]] += 1
        for _sl, _lg in expo_updates:
            st.slot_avail[_sl][_lg] += 1
        mp = r.get("map_name")
        for side in ("blue", "red"):
            team = tn[side]
            won = r.get("winner_side") == side
            st.team_use[team] *= DELTA; st.team_win[team] *= DELTA
            st.team_banc[team] *= DELTA; st.team_protc[team] *= DELTA
            st.team_banc[team] += pend_ban[team]; st.team_protc[team] += pend_prot[team]
            st.team_time[team] *= DELTA; st.team_n[team] = st.team_n[team] * DELTA + 1
            credit = {}
            for pid in [p["player_id"] for p in r["lineups"][side]]:
                shares = r["hero_time"].get(pid, {})
                ptot = sum(shares.values())
                for h, s in shares.items():
                    if h in HIDX:
                        st.team_time[team][HIDX[h]] += s
                        if ptot > 0:
                            credit[h] = max(credit.get(h, 0.0), min(1.0, s / ptot))
            for h, cr in credit.items():
                st.team_use[team][HIDX[h]] += cr
                if won: st.team_win[team][HIDX[h]] += cr
            st.g_den += 1
            for h, cr in credit.items(): st.g_num[HIDX[h]] += cr
            if mp:
                st.map_n[mp] += 1
                for h, cr in credit.items(): st.map_c[mp][HIDX[h]] += cr
    if log:
        print(f"feature pass done: {len(decisions)} decisions, {time.time()-t0:.0f}s", flush=True)
    return decisions, st


# ---------- fit ----------
def habit(d, rho):
    if rho is None:
        return d["alpha"]
    w = d["w"][rho]
    return np.log(w * np.exp(d["alpha"]) + (1.0 - w) * np.exp(d["alpha_slow"]))


def temp_of(logT, sidx, skey):
    """Temperature of a slot; a slot the fit never saw (B5 before any five-ban
    data) borrows the last fitted slot's temperature until it has its own."""
    if skey not in sidx:
        skey = max(sidx, key=lambda s: (s[0], int(s[1:])))
    return math.exp(logT[sidx[skey]])


def utilities(d, names, w, logT, sidx, rho):
    T = temp_of(logT, sidx, d["tkey"])
    u = habit(d, rho).copy()
    for j, nm in enumerate(names):
        u = u + w[j] * (d["feats"][nm] if nm in d["feats"] else d["_x"][nm])
    return u / T


def kind_objective(kind, names, rows, rho, end, slots=None, h_obs=H_OBS):
    """Recency-weighted conditional-logit NLL over all rows, vectorised on a
    padded (row x legal) grid; padded cells carry -inf so they drop out of the
    row max and the log-sum-exp exactly as if absent."""
    slots = slots or SLOTS[kind]
    present = {d["tkey"] for d in rows}
    slots = [s for s in slots if s in present]
    sidx = {sl: k for k, sl in enumerate(slots)}
    nf, n = len(names), len(rows)
    L = max(len(d["legal"]) for d in rows)
    H = np.full((n, L), -np.inf); X = np.zeros((n, L, nf))
    y = np.zeros(n, int); si = np.zeros(n, int); wts = np.zeros(n)
    for k, d in enumerate(rows):
        l = len(d["legal"])
        H[k, :l] = habit(d, rho)
        for j, nm in enumerate(names):
            X[k, :l, j] = d["feats"][nm] if nm in d["feats"] else d["_x"][nm]
        y[k] = d["y"]; si[k] = sidx[d["tkey"]]
        wts[k] = 0.5 ** ((end - d["t"]) / DAY / h_obs)
    ar = np.arange(n)

    def nll(th):
        w, logT = th[:nf], th[nf:]
        U = (H + X @ w) / np.exp(logT)[si][:, None]
        m = U.max(axis=1)
        lse = m + np.log(np.exp(U - m[:, None]).sum(axis=1))
        tot = float((wts * (lse - U[ar, y])).sum())
        return tot + 0.5 * float((w ** 2).sum())   # ridge: stabilize collinear features
    return nll, sidx


def fit_kind(kind, names, rows, rho, end, slots=None, h_obs=H_OBS):
    nll, sidx = kind_objective(kind, names, rows, rho, end, slots, h_obs)
    nf = len(names)
    th = minimize(nll, np.zeros(nf + len(sidx)), method="L-BFGS-B").x
    return th[:nf], th[nf:], sidx


def lookahead_probs(d, names, w, logT, sidx, nh):
    lk = d["look"]; surv = np.ones(nh)
    for osl, tk in zip(lk["slots"], lk["tkeys"]):
        u = lk["alpha"][osl].copy()
        # lookahead habit uses the FAST alpha only (page mirrors this)
        for j, nm in enumerate(names):
            if nm in lk: u = u + w[j] * lk[nm]
        T = temp_of(logT, sidx, tk)
        u = np.where(lk["mask"], u / T, -np.inf)
        m = u[lk["mask"]].max() if lk["mask"].any() else 0.0
        e = np.exp(u - m); p = e / e.sum()
        surv = surv * (1.0 - p)
    return 1.0 - surv


def attach_hazard(prot_rows, ban_fit, nh):
    bw, blT, bsx = ban_fit
    for d in prot_rows:
        d["_x"]["haz"] = (lookahead_probs(d, BAN_F, bw, blT, bsx, nh)[d["legal"]]
                          if d["look"] else np.zeros(len(d["legal"])))


def fit_hybrid(decisions, end, nh, log=False):
    btr = [d for d in decisions if d["kind"] == "ban"]
    ptr = [d for d in decisions if d["kind"] == "protect"]
    ban_fit = fit_kind("ban", BAN_F, btr, RHO_BAN, end)
    attach_hazard(ptr, ban_fit, nh)
    prot_fit = fit_kind("protect", PROT_F, ptr, None, end)
    if log:
        bw, blT, bsx = ban_fit; pw, plT, psx = prot_fit
        print("ban coefs:", {nm: round(float(bw[j]), 4) for j, nm in enumerate(BAN_F)},
              "T:", {sl: round(math.exp(blT[k]), 3) for sl, k in bsx.items()}, flush=True)
        print("protect coefs:", {nm: round(float(pw[j]), 4) for j, nm in enumerate(PROT_F)},
              "T:", {sl: round(math.exp(plT[k]), 3) for sl, k in psx.items()}, flush=True)
    return {"ban": ban_fit, "protect": prot_fit}


def predict(d, fits):
    names = BAN_F if d["kind"] == "ban" else PROT_F
    rho = RHO_BAN if d["kind"] == "ban" else None
    w, logT, sidx = fits[d["kind"]]
    u = utilities(d, names, w, logT, sidx, rho)
    m = u.max(); p = np.exp(u - m)
    return p / p.sum()


def frozen_fits(model):
    """Ban/protect coefficients and temperatures from an embedded modelData blob."""
    out = {}
    for kind, names in (("ban", BAN_F), ("protect", PROT_F)):
        slots = [s for s in SLOTS[kind] if s in model["coef"][kind]]
        first = model["coef"][kind][slots[0]]
        w = np.array([first[nm] for nm in names])
        logT = np.array([math.log(model["temps"][kind][s]) for s in slots])
        out[kind] = (w, logT, {s: k for k, s in enumerate(slots)})
    return out


# ---------- export / embed ----------
def data_summary(recs, excluded):
    teams = {}
    for r in recs:
        for s in ("blue", "red"):
            nm = r["teams"][s]["name"]
            e = teams.setdefault(nm, {"maps": 0, "last_seen": 0})
            e["maps"] += 1; e["last_seen"] = max(e["last_seen"], r["played_at"])
    return {"maps": len(recs), "series": len({r["match_id"] for r in recs}),
            "excluded_maps": excluded,
            "first_seen": min(r["played_at"] for r in recs),
            "last_seen": max(r["played_at"] for r in recs),
            "teams": dict(sorted(teams.items()))}


def export_params(st, fits, roles, fitted_on, summary, slot_map=None, temp_map=None):
    """As-of-END snapshot for the page. Every slot the page can ask for is
    exported; under a variant that keys B5 to B4, B5's habit/temperature are
    B4's, and a slot the fit never saw gets the last fitted temperature."""
    bw, blT, bsx = fits["ban"]; pw, plT, psx = fits["protect"]
    slot_map = slot_map or {}
    skey_of = lambda sl: slot_map.get(sl, sl)
    tkey_of = (lambda sl: temp_map.get(sl, sl)) if temp_map is not None else skey_of
    g = st.g_asof(); logg = np.log(g)
    teams_out = {}
    for team in sorted(st.team_n):
        if st.team_n[team] <= 0:
            continue
        cap = st.cap_asof(team, g); thr = st.threat_asof(team, g); ls = st.ls_asof(team)
        sb = st.team_banc[team] / (st.team_n[team] + 1.0)
        sp = st.team_protc[team] / (st.team_n[team] + 1.0)
        teams_out[team] = {"cap": [round(float(x), 5) for x in cap],
                           "thr": [round(float(x), 4) for x in thr],
                           "ls": [round(float(x), 5) for x in ls],
                           "sb": [round(float(x), 5) for x in sb],
                           "sp": [round(float(x), 5) for x in sp],
                           "v": [round(float(x), 4) for x in (logg + 0.76 * cap)]}
    alphas, alphas_slow = {}, {}
    for sl in SLOTS["ban"] + SLOTS["protect"]:
        alphas[sl] = [round(float(x), 4) for x in st.alpha_asof(skey_of(sl))]
        alphas_slow[sl] = [round(float(x), 4) for x in st.alpha_asof_slow(skey_of(sl))]
    coef = {"ban": {sl: dict({nm: round(float(bw[j]), 4) for j, nm in enumerate(BAN_F)}, den=0.0)
                    for sl in SLOTS["ban"]},
            "protect": {sl: {nm: round(float(pw[j]), 4) for j, nm in enumerate(PROT_F)}
                        for sl in SLOTS["protect"]}}
    return {"fitted_on": fitted_on,
            "tau": TAU, "patterns": [[list(c), r] for c, r in PATTERNS],
            "coef": coef,
            "temps": {"ban": {sl: round(temp_of(blT, bsx, tkey_of(sl)), 3) for sl in SLOTS["ban"]},
                      "protect": {sl: round(temp_of(plT, psx, sl), 3) for sl in SLOTS["protect"]}},
            "mix": {"rho": RHO_BAN, "w_ban": round(st.w_now("ban"), 4)},
            "heroes": st.heroes, "roles": {h: roles[h] for h in st.heroes},
            "alpha": alphas, "alpha_slow": alphas_slow,
            "g": [round(float(x), 6) for x in g],
            "map_offsets": {mp: [round(float(x), 3) for x in st.mapoff_asof(mp)] for mp in st.map_n},
            "teams": teams_out, "default_v": [round(float(x), 4) for x in logg],
            "data_summary": summary}


_MODEL_RE = re.compile(r'(<script id="modelData" type="application/json">)(.*?)(</script>)', re.S)


def embed_draft(page, params):
    """Replace the draft-model keys of the embedded modelData, keeping the
    lineup keys; the draft's own data summary lands in draft_data_summary."""
    m = _MODEL_RE.search(page)
    if not m:
        raise ValueError("No embedded modelData found")
    model = json.loads(m.group(2))
    for k, v in params.items():
        model["draft_data_summary" if k == "data_summary" else k] = v
    blob = json.dumps(model, separators=(",", ":"))
    return page[:m.start(2)] + blob + page[m.end(2):]


# ---------- holdout ----------
def score(rows, fits, dump=None):
    agg = {"ban": [0.0, 0, 0, 0], "protect": [0.0, 0, 0, 0]}
    per_slot = defaultdict(lambda: [0.0, 0])
    for d in rows:
        p = predict(d, fits)
        rank = int((p > p[d["y"]]).sum()) + 1
        nl = -math.log(max(p[d["y"]], 1e-12))
        a = agg[d["kind"]]
        a[0] += nl; a[1] += 1; a[2] += (rank == 1); a[3] += (rank <= 3)
        ps = per_slot[d["slot"]]; ps[0] += nl; ps[1] += 1
        if dump is not None:
            dump.append({"series": d["series"], "kind": d["kind"], "slot": d["slot"],
                         "nll": nl, "rank": rank})
    return agg, per_slot


def run_holdout(decisions, st, blocks, nh, frozen=None, scorable=None, label="", log=True,
                dump=None):
    tot = {"ban": [0.0, 0, 0, 0], "protect": [0.0, 0, 0, 0]}
    per_slot = defaultdict(lambda: [0.0, 0])
    t_run = time.time()
    for bi, blk in enumerate(blocks):
        blk = set(blk)
        cutoff = min(d["t"] for d in decisions if d["series"] in blk)
        test = [d for d in decisions if d["series"] in blk]
        if scorable is not None:
            test = [d for d in test if scorable(d)]
        for d in decisions: d["_x"] = {}
        if frozen is None:
            train = [d for d in decisions if d["t"] < cutoff]
            fits = fit_hybrid(train, cutoff, nh)
        else:
            fits = frozen
            attach_hazard([d for d in decisions if d["kind"] == "protect" and d["t"] < cutoff],
                          fits["ban"], nh)
        attach_hazard([d for d in test if d["kind"] == "protect"], fits["ban"], nh)
        agg, ps = score(test, fits, dump)
        for kind in tot:
            for i in range(4): tot[kind][i] += agg[kind][i]
        for sl, v in ps.items():
            per_slot[sl][0] += v[0]; per_slot[sl][1] += v[1]
        if log:
            eta = (time.time() - t_run) / (bi + 1) * (len(blocks) - bi - 1)
            print(f"   [{label}] block {bi+1}/{len(blocks)} ETA {eta:.0f}s", flush=True)
    return tot, per_slot


def format_result(label, tot, per_slot):
    line = label.ljust(14)
    for kind in ("ban", "protect"):
        nl, n, t1, t3 = tot[kind]
        if n:
            line += f" | {kind}: ll {nl/n:.4f} top1 {100*t1/n:.1f}% top3 {100*t3/n:.1f}% n={n}"
    line += "\n   per-slot ll: " + " ".join(
        f"{sl} {v[0]/v[1]:.3f}(n={v[1]})" for sl, v in sorted(per_slot.items()) if v[1])
    return line


def series_blocks(series_order, decisions, start=None, last=None, block=5, after_ts=None):
    first_t = {}
    for d in decisions:
        first_t[d["series"]] = min(first_t.get(d["series"], d["t"]), d["t"])
    chosen = list(series_order)
    if start is not None:
        t0 = _end_of_day(start) - DAY
        chosen = [s for s in chosen if first_t.get(s, 0) >= t0]
    if after_ts is not None:
        chosen = [s for s in chosen if first_t.get(s, 0) > after_ts]
    if last is not None:
        chosen = chosen[-last:]
    return [chosen[i:i + block] for i in range(0, len(chosen), block)]


def run_variant(variant, recs, heroes, roles, phi, model=None, dump=None, log=True, **block_args):
    """Holdout for one variant: `live` freezes the coefficients of an embedded
    modelData blob (scored on the slots it knows, with the hazard windows it
    knows); the others refit before every block."""
    if variant == "live":
        fits = frozen_fits(model)
        decisions, st = feature_pass(recs, heroes, roles, phi=phi, log=log,
                                     p2_sees_b5="B5" in model["temps"]["ban"])
        blocks = series_blocks(st.series_order, decisions, **block_args)
        known = set(fits["ban"][2]) | set(fits["protect"][2])
        tot, ps = run_holdout(decisions, st, blocks, st.NH, frozen=fits, log=log,
                              scorable=lambda d: d["tkey"] in known, label=variant, dump=dump)
    else:
        decisions, st = feature_pass(recs, heroes, roles, phi=phi, log=log, **VARIANTS[variant])
        blocks = series_blocks(st.series_order, decisions, **block_args)
        tot, ps = run_holdout(decisions, st, blocks, st.NH, label=variant, dump=dump, log=log)
    return tot, ps, blocks


# ---------- CLI ----------
# How the fifth ban enters the model: its own habit and temperature; B4's for
# both; or B4's habit (pooled late-ban evidence) with its own temperature.
VARIANTS = {"own": {},
            "share": {"slot_map": {"B5": "B4"}},
            "pooled": {"slot_map": {"B5": "B4"}, "temp_map": {}}}


def _common(ap):
    ap.add_argument("--data-root", default=DRAFT_ROOT)
    ap.add_argument("--as-of", default=None, help="keep maps played on/before this date (YYYY-MM-DD) and heroes released by then")
    ap.add_argument("--phi", type=float, default=0.25)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); _common(f)
    f.add_argument("--out", default="evaluation/draft_params.json")
    f.add_argument("--embed", default=None)
    f.add_argument("--variant", default="own", choices=sorted(VARIANTS))
    f.add_argument("--label", default="hybrid v5")
    h = sub.add_parser("holdout"); _common(h)
    h.add_argument("--start", default=None, help="score series starting on/after this date")
    h.add_argument("--last", type=int, default=None, help="score only the last N chosen series")
    h.add_argument("--block", type=int, default=5)
    h.add_argument("--variants", default="own")
    h.add_argument("--page", default="index.html", help="page whose embedded coefficients the `live` variant freezes")
    h.add_argument("--dump", default=None, help="write per-decision losses by variant to this JSON file")
    args = ap.parse_args(argv)

    roles = load_roles(args.data_root)
    heroes = heroes_as_of(roles, load_releases(args.data_root), args.as_of)
    recs, excluded = load_maps(args.data_root, args.as_of)
    print(f"{len(recs)} maps, {len(heroes)} heroes, as_of={args.as_of}, phi={args.phi}", flush=True)

    if args.cmd == "fit":
        decisions, st = feature_pass(recs, heroes, roles, phi=args.phi, log=True,
                                     **VARIANTS[args.variant])
        end = max(d["t"] for d in decisions)
        fits = fit_hybrid(decisions, end, st.NH, log=True)
        summary = data_summary(recs, excluded)
        fitted_on = (f"{len(recs)} maps through {time.strftime('%Y-%m-%d', time.gmtime(end))}; "
                     f"{args.label}: v4 hybrid + five-ban format (B5 {args.variant}), Season 10 break")
        params = export_params(st, fits, roles, fitted_on, summary, **VARIANTS[args.variant])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(params, open(args.out, "w"))
        print(f"wrote {args.out} ({len(json.dumps(params))} bytes)", flush=True)
        if args.embed:
            page = Path(args.embed).read_text()
            Path(args.embed).write_text(embed_draft(page, params))
            print(f"embedded into {args.embed}", flush=True)
        return

    results, dumps = {}, {}
    model = json.loads(_MODEL_RE.search(Path(args.page).read_text()).group(2))
    for variant in args.variants.split(","):
        dump = dumps.setdefault(variant, [])
        tot, ps, _ = run_variant(variant, recs, heroes, roles, args.phi, model=model, dump=dump,
                                 start=args.start, last=args.last, block=args.block)
        results[variant] = format_result(variant, tot, ps)
        print(results[variant], flush=True)
    print("\n=== holdout summary ===")
    for line in results.values():
        print(line)
    if args.dump:
        Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dumps, open(args.dump, "w"))
        print(f"wrote {args.dump}")


if __name__ == "__main__":
    main()
