import json, unittest

import numpy as np

from fit_draft import (DAY, H_OBS, VARIANTS, data_summary, embed_draft, export_params,
                       feature_pass, fit_hybrid, fit_kind, hazard_window, heroes_as_of,
                       kind_objective, lookahead_probs, n_bans_of, temp_of,
                       urgency_window, utilities)

ROLES = {"v1": "Vanguard", "v2": "Vanguard", "v3": "Vanguard",
         "d1": "Duelist", "d2": "Duelist", "d3": "Duelist",
         "s1": "Strategist", "s2": "Strategist", "s3": "Strategist"}
HEROES = sorted(ROLES)

FOUR_BAN = [
    (0, "blue", "ban", "B1", "v1"), (0, "red", "ban", "B1", "v2"),
    (1, "red", "protect", "P1", "s1"),
    (2, "blue", "ban", "B2", "d1"),
    (3, "blue", "protect", "P1", "s2"),
    (4, "red", "ban", "B2", "d2"),
    (5, "blue", "ban", "B3", "v3"), (5, "red", "ban", "B3", "d3"),
    (6, "blue", "protect", "P2", "s3"),
    (7, "red", "ban", "B4", "s1"),
    (8, "red", "protect", "P2", "d3"),
    (9, "blue", "ban", "B4", "d2"),
]
FIVE_BAN = FOUR_BAN + [(10, "blue", "ban", "B5", "v2"), (10, "red", "ban", "B5", "v3")]


def record(match_id, game, t, actions, blue="Team A", red="Team B"):
    acts = [{"phase": p, "side": s, "kind": k, "slot": sl, "hero": h}
            for p, s, k, sl, h in actions]
    lineups, hero_time = {}, {}
    for side, team in (("blue", blue), ("red", red)):
        pids = [f"{team}-p{i}" for i in range(6)]
        lineups[side] = [{"player_id": p} for p in pids]
        for i, p in enumerate(pids):
            hero_time[p] = {HEROES[i]: 600.0}
    return {"map_uid": f"{match_id}:g{game}", "match_id": match_id,
            "played_at": t, "map_name": "Midtown",
            "teams": {"blue": {"name": blue}, "red": {"name": red}},
            "actions": acts, "lineups": lineups, "hero_time": hero_time,
            "winner_side": "blue", "qa": {"status": "pass"}}


class FormatTests(unittest.TestCase):
    def test_n_bans_of_reads_the_largest_ban_slot(self):
        self.assertEqual(n_bans_of(record("m1", 1, 0, FOUR_BAN)), 4)
        self.assertEqual(n_bans_of(record("m2", 1, 0, FIVE_BAN)), 5)

    def test_p2_hazard_sees_b5_only_in_five_ban_maps(self):
        self.assertEqual(hazard_window("blue", "P2", 4), ("red", ["B4"]))
        self.assertEqual(hazard_window("red", "P2", 4), ("blue", ["B4"]))
        self.assertEqual(hazard_window("blue", "P2", 5), ("red", ["B4", "B5"]))
        self.assertEqual(hazard_window("red", "P2", 5), ("blue", ["B4", "B5"]))
        for n in (4, 5):
            self.assertEqual(hazard_window("red", "P1", n), ("blue", ["B2", "B3"]))
            self.assertEqual(hazard_window("blue", "P1", n), ("red", ["B2", "B3"]))

    def test_urgency_windows_unchanged_and_none_for_b5(self):
        self.assertEqual(urgency_window("blue", "B1"), ("red", ["P1"]))
        self.assertEqual(urgency_window("red", "B3"), ("blue", ["P2"]))
        self.assertIsNone(urgency_window("blue", "B5"))
        self.assertIsNone(urgency_window("red", "B4"))

    def test_heroes_as_of_excludes_unreleased(self):
        roles = dict(ROLES, gorr="Duelist", **{"the-hood": "Vanguard"})
        releases = {"gorr": "2026-09-11", "the-hood": "2026-08-07"}
        self.assertNotIn("gorr", heroes_as_of(roles, releases, "2026-08-01"))
        self.assertNotIn("the-hood", heroes_as_of(roles, releases, "2026-08-01"))
        self.assertIn("the-hood", heroes_as_of(roles, releases, "2026-08-10"))
        self.assertNotIn("gorr", heroes_as_of(roles, releases, "2026-08-10"))
        self.assertEqual(heroes_as_of(roles, releases, None), sorted(roles))


class FeaturePassTests(unittest.TestCase):
    def setUp(self):
        self.recs = [record("m1", 1, 1_000_000, FOUR_BAN),
                     record("m2", 1, 1_003_600, FIVE_BAN)]

    def test_emits_one_decision_per_action_including_b5(self):
        decisions, _ = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0)
        self.assertEqual(len(decisions), 12 + 14)
        b5 = [d for d in decisions if d["slot"] == "B5"]
        self.assertEqual(len(b5), 2)
        self.assertTrue(all(d["skey"] == "B5" and d["series"] == "m2" for d in b5))

    def test_p2_lookahead_covers_b5_only_on_the_five_ban_map(self):
        decisions, _ = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0)
        p2 = {d["series"]: d for d in decisions if d["slot"] == "P2" and d["side"] == "blue"}
        self.assertEqual(p2["m1"]["look"]["slots"], ["B4"])
        self.assertEqual(p2["m2"]["look"]["slots"], ["B4", "B5"])
        self.assertEqual(set(p2["m2"]["look"]["alpha"]), {"B4", "B5"})

    def test_share_b4_variant_keys_b5_to_b4(self):
        decisions, _ = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0,
                                    slot_map={"B5": "B4"})
        b5 = [d for d in decisions if d["slot"] == "B5"]
        self.assertTrue(all(d["skey"] == "B4" and d["tkey"] == "B4" for d in b5))
        p2 = next(d for d in decisions if d["slot"] == "P2" and d["series"] == "m2"
                  and d["side"] == "blue")
        self.assertEqual(p2["look"]["slots"], ["B4", "B5"])
        self.assertEqual(p2["look"]["skeys"], ["B4", "B4"])
        self.assertEqual(p2["look"]["tkeys"], ["B4", "B4"])

    def test_pooled_variant_shares_habit_but_keeps_own_temperature(self):
        decisions, _ = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0,
                                    slot_map={"B5": "B4"}, temp_map={})
        b5 = [d for d in decisions if d["slot"] == "B5"]
        self.assertTrue(all(d["skey"] == "B4" and d["tkey"] == "B5" for d in b5))
        p2 = next(d for d in decisions if d["slot"] == "P2" and d["series"] == "m2"
                  and d["side"] == "blue")
        self.assertEqual(p2["look"]["skeys"], ["B4", "B4"])
        self.assertEqual(p2["look"]["tkeys"], ["B4", "B5"])
        own, _ = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0)
        self.assertTrue(all(d["tkey"] == d["skey"] == d["slot"] for d in own))

    def test_b5_habit_accumulates_under_its_own_key(self):
        _, state = feature_pass(self.recs, HEROES, ROLES, breaks=[], phi=1.0)
        alpha_b5 = state.alpha_asof("B5")
        alpha_b4 = state.alpha_asof("B4")
        i_v2 = HEROES.index("v2")
        # v2 was banned in B5 on the five-ban map; the B5 habit must reflect that
        self.assertGreater(alpha_b5[i_v2], alpha_b5.mean())
        self.assertFalse(np.allclose(alpha_b5, alpha_b4))


class FitTests(unittest.TestCase):
    def _rows(self, slots, ragged=False):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(60):
            slot = slots[i % len(slots)]
            n_legal = 6 - (i % 3 if ragged else 0)
            legal = np.arange(n_legal)
            feats = {nm: rng.normal(size=n_legal) for nm in ("cap", "thr")}
            rows.append({"kind": "ban", "slot": slot, "skey": slot, "tkey": slot,
                         "t": 1000 + i * 40000,
                         "alpha": rng.normal(size=n_legal), "alpha_slow": rng.normal(size=n_legal),
                         "w": {0.9: 0.5}, "feats": feats, "_x": {}, "y": int(i % n_legal),
                         "legal": legal})
        return rows

    def test_unfitted_slot_borrows_the_last_fitted_temperature(self):
        sidx = {"B1": 0, "B2": 1, "B3": 2, "B4": 3}
        logT = np.array([0.1, 0.2, 0.3, 0.4])
        self.assertAlmostEqual(temp_of(logT, sidx, "B5"), np.exp(0.4))
        self.assertAlmostEqual(temp_of(logT, sidx, "B2"), np.exp(0.2))
        rows = self._rows(["B1", "B2", "B3", "B4"])
        w, logT, sidx = fit_kind("ban", ["cap", "thr"], rows, rho=0.9, end=rows[-1]["t"])
        b5 = dict(rows[0], slot="B5", skey="B5", tkey="B5")
        b4 = dict(rows[0], slot="B4", skey="B4", tkey="B4")
        np.testing.assert_allclose(utilities(b5, ["cap", "thr"], w, logT, sidx, 0.9),
                                   utilities(b4, ["cap", "thr"], w, logT, sidx, 0.9))
        look = {"slots": ["B4", "B5"], "skeys": ["B4", "B5"], "tkeys": ["B4", "B5"],
                "mask": np.ones(6, bool),
                "alpha": {"B4": np.zeros(6), "B5": np.zeros(6)},
                "cap": np.zeros(6), "thr": np.zeros(6)}
        haz = lookahead_probs({"look": look}, ["cap", "thr"], w, logT, sidx, 6)
        self.assertTrue(np.all((haz > 0) & (haz < 1)))

    def test_vectorised_objective_matches_row_by_row_loop(self):
        rows = self._rows(["B1", "B2", "B3", "B4", "B5"], ragged=True)
        names, rho, end = ["cap", "thr"], 0.9, 1000 + 59 * 40000
        nll, sidx = kind_objective("ban", names, rows, rho, end)
        rng = np.random.default_rng(1)
        for _ in range(3):
            th = rng.normal(size=len(names) + len(sidx))
            w, logT = th[:len(names)], th[len(names):]
            ref = 0.5 * float((w ** 2).sum())
            for d in rows:
                wt = 0.5 ** ((end - d["t"]) / DAY / H_OBS)
                u = utilities(d, names, w, logT, sidx, rho)
                m = u.max()
                ref += wt * (m + np.log(np.exp(u - m).sum()) - u[d["y"]])
            self.assertAlmostEqual(nll(th), ref, places=9)

    def test_temperatures_cover_present_slots_only(self):
        w, logT, sidx = fit_kind("ban", ["cap", "thr"], self._rows(["B1", "B2", "B3", "B4"]),
                                 rho=0.9, end=1100, slots=["B1", "B2", "B3", "B4", "B5"])
        self.assertEqual(set(sidx), {"B1", "B2", "B3", "B4"})
        self.assertEqual(len(logT), 4)
        w, logT, sidx = fit_kind("ban", ["cap", "thr"], self._rows(["B1", "B5"]),
                                 rho=0.9, end=1100, slots=["B1", "B2", "B3", "B4", "B5"])
        self.assertEqual(set(sidx), {"B1", "B5"})

    def test_temperature_slots_follow_tkey_not_skey(self):
        rows = self._rows(["B4", "B5"])
        for d in rows:
            d["skey"] = "B4"  # pooled habit, own temperature
        w, logT, sidx = fit_kind("ban", ["cap", "thr"], rows, rho=0.9, end=1100)
        self.assertEqual(set(sidx), {"B4", "B5"})


class ExportTests(unittest.TestCase):
    def _export(self, variant):
        recs = [record("m1", 1, 1_000_000, FOUR_BAN), record("m2", 1, 1_003_600, FIVE_BAN),
                record("m3", 1, 1_007_200, FIVE_BAN, blue="Team B", red="Team A")]
        decisions, st = feature_pass(recs, HEROES, ROLES, breaks=[], phi=1.0, **VARIANTS[variant])
        fits = fit_hybrid(decisions, max(d["t"] for d in decisions), st.NH)
        return export_params(st, fits, ROLES, "test", data_summary(recs, 0), **VARIANTS[variant])

    def test_every_ban_slot_is_exported_for_the_page(self):
        for variant in VARIANTS:
            p = self._export(variant)
            for key in ("alpha", "alpha_slow"):
                self.assertEqual(set(p[key]), {"B1", "B2", "B3", "B4", "B5", "P1", "P2"}, variant)
            self.assertEqual(set(p["coef"]["ban"]), {"B1", "B2", "B3", "B4", "B5"}, variant)
            self.assertEqual(set(p["temps"]["ban"]), {"B1", "B2", "B3", "B4", "B5"}, variant)

    def test_share_aliases_b5_to_b4_and_own_does_not(self):
        share, own, pooled = self._export("share"), self._export("own"), self._export("pooled")
        self.assertEqual(share["alpha"]["B5"], share["alpha"]["B4"])
        self.assertEqual(share["temps"]["ban"]["B5"], share["temps"]["ban"]["B4"])
        self.assertNotEqual(own["alpha"]["B5"], own["alpha"]["B4"])
        self.assertEqual(pooled["alpha"]["B5"], pooled["alpha"]["B4"])
        self.assertIn("B5", pooled["temps"]["ban"])


class EmbedTests(unittest.TestCase):
    def test_replaces_draft_keys_and_keeps_lineup_keys(self):
        old = {"coef": {"ban": {"B1": {"cap": 1.0}}}, "alpha": {"B1": [0.0]},
               "heroes": ["a"], "roles": {"a": "Duelist"},
               "player_q": {"T": []}, "lineup_config": {"dp": 0.9},
               "data_summary": {"maps": 868}, "draft_data_summary": {"maps": 868}}
        page = ('<html><script id="modelData" type="application/json">'
                + json.dumps(old) + '</script><script>var x=1;</script></html>')
        new = {"coef": {"ban": {"B1": {"cap": 2.0}, "B5": {"cap": 3.0}}},
               "alpha": {"B1": [1.0], "B5": [2.0]}, "heroes": ["a", "b"],
               "roles": {"a": "Duelist", "b": "Vanguard"},
               "data_summary": {"maps": 947}}
        out = embed_draft(page, new)
        m = json.loads(out.split('type="application/json">')[1].split("</script>")[0])
        self.assertEqual(m["coef"]["ban"]["B5"]["cap"], 3.0)
        self.assertEqual(m["heroes"], ["a", "b"])
        self.assertEqual(m["player_q"], {"T": []})
        self.assertEqual(m["lineup_config"], {"dp": 0.9})
        self.assertEqual(m["data_summary"], {"maps": 868})
        self.assertEqual(m["draft_data_summary"], {"maps": 947})
        self.assertIn("var x=1;", out)


if __name__ == "__main__":
    unittest.main()
