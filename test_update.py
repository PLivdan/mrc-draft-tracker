import json, unittest
from pathlib import Path
import tempfile

from update import (append_log, coef_delta, new_series_after, qa_guards,
                    unknown_heroes_from)


class HeroScanTests(unittest.TestCase):
    def test_parses_unknown_hero_lines(self):
        out = "gorr\tDuelist\tfirst seen in match js71dg\nnightcrawler\tDuelist\tfirst seen in match js9\n"
        self.assertEqual(unknown_heroes_from(out), [("gorr", "Duelist"), ("nightcrawler", "Duelist")])

    def test_clean_scan_is_empty(self):
        self.assertEqual(unknown_heroes_from("no heroes outside data/roles/heroes.json\n"), [])

    def test_placeholder_hero_is_tolerated_in_scan(self):
        out = "hero-0\tDuelist\tfirst seen in match js700124\ngorr\tDuelist\tfirst seen in match js71dg\n"
        self.assertEqual(unknown_heroes_from(out), [("gorr", "Duelist")])


QA_OK = """# QA report

947/964 maps pass.

- REJECT js71tap:g1: ["heroes not in legal roster: ['hero-0']"]
- MISSING raw page for match js711ty
- REJECT js7012g:g1: ["heroes not in legal roster: ['hero-0']"]
"""


class QaGuardTests(unittest.TestCase):
    def test_placeholder_hero_rejects_are_tolerated(self):
        g = qa_guards(QA_OK)
        self.assertEqual(g["status"], "ok")
        self.assertEqual(g["passed"], 947)
        self.assertEqual(g["total"], 964)
        self.assertEqual(g["missing"], 1)
        self.assertEqual(g["warnings"], [])

    def test_structure_mismatch_halts_as_a_format_change(self):
        text = QA_OK + '- REJECT js7abc:g2: ["expected 12 or 14 draft choices, got 16", "draft structure mismatch: missing [], extra [(\'blue\', \'ban\', 11)]"]\n'
        g = qa_guards(text)
        self.assertEqual(g["status"], "halt")
        self.assertTrue(any("format" in r for r in g["reasons"]))

    def test_unknown_roster_hero_halts(self):
        text = QA_OK + '- REJECT js7abc:g2: ["heroes not in legal roster: [\'nightcrawler\']"]\n'
        g = qa_guards(text)
        self.assertEqual(g["status"], "halt")
        self.assertTrue(any("nightcrawler" in r for r in g["reasons"]))

    def test_other_rejects_only_warn(self):
        text = QA_OK + '- REJECT js7abc:g2: ["blue lineup has 5 players"]\n'
        g = qa_guards(text)
        self.assertEqual(g["status"], "ok")
        self.assertEqual(len(g["warnings"]), 1)


class SeriesTests(unittest.TestCase):
    def test_new_series_are_those_starting_after_last_seen(self):
        decisions = [{"series": "a", "t": 100}, {"series": "a", "t": 150},
                     {"series": "b", "t": 160}, {"series": "b", "t": 300},
                     {"series": "c", "t": 200}, {"series": "c", "t": 250}]
        order = ["a", "b", "c"]
        self.assertEqual(new_series_after(order, decisions, last_seen=150), ["b", "c"])
        self.assertEqual(new_series_after(order, decisions, last_seen=160), ["c"])
        self.assertEqual(new_series_after(order, decisions, last_seen=300), [])


class LogTests(unittest.TestCase):
    def test_append_creates_and_preserves(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.json"
            append_log(p, {"date": "2026-09-21", "n": 1})
            append_log(p, {"date": "2026-09-22", "n": 2})
            self.assertEqual([e["date"] for e in json.loads(p.read_text())],
                             ["2026-09-21", "2026-09-22"])


class DeltaTests(unittest.TestCase):
    def test_coef_delta_lists_changed_weights_and_temps(self):
        old = {"coef": {"ban": {"B1": {"cap": 1.0, "thr": 2.0}}, "protect": {"P1": {"haz": 1.0}}},
               "temps": {"ban": {"B1": 1.0, "B5": 1.4}, "protect": {"P1": 1.2}}}
        new = {"coef": {"ban": {"B1": {"cap": 1.5, "thr": 2.0}}, "protect": {"P1": {"haz": 1.1}}},
               "temps": {"ban": {"B1": 1.0, "B5": 1.5}, "protect": {"P1": 1.2}}}
        rows = coef_delta(old, new)
        self.assertIn(("ban", "cap", 1.0, 1.5), rows)
        self.assertIn(("protect", "haz", 1.0, 1.1), rows)
        self.assertIn(("temp ban", "B5", 1.4, 1.5), rows)
        self.assertNotIn(("ban", "thr", 2.0, 2.0), rows)


if __name__ == "__main__":
    unittest.main()
