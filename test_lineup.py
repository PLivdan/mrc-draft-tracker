import json, subprocess, unittest
from pathlib import Path

import numpy as np

from lineup_model import Config, assignment, base_q, collect, condition, export_state, load_data, project, score, split_series


class LineupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roles, cls.records, cls.rejected = load_data()

    def test_qa_and_series_split(self):
        self.assertEqual(self.rejected, 16)
        self.assertTrue(all(r['qa']['status'] == 'pass' for r in self.records))
        inner, replay = split_series(self.records)
        self.assertFalse(inner & replay)
        self.assertLess(max(r['played_at'] for r in self.records if r['match_id'] in inner),
                        min(r['played_at'] for r in self.records if r['match_id'] in replay))

    def test_prefix_is_unchanged_by_future_data(self):
        short, _ = collect(self.roles, self.records[:5], .95)
        longer, _ = collect(self.roles, self.records[:8], .95)
        for a, b in zip(short, longer):
            np.testing.assert_array_equal(a['offset'], b['offset'])
            for p, q in zip(a['players'], b['players']):
                for key in ('p', 'available', 'team', 'league'):
                    np.testing.assert_array_equal(p[key], q[key])
        self.assertTrue(all(not p['p'].any() for p in short[0]['players']))

    def test_blind_phase_masks(self):
        obs, state = collect(self.roles, self.records[:1], .95)
        r = self.records[0]
        for ob in obs:
            self.assertFalse(ob['stages']['opening'][0].any())
            self.assertFalse(ob['stages']['opening'][1].any())
            other = 'red' if ob['side'] == 'blue' else 'blue'
            for a in r['actions']:
                if a['kind'] == 'ban' and a['side'] == other and a['phase'] == 5:
                    self.assertFalse(ob['stages']['first_protects'][0][state.hidx[a['hero']]])
                    self.assertTrue(ob['stages']['second_blind_bans'][0][state.hidx[a['hero']]])

    def test_export_has_all_candidates_and_matches_snapshot(self):
        _, state = collect(self.roles, self.records[:10], .95)
        config = Config(role_pooling=1)
        payload = export_state(state, config, self.records[:10], 0)
        for team, players in payload['player_q'].items():
            for p, source in zip(players, state.latest[team]):
                self.assertEqual([x[0] for x in p['q']], state.heroes)
                self.assertAlmostEqual(sum(x[1] for x in p['q']), 1)
                expected = base_q(state.ingredients(team, source), state.role_idx, config)
                np.testing.assert_allclose([x[1] for x in p['q']], expected, atol=1e-14)
        for name, offsets in payload['lineup_map_offsets'].items():
            np.testing.assert_array_equal(list(offsets.values()), state.map_offset(name))

    def test_assignment_and_box_coverage(self):
        config = Config()
        primary, alt = project(np.array([[.55,.45,0],[.99,.005,.005]]), config)
        self.assertEqual(primary, [1,0])
        self.assertEqual(alt, [0,None])
        q = condition(np.array([.6,.4]), np.array([True,True]), np.zeros(2), np.zeros(2), config)
        np.testing.assert_array_equal(q, [0,0])
        self.assertEqual(project([q], config), ([None],[None]))
        qs = np.array([[.6,.3,.1],[.9,.06,.04]])
        primary, alt = project(qs, config)
        self.assertEqual(primary, [1,0])
        self.assertIsNone(alt[1])

    def test_assignment_against_scipy(self):
        from scipy.optimize import linear_sum_assignment
        rng = np.random.default_rng(42)
        for n in (1, 3, 6):
            for _ in range(15):
                cost = rng.normal(size=(n, 54))
                rows, cols = linear_sum_assignment(cost)
                ours = assignment(cost.tolist())
                self.assertAlmostEqual(sum(cost[i, j] for i, j in enumerate(ours)), float(cost[rows, cols].sum()), places=10)

    def test_javascript_parity(self):
        obs, state = collect(self.roles, self.records[:20], .95)
        config = Config(role_pooling=1)
        examples = []
        for stage in ('opening', 'first_protects', 'second_blind_bans', 'complete'):
            _, _, fixtures = score(obs, state, config, stage, details=True)
            examples.extend(fixtures)
        result = subprocess.run(['node', str(Path(__file__).with_name('check_lineup.js'))],
                                input=json.dumps(examples), text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
