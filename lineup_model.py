import json, math, tarfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    dp: float = 0.95
    kp: float = 5.0
    kt: float = 20.0
    exposure: bool = True
    role_pooling: float = 0.0
    protect: float = 1.0
    map: float = 0.5
    alternative_min: float = 0.10


def load_data(data_root=None, bundle=None):
    if data_root:
        root = Path(data_root)
        roles = json.loads((root / 'data/roles/heroes.json').read_text())
        records = [json.loads(p.read_text()) for p in sorted((root / 'data/processed/maps').glob('*.json'))]
    else:
        with tarfile.open(bundle or Path(__file__).with_name('league_bundle.tar.gz')) as tar:
            roles = json.load(tar.extractfile('data/roles/heroes.json'))
            records = [json.load(tar.extractfile(m)) for m in tar.getmembers()
                       if m.isfile() and m.name.startswith('data/processed/maps/') and m.name.endswith('.json')]
    kept = [r for r in records if r.get('qa', {}).get('status') == 'pass']
    kept.sort(key=lambda r: (r['played_at'], r['map_uid']))
    if not kept:
        raise ValueError('No QA-passed maps found')
    if len({r['map_uid'] for r in kept}) != len(kept):
        raise ValueError('Duplicate map_uid in input')
    return roles, kept, len(records) - len(kept)


def split_series(records, hold_count=45, inner_maps=120):
    series = list(dict.fromkeys(r['match_id'] for r in records))
    if len(series) <= hold_count + 1:
        raise ValueError('Not enough series for the requested split')
    replay = set(series[-hold_count:])
    cutoff = min(r['played_at'] for r in records if r['match_id'] in replay)
    ends = {sid: max(r['played_at'] for r in records if r['match_id'] == sid) for sid in series}
    pre = [r for r in records if r['match_id'] not in replay and ends[r['match_id']] < cutoff]
    inner = set(r['match_id'] for r in pre[-inner_maps:])
    return inner, replay


class State:
    def __init__(self, roles, dp):
        self.heroes = sorted(roles)
        self.hidx = {h: i for i, h in enumerate(self.heroes)}
        self.role_names = sorted(set(roles.values()))
        self.role_idx = np.array([self.role_names.index(roles[h]) for h in self.heroes])
        self.nh, self.dp = len(self.heroes), dp
        self.p = defaultdict(self.zero)
        self.available = defaultdict(self.zero)
        self.team = defaultdict(self.zero)
        self.league = self.zero()
        self._global_prior_cache = None
        self.map_use = defaultdict(self.zero)
        self.map_n = defaultdict(float)
        self.player_maps = defaultdict(int)
        self.player_last = {}
        self.team_maps = defaultdict(int)
        self.team_last = {}
        self.latest = {}

    def zero(self):
        return np.zeros(self.nh)

    def share(self, record, pid):
        raw = record['hero_time'].get(pid, {})
        vec = self.zero()
        for h, value in raw.items():
            if h not in self.hidx:
                raise ValueError('Unknown observed hero: ' + h)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid hero time')
            vec[self.hidx[h]] += value
        total = vec.sum()
        return vec / total if total > 0 else vec

    def global_prior(self):
        if self._global_prior_cache is None:
            self._global_prior_cache = (self.league + 0.1) / (self.league.sum() + 0.1 * self.nh)
        return self._global_prior_cache

    def map_offset(self, name):
        if not name or self.map_n.get(name, 0) < 3 or self.league.sum() <= 0:
            return self.zero()
        reference = self.global_prior()
        local = (self.map_use[name] + 90.0 * reference) / (6.0 * self.map_n[name] + 90.0)
        return np.clip(np.log(np.maximum(local / reference, 1e-6)), -1.2, 1.2)

    def ingredients(self, team, player):
        pid = player['player_id']
        return {'pid': pid, 'name': player['name'], 'p': self.p[pid].copy(),
                'available': self.available[pid].copy(), 'team': self.team[team].copy(),
                'league': self.global_prior(), 'maps': self.player_maps[pid],
                'last_seen': self.player_last.get(pid)}

    def observations(self, record):
        stages = {}
        for label, limit in [('opening', -1), ('first_protects', 3), ('second_blind_bans', 5), ('complete', 9)]:
            masks = {}
            for side in ('blue', 'red'):
                other = 'red' if side == 'blue' else 'blue'
                banned = np.zeros(self.nh, dtype=bool)
                protected = self.zero()
                for a in record['actions']:
                    if a['phase'] > limit:
                        continue
                    if a['kind'] == 'ban' and a['side'] == other:
                        banned[self.hidx[a['hero']]] = True
                    if a['kind'] == 'protect' and a['side'] == side:
                        protected[self.hidx[a['hero']]] = 1.0
                masks[side] = (banned, protected)
            stages[label] = masks
        out = []
        for side in ('blue', 'red'):
            team = record['teams'][side]['name']
            previous_ids = {p['player_id'] for p in self.latest.get(team, [])}
            players = []
            for player in record['lineups'][side]:
                row = self.ingredients(team, player)
                row['actual'] = self.share(record, row['pid'])
                row['roster_known'] = row['pid'] in previous_ids
                players.append(row)
            out.append({'map_uid': record['map_uid'], 'series': record['match_id'], 't': record['played_at'],
                        'team': team, 'side': side, 'players': players,
                        'offset': self.map_offset(record.get('map_name')),
                        'stages': {k: v[side] for k, v in stages.items()}})
        return out

    def update(self, record):
        name = record.get('map_name')
        self.league *= 0.995
        if name:
            self.map_use[name] *= 0.995
            self.map_n[name] *= 0.995
        for side in ('blue', 'red'):
            team = record['teams'][side]['name']
            other = 'red' if side == 'blue' else 'blue'
            available = np.ones(self.nh)
            for a in record['actions']:
                if a['kind'] == 'ban' and a['side'] == other:
                    available[self.hidx[a['hero']]] = 0.0
            self.team[team] *= 0.88
            self.latest[team] = record['lineups'][side]
            self.team_maps[team] += 1
            self.team_last[team] = record['played_at']
            for player in record['lineups'][side]:
                pid = player['player_id']
                vec = self.share(record, pid)
                if vec.sum() <= 0:
                    continue
                if np.any((available == 0) & (vec > 0)):
                    raise ValueError('QA-passed map contains playtime on a banned hero: ' + record['map_uid'])
                self.p[pid] = self.dp * self.p[pid] + vec
                self.available[pid] = self.dp * self.available[pid] + available
                self.team[team] += vec
                self.league += vec
                self.player_maps[pid] += 1
                self.player_last[pid] = record['played_at']
                if name:
                    self.map_use[name] += vec
            if name:
                self.map_n[name] += 1
        self._global_prior_cache = None


def collect(roles, records, dp):
    state = State(roles, dp)
    observations = []
    for record in records:
        observations.extend(state.observations(record))
        state.update(record)
    return observations, state


def base_q(player, role_idx, config):
    prior = (player['team'] + config.kt * player['league']) / (player['team'].sum() + config.kt)
    if config.role_pooling:
        nr = int(role_idx.max()) + 1
        role_mass = np.bincount(role_idx, weights=prior, minlength=nr)
        player_role = np.bincount(role_idx, weights=player['p'], minlength=nr)
        player_role = (player_role + role_mass) / (player['p'].sum() + 1.0)
        specialized = prior * player_role[role_idx] / role_mass[role_idx]
        prior = (1.0 - config.role_pooling) * prior + config.role_pooling * specialized
    denominator = player['available'] if config.exposure else player['p'].sum()
    q = (player['p'] + config.kp * prior) / (denominator + config.kp)
    return q / q.sum()


def condition(q, banned, protected, offset, config):
    q = np.where(banned, 0.0, q * np.exp(config.protect * protected + config.map * offset))
    return q / q.sum() if q.sum() > 0 else np.zeros_like(q)


def assignment(cost):
    # Deliberately not scipy.optimize.linear_sum_assignment: lineup-engine.js
    # hand-ports this exact Jonker-Volgenant variant so Python and the browser
    # agree on tie-breaking (see test_javascript_parity). A different optimal
    # solver can pick a different, equally-optimal assignment under ties.
    n, m = len(cost), len(cost[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = [float('inf')] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float('inf'), 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            prev = way[j0]
            p[j0], j0 = p[prev], prev
            if j0 == 0:
                break
    out = [None] * n
    for j in range(1, m + 1):
        if p[j]:
            out[p[j] - 1] = j - 1
    return out


def project(qs, config):
    if not len(qs):
        return [], []
    qs = np.asarray(qs)
    n, nh = qs.shape
    cost = [[round(-math.log(q), 12) if q > 0 else 1e6 for q in row] + [5e5] * n for row in qs]
    cols = assignment(cost)
    primary = [None] * n
    for r, c in enumerate(cols):
        if c < nh and qs[r, c] > 0:
            primary[int(r)] = int(c)
    alternate = []
    for i, q in enumerate(qs):
        ranked = sorted(range(nh), key=lambda j: (-q[j], j))
        alt = next((j for j in ranked if j != primary[i] and q[j] >= config.alternative_min), None)
        alternate.append(alt)
    return primary, alternate


def score(observations, state, config, stage='complete', series=None, details=False):
    sums = defaultdict(float)
    by_series = defaultdict(lambda: defaultdict(float))
    fixtures = []
    for ob in observations:
        if series is not None and ob['series'] not in series:
            continue
        banned, protected = ob['stages'][stage]
        qs = [condition(base_q(p, state.role_idx, config), banned, protected, ob['offset'], config) for p in ob['players']]
        primary, alternate = project(qs, config)
        block = by_series[ob['series']]
        for i, player in enumerate(ob['players']):
            a, q = player['actual'], qs[i]
            if a.sum() <= 0:
                continue
            best = int(np.argmax(a))
            shown = [h for h in (primary[i], alternate[i]) if h is not None]
            vals = {'n': 1, 'ce': float(-(a * np.log(np.maximum(q, 1e-12))).sum()),
                    'primary': float(primary[i] == best), 'either': float(best in shown),
                    'coverage': float(a[shown].sum()), 'roster_known': float(player['roster_known'])}
            for k, v in vals.items():
                sums[k] += v
                block[k] += v
        if details and len(fixtures) < 24:
            fixtures.append({'players': [{'name': p['name'], 'q': [[h, float(x)] for h, x in zip(state.heroes, base_q(p, state.role_idx, config))]}
                                         for p in ob['players']],
                             'banned': [h for h, b in zip(state.heroes, banned) if b],
                             'protected': [h for h, b in zip(state.heroes, protected) if b],
                             'offsets': dict(zip(state.heroes, ob['offset'].tolist())), 'config': asdict(config),
                             'expected': {'q': [q.tolist() for q in qs],
                                          'primary': [state.heroes[h] if h is not None else None for h in primary],
                                          'alternate': [state.heroes[h] if h is not None else None for h in alternate]}})
    n = int(sums['n'])
    result = {'n': n, **{k: v / max(n, 1) for k, v in sums.items() if k != 'n'}, 'series': len(by_series)}
    return (result, dict(by_series), fixtures) if details else result


def export_state(state, config, records, rejected):
    player_q, player_pools, metadata = {}, {}, {}
    for team, roster in sorted(state.latest.items()):
        player_q[team], player_pools[team] = [], []
        for player in roster:
            row = state.ingredients(team, player)
            q = base_q(row, state.role_idx, config)
            player_q[team].append({'name': row['name'], 'player_id': row['pid'],
                                   'maps': row['maps'], 'last_seen': row['last_seen'],
                                   'q': [[h, float(v)] for h, v in zip(state.heroes, q)]})
            history = row['p'] / row['p'].sum() if row['p'].sum() else state.zero()
            order = sorted(range(state.nh), key=lambda i: (-history[i], i))
            player_pools[team].append({'name': row['name'], 'maps': row['maps'],
                                      'heroes': [[state.heroes[i], float(history[i])] for i in order[:8] if history[i] >= 0.02],
                                      'hero_history': {state.heroes[i]: float(history[i]) for i in order if history[i] > 0}})
        metadata[team] = {'maps': state.team_maps[team], 'last_seen': state.team_last[team]}
    return {'schema_version': 1, 'config': asdict(config), 'player_q': player_q, 'player_pools': player_pools,
            'lineup_map_offsets': {name: dict(zip(state.heroes, state.map_offset(name).tolist())) for name in sorted(state.map_n)},
            'data_summary': {'maps': len(records), 'series': len({r['match_id'] for r in records}),
                             'excluded_maps': rejected, 'first_seen': records[0]['played_at'],
                             'last_seen': records[-1]['played_at'], 'teams': metadata}}
