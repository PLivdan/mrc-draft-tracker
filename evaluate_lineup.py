import argparse, hashlib, json, os, re, subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lineup_model import Config, collect, export_state, load_data, score, split_series


def paired_intervals(before, after):
    keys = sorted(before.keys() & after.keys())
    rng = np.random.default_rng(7301)
    picks = rng.integers(0, len(keys), (2000, len(keys)))
    counts = np.array([after[k]['n'] for k in keys])
    result = {}
    for metric in ('ce', 'primary', 'either', 'coverage'):
        delta = np.array([after[k][metric] - before[k][metric] for k in keys])
        draws = delta[picks].sum(axis=1) / counts[picks].sum(axis=1)
        result[metric] = {'change': float(delta.sum() / counts.sum()),
                          'series_bootstrap_95': np.quantile(draws, [0.025, 0.975]).tolist()}
    return result


def embed(page_path, payload):
    page = Path(page_path).read_text()
    match = re.search(r'(<script id="modelData" type="application/json">)(.*?)(</script>)', page, re.S)
    if not match:
        raise ValueError('No embedded modelData found')
    model = json.loads(match[2])
    heroes = {h for players in payload['player_q'].values() for player in players for h, _ in player['q']}
    missing = heroes - set(model['roles'])
    if missing:
        raise ValueError('Add observed heroes to the draft roster with add_hero.py before embedding: ' + ', '.join(sorted(missing)))
    model.setdefault('draft_data_summary', model.get('data_summary', payload['data_summary']))
    for key in ('player_q', 'player_pools', 'lineup_map_offsets', 'data_summary'):
        model[key] = payload[key]
    model['lineup_config'] = payload['config']
    model['lineup_validation'] = payload['validation']
    model['lineup_schema_version'] = payload['schema_version']
    serialized = json.dumps(model, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    Path(page_path).write_text(page[:match.start(2)] + serialized + page[match.end(2):])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default=os.environ.get('DRAFT_ROOT'))
    parser.add_argument('--bundle')
    parser.add_argument('--output', default='evaluation/lineup_report.json')
    parser.add_argument('--export', default='evaluation/lineup_params.json')
    parser.add_argument('--embed')
    args = parser.parse_args()
    roles, records, rejected = load_data(args.data_root, args.bundle)
    inner, replay = split_series(records)
    cache = {dp: collect(roles, records, dp) for dp in (0.88, 0.95)}
    candidates = []
    for dp in cache:
        obs, state = cache[dp]
        for exposure in (False, True):
            for kp in (2.0, 5.0, 10.0):
                for kt in (5.0, 20.0, 50.0):
                    config = Config(dp=dp, exposure=exposure, kp=kp, kt=kt, protect=0.0, map=0.0)
                    result = score(obs, state, config, series=inner)
                    candidates.append((result['ce'], config, result))
    frozen = min(candidates, key=lambda row: row[0])[1]
    obs, state = cache[frozen.dp]
    extensions = []
    for rp in (0.0, 0.5, 1.0):
        for protect in (0.0, 1.0, 2.0):
            for map_weight in (0.0, 0.5, 1.0):
                config = replace(frozen, role_pooling=rp, protect=protect, map=map_weight)
                result = score(obs, state, config, series=inner)
                extensions.append((result['ce'], config, result))
    # corrected_baseline is old pooling (role_pooling=0) with its own best-tuned
    # protect/map, from the same search space selected is drawn from -- so the
    # reported comparison isolates role-aware pooling rather than conflating it
    # with retuned conditioning strength.
    baseline = min((row for row in extensions if row[1].role_pooling == 0.0), key=lambda row: row[0])[1]
    selected = min(extensions, key=lambda row: row[0])[1]
    print('Selected using development series only:', asdict(selected), flush=True)
    print('Historical replay has been reused in prior experiments; it is not a fresh holdout.', flush=True)
    metrics, differences, fixtures = {}, {}, []
    for stage in ('opening', 'first_protects', 'second_blind_bans', 'complete'):
        ref, ref_blocks, _ = score(obs, state, baseline, stage, replay, details=True)
        new, new_blocks, examples = score(obs, state, selected, stage, replay, details=True)
        metrics[stage] = {'corrected_baseline': ref, 'selected': new}
        differences[stage] = paired_intervals(ref_blocks, new_blocks)
        fixtures.extend(examples)
        print(stage, json.dumps(metrics[stage]), flush=True)
    node = subprocess.run(['node', str(Path(__file__).with_name('check_lineup.js'))],
                          input=json.dumps(fixtures), text=True, capture_output=True, check=True)
    print(node.stdout.strip(), flush=True)
    payload = export_state(state, selected, records, rejected)
    replay_records = [r for r in records if r['match_id'] in replay]
    validation = {'status': 'historical_replay_not_untouched_holdout',
                  'protocol': 'Chronological per-map state updates; known recorded rosters; whole-series development split; parameters selected before replay scoring.',
                  'target': 'Per-player map-level hero playtime share; unique primary boxes are a presentation constraint, not a joint lineup probability.',
                  'limitations': ['The 45-series replay window has been reused in earlier repository experiments.',
                                  'Metrics assume the recorded roster is known; last-known roster overlap is reported separately.',
                                  'Per-map refresh in this replay is more frequent than a static website export.',
                                  'Historical hero availability is not supplied. Exposure is adjusted for observed bans, not release or event eligibility.',
                                  'These probabilities do not measure win-rate effects or the value of forcing a ban.'],
                  'inner_series': len(inner), 'inner_maps': sum(r['match_id'] in inner for r in records),
                  'replay_series': len(replay), 'replay_maps': len(replay_records),
                  'replay_start': replay_records[0]['played_at'], 'replay_end': replay_records[-1]['played_at'],
                  'stages': {stage: value['selected'] for stage, value in metrics.items()}}
    payload['validation'] = validation
    report = {'generated_at': datetime.now(timezone.utc).isoformat(), 'data': payload['data_summary'],
              'records_sha256': hashlib.sha256(json.dumps(
                  sorted((r['map_uid'], r['played_at']) for r in records),
                  separators=(',', ':')).encode()).hexdigest(),
              'baseline_config': asdict(baseline), 'selected_config': asdict(selected),
              'validation': validation, 'metrics': metrics, 'paired_differences': differences,
              'development_base_grid': [{'config': asdict(c), 'metrics': r} for _, c, r in candidates],
              'development_extension_grid': [{'config': asdict(c), 'metrics': r} for _, c, r in extensions],
              'runtime_parity': {'fixtures': len(fixtures), 'status': 'passed'}}
    for filename, data in [(args.output, report), (args.export, payload)]:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    if args.embed:
        embed(args.embed, payload)
    print('Saved evaluation and full lineup distributions.', flush=True)


if __name__ == '__main__':
    main()
