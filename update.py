"""One guarded update of the tracker when new matches exist.

    python update.py [--data-root DIR] [--phi 0.25] [--skip-fetch] [--commit]

Refreshes the research repo (fetch, discover events, hero scan, build), halts
on anything that needs a human (an unknown hero, a roster gap, a changed draft
format, a failed test), scores the currently embedded model prospectively on
the series played since its data cutoff and appends that to
evaluation/prospective_log.json, then refits, embeds, refits the lineup boxes
and runs the unit tests. Nothing is pushed; --commit commits both repos.
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path

import fit_draft as fd

HERE = Path(__file__).resolve().parent
PAGE = HERE / "index.html"
LOG = HERE / "evaluation" / "prospective_log.json"
TOLERATED_HEROES = {"hero-0"}   # mrvl.net's unknown-hero placeholder


# ---------- guards (pure) ----------
def unknown_heroes_from(output):
    found = []
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0] and not line.startswith("no heroes"):
            slug = parts[0].strip()
            if slug not in TOLERATED_HEROES:
                found.append((slug, parts[1].strip()))
    return found


_REJECT = re.compile(r"^- REJECT (\S+): (.*)$")
_PASS = re.compile(r"^(\d+)/(\d+) maps pass")


def qa_guards(report):
    out = {"status": "ok", "reasons": [], "warnings": [], "passed": None, "total": None,
           "missing": 0}
    for line in report.splitlines():
        m = _PASS.match(line)
        if m:
            out["passed"], out["total"] = int(m.group(1)), int(m.group(2))
        if line.startswith("- MISSING raw page"):
            out["missing"] += 1
        m = _REJECT.match(line)
        if not m:
            continue
        uid, reasons = m.group(1), m.group(2)
        if "structure mismatch" in reasons or "draft choices" in reasons:
            out["reasons"].append(f"{uid}: draft format looks different ({reasons}) -- "
                                  f"a new schedule needs a design decision, not a refit")
        elif "heroes not in legal roster" in reasons:
            heroes = set(re.findall(r"'([^']+)'", reasons))
            if heroes - TOLERATED_HEROES:
                out["reasons"].append(f"{uid}: roster lacks {sorted(heroes - TOLERATED_HEROES)} -- "
                                      f"add the hero (release date) or fix the event roster")
        else:
            out["warnings"].append(f"{uid}: {reasons}")
    if out["reasons"]:
        out["status"] = "halt"
    return out


def new_series_after(series_order, decisions, last_seen):
    first = {}
    for d in decisions:
        first[d["series"]] = min(first.get(d["series"], d["t"]), d["t"])
    return [s for s in series_order if first.get(s, 0) > last_seen]


def append_log(path, entry):
    path = Path(path)
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=1) + "\n")


def coef_delta(old, new):
    rows = []
    for kind in ("ban", "protect"):
        o = next(iter(old["coef"][kind].values())); n = next(iter(new["coef"][kind].values()))
        for nm in n:
            if o.get(nm) != n[nm]:
                rows.append((kind, nm, o.get(nm), n[nm]))
        for sl in new["temps"][kind]:
            if old["temps"][kind].get(sl) != new["temps"][kind][sl]:
                rows.append((f"temp {kind}", sl, old["temps"][kind].get(sl), new["temps"][kind][sl]))
    return rows


# ---------- steps ----------
def research_python(root):
    venv = Path(root) / ".venv" / "bin" / "python"
    return str(venv) if venv.exists() else sys.executable


def run(cmd, cwd, label):
    print(f"\n== {label}: {' '.join(cmd)}", flush=True)
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    print(out.rstrip()[-3000:], flush=True)
    if p.returncode != 0:
        halt(f"{label} failed (exit {p.returncode})")
    return out


def halt(msg):
    print(f"\nHALT: {msg}", flush=True)
    sys.exit(2)


def embedded_model(page=PAGE):
    return json.loads(fd._MODEL_RE.search(Path(page).read_text()).group(2))


def metrics(tot, per_slot):
    out = {}
    for kind, (nl, n, t1, t3) in tot.items():
        if n:
            out[kind] = {"n": n, "ll": round(nl / n, 4), "top1": round(t1 / n, 4), "top3": round(t3 / n, 4)}
    out["per_slot_ll"] = {sl: round(v[0] / v[1], 4) for sl, v in sorted(per_slot.items()) if v[1]}
    return out


def prospective(root, phi, model):
    roles = fd.load_roles(root)
    heroes = fd.heroes_as_of(roles, fd.load_releases(root), None)
    recs, _ = fd.load_maps(root)
    last_seen = model["draft_data_summary"]["last_seen"]
    decisions, st = fd.feature_pass(recs, heroes, roles, phi=phi)
    series = new_series_after(st.series_order, decisions, last_seen)
    entry = {"date": time.strftime("%Y-%m-%d", time.gmtime()),
             "model": model["fitted_on"], "scored_after": last_seen,
             "new_series": len(series),
             "new_maps": sum(r["match_id"] in set(series) for r in recs)}
    if not series:
        entry["note"] = "no new series since the embedded model's data cutoff"
        print("prospective: no new series to score", flush=True)
        return entry
    for variant in ("live", "share"):
        tot, ps, _ = fd.run_variant(variant, recs, heroes, roles, phi, model=model, log=False,
                                    after_ts=last_seen, block=5)
        entry[variant] = metrics(tot, ps)
        print(fd.format_result(variant, tot, ps), flush=True)
    return entry


def refit_and_embed(root, phi):
    roles = fd.load_roles(root)
    heroes = fd.heroes_as_of(roles, fd.load_releases(root), None)
    recs, excluded = fd.load_maps(root)
    decisions, st = fd.feature_pass(recs, heroes, roles, phi=phi, log=True, **fd.VARIANTS["share"])
    end = max(d["t"] for d in decisions)
    fits = fd.fit_hybrid(decisions, end, st.NH, log=True)
    fitted_on = (f"{len(recs)} maps through {time.strftime('%Y-%m-%d', time.gmtime(end))}; "
                 f"hybrid v5: v4 hybrid + five-ban format (B5 share), Season 10 break")
    params = fd.export_params(st, fits, roles, fitted_on, fd.data_summary(recs, excluded),
                              **fd.VARIANTS["share"])
    PAGE.write_text(fd.embed_draft(PAGE.read_text(), params))
    print(f"embedded draft model: {fitted_on}", flush=True)
    return params


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=fd.DRAFT_ROOT)
    ap.add_argument("--phi", type=float, default=0.25)
    ap.add_argument("--skip-fetch", action="store_true", help="use the research repo's data as is")
    ap.add_argument("--commit", action="store_true", help="commit both repos when everything passes")
    args = ap.parse_args(argv)
    root = args.data_root
    py = research_python(root)

    before = embedded_model()
    if not args.skip_fetch:
        run([py, "-m", "src.ingest", "fetch"], root, "fetch")
        run([py, "-m", "src.ingest", "discover"], root, "discover events")
    unknown = unknown_heroes_from(run([py, "-m", "src.ingest", "heroes"], root, "hero scan"))
    if unknown:
        halt("unknown heroes on archived pages: " + ", ".join(f"{s} ({r})" for s, r in unknown)
             + " -- add them to data/roles/heroes.json and releases.json, then rerun")
    run([py, "-m", "src.ingest", "build"], root, "build")
    guards = qa_guards((Path(root) / "outputs/tables/qa_report.md").read_text())
    print(f"QA: {guards['passed']}/{guards['total']} pass, {guards['missing']} matches not yet archived")
    for w in guards["warnings"]:
        print("  warning:", w)
    if guards["status"] == "halt":
        halt("\n  ".join(guards["reasons"]))

    entry = prospective(root, args.phi, before)
    append_log(LOG, entry)
    print(f"prospective entry appended to {LOG.relative_to(HERE)}", flush=True)

    after = refit_and_embed(root, args.phi)
    run([sys.executable, "evaluate_lineup.py", "--data-root", root, "--embed", str(PAGE)], HERE, "lineup embed")
    run([sys.executable, "-m", "unittest", "test_fit_draft", "test_lineup", "test_update"], HERE, "unit tests")

    print("\n== summary")
    ds_b, ds_a = before["draft_data_summary"], after["data_summary"]
    print(f"data: {ds_b['maps']} -> {ds_a['maps']} maps, {ds_b['series']} -> {ds_a['series']} series, "
          f"through {time.strftime('%Y-%m-%d', time.gmtime(ds_a['last_seen']))}")
    for row in coef_delta(before, after):
        print(f"  {row[0]:13s} {row[1]:8s} {row[2]} -> {row[3]}")
    if args.commit:
        run(["git", "add", "data", "outputs/tables/qa_report.md"], root, "stage research data")
        run(["git", "commit", "-q", "-m", f"data: update through {time.strftime('%Y-%m-%d', time.gmtime(ds_a['last_seen']))}"],
            root, "commit research data")
        run(["git", "add", "index.html", "evaluation/lineup_report.json", str(LOG.relative_to(HERE))], HERE, "stage tracker")
        run(["git", "commit", "-q", "-m", f"Refit through {time.strftime('%Y-%m-%d', time.gmtime(ds_a['last_seen']))} ({ds_a['maps']} maps)"],
            HERE, "commit tracker")
    print("\nupdate complete; nothing pushed", flush=True)


if __name__ == "__main__":
    main()
