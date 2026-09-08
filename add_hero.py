"""Insert a post-fit hero into the tracker's embedded model params.

A hero released after the model's data cutoff has no competitive history, so
every hero-indexed vector gets a no-evidence value at that hero's SORTED index
(the page derives HEROES from Object.keys(roles).sort(), so appending would
silently shift every later hero's parameters).

  python add_hero.py the-hood Vanguard 2026-08-07

Re-running is idempotent. Once real data exists for the hero, drop them from
NEW_SINCE_FIT in index.html and re-export params normally.
"""
import json, re, sys

def insert(params, slug, role):
    heroes = params["heroes"]
    if slug in heroes:
        return params, heroes.index(slug), False
    idx = sorted(heroes + [slug]).index(slug)

    def put(vec, val):
        vec.insert(idx, val)

    for sl, a in params["alpha"].items():
        put(a, round(min(a), 4))                 # structural floor: unseen-hero prior
    put(params["g"], round(min(params["g"]), 6))
    for mp, v in params["map_offsets"].items():
        put(v, 0.0)                              # no map evidence -> neutral
    for t in params["teams"].values():
        put(t["cap"], round(min(t["cap"]), 5))
        put(t["thr"], 0.0)                       # no win-rate evidence
        put(t["ls"], 0.0)                        # no playtime
        put(t["sb"], 0.0)                        # no ban habit
        put(t["sp"], 0.0)                        # no protect habit
    heroes.insert(idx, slug)
    params["roles"][slug] = role
    return params, idx, True

def main():
    slug, role, released = sys.argv[1], sys.argv[2], sys.argv[3]
    page = open("index.html").read()
    m = re.search(r'(<script id="modelData" type="application/json">)(.*?)(</script>)', page, re.S)
    params = json.loads(m.group(2))
    n_before = len(params["heroes"])
    params, idx, added = insert(params, slug, role)
    page = page[:m.start(2)] + json.dumps(params) + page[m.end(2):]

    # register for the UI's "no competitive data" flag
    entry = '{slug:"%s",released:"%s"}' % (slug, released)
    mm = re.search(r'var NEW_SINCE_FIT = \[(.*?)\];', page)
    if mm:
        if slug not in mm.group(1):
            inner = (mm.group(1) + "," + entry) if mm.group(1).strip() else entry
            page = page[:mm.start(1)] + inner + page[mm.end(1):]
    else:
        page = page.replace("var HEROES = Object.keys(ROLES).sort();",
                            "var NEW_SINCE_FIT = [%s];\nvar HEROES = Object.keys(ROLES).sort();" % entry)
    open("index.html", "w").write(page)
    print(f"{slug} ({role}, released {released}) -> index {idx}; roster {n_before} -> {len(params['heroes'])}"
          + ("" if added else "  [already present]"))

main()
