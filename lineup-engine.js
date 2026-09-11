(function (root) {
  "use strict";

  function assignment(cost) {
    var n = cost.length, m = n ? cost[0].length : 0;
    var u = Array(n + 1).fill(0), v = Array(m + 1).fill(0);
    var p = Array(m + 1).fill(0), way = Array(m + 1).fill(0);
    for (var i = 1; i <= n; i++) {
      p[0] = i;
      var j0 = 0, minv = Array(m + 1).fill(Infinity), used = Array(m + 1).fill(false);
      do {
        used[j0] = true;
        var i0 = p[j0], delta = Infinity, j1 = 0;
        for (var j = 1; j <= m; j++) {
          if (used[j]) continue;
          var cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
          if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
          if (minv[j] < delta) { delta = minv[j]; j1 = j; }
        }
        for (var k = 0; k <= m; k++) {
          if (used[k]) { u[p[k]] += delta; v[k] -= delta; }
          else minv[k] -= delta;
        }
        j0 = j1;
      } while (p[j0] !== 0);
      do {
        var prev = way[j0]; p[j0] = p[prev]; j0 = prev;
      } while (j0 !== 0);
    }
    var out = Array(n).fill(null);
    for (var col = 1; col <= m; col++) if (p[col]) out[p[col] - 1] = col - 1;
    return out;
  }

  function project(players, banned, protectedHeroes, offsets, config) {
    config = config || {};
    offsets = offsets || {};
    var bp = config.protect === undefined ? 1 : config.protect;
    var lm = config.map === undefined ? 0.5 : config.map;
    var altMin = config.alternative_min === undefined ? 0.10 : config.alternative_min;
    var blocked = new Set(banned || []), protectedSet = new Set(protectedHeroes || []);
    var heroes = Array.from(new Set(players.flatMap(function (p) { return p.q.map(function (e) { return e[0]; }); }))).sort();
    var rows = players.map(function (player) {
      var weights = Object.create(null), total = 0;
      player.q.forEach(function (e) {
        if (blocked.has(e[0]) || !Number.isFinite(e[1]) || e[1] <= 0) return;
        var w = e[1] * Math.exp(bp * (protectedSet.has(e[0]) ? 1 : 0) + lm * (offsets[e[0]] || 0));
        weights[e[0]] = (weights[e[0]] || 0) + w;
        total += w;
      });
      var q = heroes.map(function (h) { return total > 0 ? (weights[h] || 0) / total : 0; });
      return { name: player.name, q: q };
    });
    var cost = rows.map(function (r) {
      return r.q.map(function (q) { return q > 0 ? Number((-Math.log(q)).toFixed(12)) : 1e6; }).concat(Array(players.length).fill(5e5));
    });
    var picks = assignment(cost);
    return rows.map(function (r, i) {
      var pick = picks[i];
      var primary = pick !== null && pick < heroes.length && r.q[pick] > 0 ? {hero: heroes[pick], p: r.q[pick]} : null;
      var ranked = heroes.map(function (h, j) { return {hero: h, p: r.q[j]}; })
        .filter(function (e) { return e.p > 0; })
        .sort(function (a, b) { return b.p - a.p || (a.hero < b.hero ? -1 : 1); });
      var alternate = ranked.find(function (e) { return (!primary || e.hero !== primary.hero) && e.p >= altMin; }) || null;
      return {name: r.name, primary: primary, alternate: alternate, ranked: ranked,
              q: heroes.map(function (h, j) { return [h, r.q[j]]; })};
    });
  }

  var api = {project: project};
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.LineupEngine = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
