"use strict";
var assert = require('node:assert/strict');
var engine = require('./lineup-engine.js');
var fixtures = JSON.parse(require('node:fs').readFileSync(0, 'utf8'));
fixtures.forEach(function (f) {
  var rows = engine.project(f.players, f.banned, f.protected, f.offsets, f.config);
  rows.forEach(function (row, i) {
    assert.equal(row.primary && row.primary.hero, f.expected.primary[i]);
    assert.equal(row.alternate && row.alternate.hero, f.expected.alternate[i]);
    row.q.forEach(function (e, j) { assert.ok(Math.abs(e[1] - f.expected.q[i][j]) < 2e-12); });
  });
});
var rows = engine.project([{name:'A', q:[['a',0.6],['b',0.4]]}, {name:'B',q:[['a',1]]}], [], [], {}, {});
assert.deepEqual(rows.map(function (r) { return r.primary.hero; }), ['b','a']);
rows = engine.project([{name:'A',q:[['a',0.6],['b',0.25],['c',0.15]]}], [], ['c'], {}, {protect:2});
assert.equal(rows[0].primary.hero, 'c');
assert.equal(rows[0].alternate.hero, 'a');
rows = engine.project([{name:'A',q:[['a',1]]},{name:'B',q:[]}], ['a'], [], {}, {});
assert.deepEqual(rows.map(function (r) { return r.primary; }), [null, null]);
assert.ok(rows.every(function (r) { return r.q.every(function (e) { return e[1] === 0; }); }));
console.log('JavaScript/Python parity passed for ' + fixtures.length + ' draft states; assignment, conditioning and empty-pool checks passed.');
