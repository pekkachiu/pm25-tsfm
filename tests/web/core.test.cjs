// Unit tests for docs/assets/core.js (pure helpers of the web demo). Run: node --test tests/web
const test = require("node:test");
const assert = require("node:assert/strict");
const core = require("../../docs/assets/core.js");

test("rollingAbsError: trailing mean of |pred - obs|", () => {
  const obs = [10, 10, 10, 10];
  const pred = [12, 8, 10, 14];
  const out = core.rollingAbsError(pred, obs, 2, 1);
  assert.deepEqual(out, [2, 2, 1, 2]);
});

test("rollingAbsError: missing slots stay null and are skipped in the window", () => {
  const obs = [10, null, 10, 10];
  const pred = [12, 9, 13, 10];
  const out = core.rollingAbsError(pred, obs, 3, 1);
  assert.equal(out[1], null);
  assert.equal(out[2], 2.5); // (2 + 3) / 2, the null slot is skipped
  assert.equal(out[3], 1.5); // (3 + 0) / 2
});

test("rollingAbsError: needs minValid observations in the window", () => {
  const out = core.rollingAbsError([1, 1, 1], [0, 0, 0], 24, 2);
  assert.deepEqual(out, [null, 1, 1]);
});

test("rollingAbsError: rejects mismatched lengths", () => {
  assert.throws(() => core.rollingAbsError([1], [1, 2], 2, 1), /length/);
});

test("mergeObserved: first non-null value per slot across leads", () => {
  assert.deepEqual(core.mergeObserved([[1, null, null], [null, 2, null]]), [1, 2, null]);
});

test("findEpisodes: longest runs at or above the threshold, longest first", () => {
  const y = [5, 40, 40, 5, 31, 31, 31, 5, 30.5, 5];
  const out = core.findEpisodes(y, 30.5, 3, 0);
  assert.deepEqual(out, [
    { start: 4, end: 6, hours: 3, peak: 31 },
    { start: 1, end: 2, hours: 2, peak: 40 },
    { start: 8, end: 8, hours: 1, peak: 30.5 },
  ]);
});

test("findEpisodes: null breaks a run", () => {
  const out = core.findEpisodes([40, null, 40], 30.5, 5, 0);
  assert.equal(out.length, 2);
  assert.equal(out[0].hours, 1);
});

test("findEpisodes: keeps episodes at least minGap slots apart", () => {
  const y = [40, 40, 40, 5, 40, 5, 5, 5, 5, 40];
  const out = core.findEpisodes(y, 30.5, 3, 4);
  // the 1-hour run at 4 is within 4 slots of the first episode, so it is dropped
  assert.deepEqual(out.map((e) => e.start), [0, 9]);
});

test("findEpisodes: nothing above the threshold", () => {
  assert.deepEqual(core.findEpisodes([1, 2, 3], 30.5, 3, 0), []);
});

test("windowAround: fixed-width window centred on an episode, clamped to the data", () => {
  assert.deepEqual(core.windowAround({ start: 100, end: 110 }, 168, 1000), [21, 189]);
  assert.deepEqual(core.windowAround({ start: 5, end: 10 }, 168, 1000), [0, 168]);
  assert.deepEqual(core.windowAround({ start: 990, end: 995 }, 168, 1000), [832, 1000]);
});

test("formatSlot: Taiwan wall-clock date of a slot", () => {
  const start = Date.parse("2025-01-01T01:00:00+08:00");
  assert.equal(core.formatSlot(start, 0), "1/1");
  assert.equal(core.formatSlot(start, 24 * 31), "2/1");
});
