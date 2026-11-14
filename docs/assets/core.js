/* Pure helpers for the PM2.5 forecast explorer (no DOM). Browser: window.PM25Core; Node: require. */
"use strict";

(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PM25Core = factory();
})(typeof self !== "undefined" ? self : this, function () {
  const HOUR_MS = 3600 * 1000;
  const TAIPEI_OFFSET_MS = 8 * HOUR_MS;

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  /** Trailing `window`-slot mean of |pred - obs|; null where the slot itself is missing
   *  or fewer than `minValid` slots in the window have both values. */
  function rollingAbsError(pred, obs, window, minValid) {
    if (pred.length !== obs.length) throw new Error("pred and obs length differ");
    const err = pred.map((p, i) => (isNum(p) && isNum(obs[i]) ? Math.abs(p - obs[i]) : null));
    const out = new Array(err.length).fill(null);
    let sum = 0;
    let count = 0;
    for (let i = 0; i < err.length; i += 1) {
      if (err[i] !== null) { sum += err[i]; count += 1; }
      const drop = i - window;
      if (drop >= 0 && err[drop] !== null) { sum -= err[drop]; count -= 1; }
      if (err[i] !== null && count >= minValid) out[i] = sum / count;
    }
    return out;
  }

  /** One observed series from the per-lead arrays (same truth, different availability). */
  function mergeObserved(series) {
    const size = Math.max(...series.map((s) => s.length));
    return Array.from({ length: size }, (_, i) => {
      const hit = series.find((s) => isNum(s[i]));
      return hit ? hit[i] : null;
    });
  }

  function runsAbove(y, threshold) {
    const runs = [];
    let start = -1;
    for (let i = 0; i <= y.length; i += 1) {
      const above = i < y.length && isNum(y[i]) && y[i] >= threshold;
      if (above && start < 0) start = i;
      if (!above && start >= 0) {
        const slice = y.slice(start, i);
        runs.push({ start, end: i - 1, hours: i - start, peak: Math.max(...slice) });
        start = -1;
      }
    }
    return runs;
  }

  /** Up to `maxCount` longest runs with y >= threshold, more than `minGap` slots apart. */
  function findEpisodes(y, threshold, maxCount, minGap) {
    const ranked = runsAbove(y, threshold)
      .sort((a, b) => b.hours - a.hours || b.peak - a.peak || a.start - b.start);
    const picked = [];
    for (const run of ranked) {
      if (picked.length >= maxCount) break;
      const clash = picked.some((p) => run.start - p.end <= minGap && p.start - run.end <= minGap);
      if (!clash) picked.push(run);
    }
    return picked;
  }

  /** [from, to) slot range of `width` centred on an episode, clamped to [0, size). */
  function windowAround(episode, width, size) {
    const centre = (episode.start + episode.end) / 2;
    let from = Math.floor(centre - width / 2);
    from = Math.max(0, Math.min(from, size - width));
    return [from, Math.min(size, from + width)];
  }

  /** "M/D" in Taiwan time for slot i of a series starting at startMs (UTC instant). */
  function formatSlot(startMs, i) {
    const d = new Date(startMs + i * HOUR_MS + TAIPEI_OFFSET_MS);
    return `${d.getUTCMonth() + 1}/${d.getUTCDate()}`;
  }

  return { HOUR_MS, TAIPEI_OFFSET_MS, rollingAbsError, mergeObserved, findEpisodes, windowAround, formatSlot };
});
