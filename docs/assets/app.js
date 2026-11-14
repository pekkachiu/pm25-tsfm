/* PM2.5 forecast explorer - static page, data from data/*.json (same origin). */
"use strict";

(function () {
  const core = window.PM25Core || null;
  const DATA_ROOT = "data/";
  const HOUR_MS = 3600 * 1000;
  const TAIPEI_OFFSET_MS = 8 * HOUR_MS; // shifted instants let the UTC axis read Taiwan time
  const THRESHOLD = 30.5;
  const EPISODE_COUNT = 3;
  const EPISODE_WINDOW_HOURS = 7 * 24; // one week around each episode
  const EPISODE_MIN_GAP_HOURS = 7 * 24;
  const ERROR_WINDOW_HOURS = 24; // rolling window of the error view
  const ERROR_MIN_VALID = 12;
  const FADED_OPACITY = 0.15;
  const DEFAULT_VISIBLE = ["chronos2_ft_nwp", "xgboost_nwp", "chronos2_uni"];
  const DASHED = new Set(["persistence"]);
  const FINETUNED = "chronos2_ft_nwp";
  const ZERO_SHOT = "chronos2_uni";
  const TRAINED = "xgboost_nwp";
  const OBSERVED_LABEL = "觀測值";
  const VIEWS = [
    { id: "value", label: "預測值" },
    { id: "error", label: "誤差" },
  ];
  const NOTES = {
    value:
      "虛線為 PM2.5 30.5 μg/m³（AQI 100，對敏感族群不健康）。每個點是「H 小時前」做出的預測中位數；色帶為所選模型的 10%–90% 分位數區間。拖曳下方滑桿可平移、縮放時間。",
    error:
      "每條線是該模型過去 24 小時的平均絕對誤差 |預測 − 觀測|，越低越準。拖曳下方滑桿可平移、縮放時間。",
  };

  const state = {
    index: null,
    station: null,
    year: null,
    lead: 24,
    view: "value",
    period: "ep0", // ep<k> | m<month> | all
    band: FINETUNED,
    focus: null,
    visible: new Set(DEFAULT_VISIBLE),
    data: null,
    episodes: [],
    observed: [],
  };
  const cache = new Map();
  let chart = null;
  let map = null;
  const markers = new Map();

  const $ = (id) => document.getElementById(id);

  function cssVar(name) {
    return getComputedStyle(document.documentElement)
      .getPropertyValue(name)
      .trim();
  }

  function modelColor(id) {
    return cssVar(`--m-${id}`) || cssVar("--text-secondary");
  }

  function modelLabel(id) {
    const model = state.index.models.find((m) => m.id === id);
    return model ? model.label : id;
  }

  function escapeHtml(text) {
    return String(text).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  }

  function setStatus(text) {
    $("status").textContent = text;
  }

  // ---------- data ----------

  async function fetchJson(path) {
    const response = await fetch(DATA_ROOT + path, {
      credentials: "same-origin",
    });
    if (!response.ok)
      throw new Error(`讀取 ${path} 失敗（HTTP ${response.status}）`);
    return response.json();
  }

  function validateIndex(index) {
    const ok =
      index &&
      Array.isArray(index.models) &&
      Array.isArray(index.years) &&
      Array.isArray(index.leads) &&
      Array.isArray(index.stations) &&
      index.stations.length > 0;
    if (!ok) throw new Error("index.json 格式不符");
    return index;
  }

  function validateStation(data) {
    const ok =
      data &&
      typeof data.start === "string" &&
      data.leads &&
      state.index.leads.every(
        (h) => data.leads[String(h)] && Array.isArray(data.leads[String(h)].y),
      );
    if (!ok) throw new Error("測站資料格式不符");
    return data;
  }

  async function loadStation(station, year) {
    const entry = state.index.stations.find((s) => s.name === station);
    const path = entry && entry.file[String(year)];
    if (!path || !/^\d{4}\/[A-Za-z]+\.json$/.test(path))
      throw new Error("找不到測站資料");
    if (!cache.has(path))
      cache.set(path, fetchJson(path).then(validateStation));
    return cache.get(path);
  }

  function startMsOf(data) {
    return Date.parse(data.start) + TAIPEI_OFFSET_MS;
  }

  // ---------- controls ----------

  function option(value, label) {
    const el = document.createElement("option");
    el.value = String(value);
    el.textContent = label;
    return el;
  }

  function segmented(groupId, name, items, current, onChange) {
    const group = $(groupId);
    items.forEach((item) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = name;
      input.value = String(item.id);
      input.checked = item.id === current;
      input.addEventListener("change", () => onChange(item.id));
      label.append(input, document.createTextNode(item.label));
      group.append(label);
    });
  }

  function buildControls() {
    const { index } = state;
    const stationSelect = $("station-select");
    index.stations.forEach((s) =>
      stationSelect.append(option(s.name, `${s.name}（${s.region_zh}）`)),
    );
    stationSelect.addEventListener("change", () =>
      selectStation(stationSelect.value),
    );

    const yearSelect = $("year-select");
    index.years
      .slice()
      .sort((a, b) => b - a)
      .forEach((y) => yearSelect.append(option(y, `${y}`)));
    yearSelect.addEventListener("change", () => {
      state.year = Number(yearSelect.value);
      refresh();
    });

    const leads = index.leads.map((h) => ({ id: h, label: `${h} 小時` }));
    segmented("lead-group", "lead", leads, state.lead, (h) => {
      state.lead = h;
      render();
    });
    segmented("view-group", "view", VIEWS, state.view, (v) => {
      state.view = v;
      render();
    });

    $("period-select").addEventListener("change", () => {
      state.period = $("period-select").value;
      applyZoom();
    });

    const bandSelect = $("band-select");
    bandSelect.append(option("", "不顯示"));
    index.models.forEach((m) => bandSelect.append(option(m.id, m.label)));
    bandSelect.value = state.band;
    bandSelect.addEventListener("change", () => {
      state.band = bandSelect.value;
      render();
    });
    buildModelToggles();
  }

  function keySwatch(id) {
    const key = document.createElement("span");
    key.className = DASHED.has(id) ? "key dashed" : "key";
    key.dataset.model = id;
    return key;
  }

  function buildModelToggles() {
    const toggles = $("model-toggles");
    const observed = document.createElement("span");
    observed.className = "model-toggle observed-chip";
    const observedKey = document.createElement("span");
    observedKey.className = "key observed";
    observed.append(observedKey, document.createTextNode(OBSERVED_LABEL));
    toggles.append(observed);
    state.index.models.forEach((m) => {
      const chip = document.createElement("span");
      chip.className = "model-toggle";
      chip.dataset.model = m.id;
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.visible.has(m.id);
      input.setAttribute("aria-label", `顯示 ${m.label}`);
      input.addEventListener("change", () => setVisible(m.id, input.checked));
      const button = document.createElement("button");
      button.type = "button";
      button.className = "focus-btn";
      button.setAttribute("aria-pressed", "false");
      button.title = "聚焦這個模型（再點一次取消）";
      button.append(keySwatch(m.id), document.createTextNode(m.label));
      button.addEventListener("click", () => toggleFocus(m.id));
      chip.append(input, button);
      toggles.append(chip);
    });
    paintKeys();
  }

  function setVisible(id, visible) {
    if (visible) state.visible.add(id);
    else state.visible.delete(id);
    if (!visible && state.focus === id) state.focus = null;
    render();
  }

  function toggleFocus(id) {
    state.focus = state.focus === id ? null : id;
    if (state.focus) state.visible.add(id);
    render();
  }

  function syncModelToggles() {
    document.querySelectorAll(".model-toggle[data-model]").forEach((chip) => {
      const id = chip.dataset.model;
      chip.querySelector("input").checked = state.visible.has(id);
      chip
        .querySelector("button")
        .setAttribute("aria-pressed", String(state.focus === id));
      chip.classList.toggle(
        "faded",
        state.focus !== null && state.focus !== id,
      );
    });
    $("model-toggles").classList.toggle("has-focus", state.focus !== null);
  }

  function paintKeys() {
    document.querySelectorAll(".key[data-model]").forEach((el) => {
      el.style.setProperty("--key-color", modelColor(el.dataset.model));
    });
  }

  // ---------- periods (episodes / months / year) ----------

  function computeEpisodes(data) {
    state.observed = core.mergeObserved(
      state.index.leads.map((h) => data.leads[String(h)].y),
    );
    state.episodes = core.findEpisodes(
      state.observed,
      THRESHOLD,
      EPISODE_COUNT,
      EPISODE_MIN_GAP_HOURS,
    );
  }

  function buildPeriodOptions(data) {
    const select = $("period-select");
    select.replaceChildren();
    const startMs = Date.parse(data.start);
    if (state.episodes.length > 0) {
      const group = document.createElement("optgroup");
      group.label = "高污染事件（一週）";
      state.episodes.forEach((ep, k) => {
        const from = core.formatSlot(startMs, ep.start);
        const to = core.formatSlot(startMs, ep.end);
        const span = from === to ? from : `${from}–${to}`;
        group.append(option(`ep${k}`, `${span}，連續 ${ep.hours} 小時 ≥ 30.5`));
      });
      select.append(group);
    }
    const months = document.createElement("optgroup");
    months.label = "月份";
    for (let m = 0; m < 12; m += 1)
      months.append(option(`m${m}`, `${m + 1} 月`));
    select.append(months, option("all", "全年"));
    const valid = [...select.querySelectorAll("option")].some(
      (o) => o.value === state.period,
    );
    if (!valid) state.period = state.episodes.length > 0 ? "ep0" : "m0";
    select.value = state.period;
  }

  function zoomRange() {
    if (!state.data) return null;
    const startMs = startMsOf(state.data);
    const size = state.observed.length;
    if (state.period.startsWith("ep")) {
      const ep = state.episodes[Number(state.period.slice(2))];
      if (!ep) return null;
      const [from, to] = core.windowAround(ep, EPISODE_WINDOW_HOURS, size);
      return [startMs + from * HOUR_MS, startMs + (to - 1) * HOUR_MS];
    }
    if (state.period.startsWith("m")) {
      const m = Number(state.period.slice(1));
      return [Date.UTC(state.year, m, 1), Date.UTC(state.year, m + 1, 1)];
    }
    return null;
  }

  function applyZoom() {
    if (!chart) return;
    const range = zoomRange();
    chart.dispatchAction(
      range
        ? { type: "dataZoom", startValue: range[0], endValue: range[1] }
        : { type: "dataZoom", start: 0, end: 100 },
    );
  }

  // ---------- map ----------

  function buildMap() {
    const container = $("map");
    if (typeof L === "undefined") {
      container.textContent = "地圖載入失敗，請改用上方的測站選單。";
      return;
    }
    map = L.map(container, {
      scrollWheelZoom: false,
      attributionControl: true,
    }).setView([23.65, 120.85], 7);
    // OpenStreetMap tiles need no API key; their usage policy requires a Referer, so tiles
    // send the page origin only (the page itself sends no referrer anywhere else).
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 12,
      minZoom: 6,
      referrerPolicy: "strict-origin",
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    }).addTo(map);
    state.index.stations.forEach((s) => {
      const marker = L.circleMarker([s.lat, s.lon], {
        radius: 8,
        weight: 2,
      }).addTo(map);
      const tip = document.createElement("span");
      tip.textContent = `${s.name}（${s.region_zh}）`;
      marker.bindTooltip(tip, { direction: "top", offset: [0, -6] });
      marker.on("click", () => selectStation(s.name));
      markers.set(s.name, marker);
    });
    map.fitBounds(
      state.index.stations.map((s) => [s.lat, s.lon]),
      { padding: [24, 24] },
    );
    paintMarkers();
  }

  function paintMarkers() {
    const accent = cssVar("--accent");
    const surface = cssVar("--surface-1");
    const muted = cssVar("--text-muted");
    markers.forEach((marker, name) => {
      const selected = name === state.station;
      marker.setStyle({
        color: selected ? cssVar("--text-primary") : surface,
        fillColor: selected ? accent : muted,
        fillOpacity: selected ? 1 : 0.75,
        radius: selected ? 10 : 7,
      });
      if (selected) marker.bringToFront();
    });
  }

  // ---------- chart ----------

  function toPairs(values, startMs) {
    return values.map((v, i) => [startMs + i * HOUR_MS, v]);
  }

  function bandSeries(pred, startMs, color) {
    const width = pred.q90.map((hi, i) =>
      hi == null || pred.q10[i] == null ? null : hi - pred.q10[i],
    );
    const common = {
      type: "line",
      stack: "band",
      symbol: "none",
      silent: true,
      connectNulls: false,
      lineStyle: { opacity: 0 },
      emphasis: { disabled: true },
      z: 1,
    };
    return [
      { ...common, name: "_band_low", data: toPairs(pred.q10, startMs) },
      {
        ...common,
        name: "_band_width",
        data: toPairs(width, startMs),
        areaStyle: { color, opacity: 0.18 },
      },
    ];
  }

  function lineSeries(id, values, startMs) {
    const faded = state.focus !== null && state.focus !== id;
    return {
      name: modelLabel(id),
      type: "line",
      symbol: "none",
      connectNulls: false,
      sampling: "lttb",
      z: faded ? 2 : 3,
      data: toPairs(values, startMs),
      lineStyle: {
        width: state.focus === id ? 2.5 : 2,
        color: modelColor(id),
        opacity: faded ? FADED_OPACITY : 1,
        type: DASHED.has(id) ? "dashed" : "solid",
      },
      itemStyle: { color: modelColor(id) },
      emphasis: { disabled: true },
    };
  }

  function observedSeries(y, startMs) {
    const ink = cssVar("--text-primary");
    return {
      name: OBSERVED_LABEL,
      type: "line",
      symbol: "none",
      connectNulls: false,
      sampling: "lttb",
      z: 4,
      data: toPairs(y, startMs),
      lineStyle: { width: 2.5, color: ink },
      itemStyle: { color: ink },
      emphasis: { disabled: true },
      markLine: {
        silent: true,
        symbol: "none",
        lineStyle: { color: cssVar("--threshold"), type: "dashed", width: 1 },
        label: {
          formatter: "AQI 100",
          color: cssVar("--text-muted"),
          position: "insideEndTop",
        },
        data: [{ yAxis: THRESHOLD }],
      },
    };
  }

  function shownModels(lead) {
    return state.index.models
      .map((m) => m.id)
      .filter((id) => state.visible.has(id) && lead.models[id]);
  }

  function valueSeries(lead, startMs) {
    const series = [];
    const band = state.band ? state.focus || state.band : "";
    if (band && lead.models[band])
      series.push(...bandSeries(lead.models[band], startMs, modelColor(band)));
    shownModels(lead).forEach((id) =>
      series.push(lineSeries(id, lead.models[id].q50, startMs)),
    );
    series.push(observedSeries(lead.y, startMs));
    return series;
  }

  function errorSeries(lead, startMs) {
    return shownModels(lead).map((id) =>
      lineSeries(
        id,
        core.rollingAbsError(
          lead.models[id].q50,
          lead.y,
          ERROR_WINDOW_HOURS,
          ERROR_MIN_VALID,
        ),
        startMs,
      ),
    );
  }

  function formatTime(ms) {
    const d = new Date(ms);
    const pad = (n) => String(n).padStart(2, "0");
    return `${d.getUTCFullYear()}-${pad(d.getUTCMonth() + 1)}-${pad(d.getUTCDate())} ${pad(d.getUTCHours())}:00`;
  }

  function tooltipFormatter(params) {
    const rows = params.filter(
      (p) => !p.seriesName.startsWith("_band") && p.value[1] != null,
    );
    if (rows.length === 0) return "";
    const lines = rows.map(
      (p) =>
        `<div class="tt-row"><span class="tt-key" style="border-color:${escapeHtml(p.color)}"></span>` +
        `<strong>${Number(p.value[1]).toFixed(1)}</strong><span>${escapeHtml(p.seriesName)}</span></div>`,
    );
    const suffix = state.view === "error" ? "，過去 24 小時平均誤差" : "";
    return `<div class="tt-time">${escapeHtml(formatTime(params[0].value[0]))}（台灣時間${suffix}）</div>${lines.join("")}`;
  }

  function chartOption(data) {
    const startMs = startMsOf(data);
    const lead = data.leads[String(state.lead)];
    const series =
      state.view === "error"
        ? errorSeries(lead, startMs)
        : valueSeries(lead, startMs);
    const text = cssVar("--text-secondary");
    const grid = cssVar("--grid");
    return {
      useUTC: true,
      animation: false,
      backgroundColor: "transparent",
      textStyle: { fontFamily: getComputedStyle(document.body).fontFamily },
      grid: { left: 48, right: 16, top: 36, bottom: 72 },
      tooltip: {
        trigger: "axis",
        formatter: tooltipFormatter,
        confine: true,
        backgroundColor: cssVar("--surface-1"),
        borderColor: cssVar("--border"),
        textStyle: { color: cssVar("--text-primary"), fontSize: 13 },
        axisPointer: {
          type: "line",
          lineStyle: { color: cssVar("--text-muted"), width: 1 },
        },
      },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: grid } },
        axisTick: { show: false },
        axisLabel: {
          color: text,
          hideOverlap: true,
          formatter: {
            year: "{yyyy}",
            month: "{M} 月",
            day: "{M}/{d}",
            hour: "{HH}:00",
          },
        },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        name: state.view === "error" ? "誤差 μg/m³" : "μg/m³",
        nameTextStyle: { color: text, align: "left" },
        min: 0,
        axisLabel: { color: text },
        splitLine: { lineStyle: { color: grid } },
      },
      dataZoom: [
        { type: "inside", zoomOnMouseWheel: "ctrl", moveOnMouseWheel: false },
        {
          type: "slider",
          height: 28,
          bottom: 16,
          borderColor: grid,
          showDetail: false,
          fillerColor: "rgba(127,127,127,0.15)",
          dataBackground: {
            lineStyle: { color: text, opacity: 0.4 },
            areaStyle: { opacity: 0 },
          },
        },
      ],
      series,
    };
  }

  // ---------- tables & text ----------

  function pooledMae(modelId, year, lead) {
    let total = 0;
    let count = 0;
    state.index.stations.forEach((s) => {
      const m =
        s.metrics &&
        s.metrics[String(year)] &&
        s.metrics[String(year)][modelId];
      const v = m && m[String(lead)];
      if (v) {
        total += v.mae * v.n;
        count += v.n;
      }
    });
    return count ? total / count : null;
  }

  function cell(text, className) {
    const td = document.createElement("td");
    if (className) td.className = className;
    td.textContent = text;
    return td;
  }

  function renderTable(data) {
    const body = $("metrics-body");
    body.replaceChildren();
    const key = String(state.lead);
    const rows = state.index.models
      .map((m) => ({ m, v: data.metrics[m.id] && data.metrics[m.id][key] }))
      .filter((r) => r.v);
    const best = Math.min(...rows.map((r) => r.v.mae));
    rows.forEach(({ m, v }) => {
      const tr = document.createElement("tr");
      if (v.mae === best) tr.className = "best";
      const name = document.createElement("td");
      const wrap = document.createElement("span");
      wrap.className = "model-name";
      wrap.append(keySwatch(m.id), document.createTextNode(m.label));
      name.append(wrap);
      const pooled = pooledMae(m.id, state.year, state.lead);
      tr.append(
        name,
        cell(v.mae.toFixed(2), "num mae-site"),
        cell(v.coverage80 == null ? "—" : v.coverage80.toFixed(2), "num"),
        cell(pooled == null ? "—" : pooled.toFixed(2), "num"),
      );
      body.append(tr);
    });
    paintKeys();
    $("table-title").textContent =
      `指標：${state.station}，${state.year} 年，提前 ${state.lead} 小時`;
  }

  function relativeGap(a, b) {
    return ((a - b) / b) * 100;
  }

  function renderVerdict(data) {
    const el = $("station-verdict");
    el.replaceChildren();
    const key = String(state.lead);
    const get = (id) =>
      data.metrics[id] && data.metrics[id][key] && data.metrics[id][key].mae;
    const ft = get(FINETUNED);
    const zs = get(ZERO_SHOT);
    const xgb = get(TRAINED);
    if (ft == null || zs == null || xgb == null) return;
    const sentence = (id, value) => {
      const gap = relativeGap(value, xgb);
      const span = document.createElement("span");
      const strong = document.createElement("strong");
      strong.textContent = `${gap > 0 ? "高" : "低"} ${Math.abs(gap).toFixed(1)}%`;
      span.append(
        document.createTextNode(
          `${modelLabel(id)} 的 MAE 比 ${modelLabel(TRAINED)} `,
        ),
        strong,
      );
      return span;
    };
    el.append(
      sentence(ZERO_SHOT, zs),
      document.createElement("br"),
      sentence(FINETUNED, ft),
    );
  }

  function renderText() {
    const entry = state.index.stations.find((s) => s.name === state.station);
    const what =
      state.view === "error" ? "誤差（24 小時移動平均）" : "預測與觀測";
    $("station-title").textContent = `${entry.name}（${entry.en}）`;
    $("station-sub").textContent =
      `${entry.region_zh}空品區 · 測試年 ${state.year} · 提前 ${state.lead} 小時`;
    $("chart-title").textContent =
      `${entry.name} 提前 ${state.lead} 小時的${what}`;
    $("chart-note").textContent = NOTES[state.view];
    $("band-select").disabled = state.view === "error";
  }

  // ---------- flow ----------

  function render() {
    syncModelToggles();
    if (!state.data || !chart) return;
    chart.setOption(chartOption(state.data), { notMerge: true });
    applyZoom();
    renderTable(state.data);
    renderVerdict(state.data);
    renderText();
  }

  async function refresh() {
    $("station-select").value = state.station;
    $("year-select").value = String(state.year);
    paintMarkers();
    const wanted = `${state.station}|${state.year}`;
    setStatus("載入中…");
    if (chart)
      chart.showLoading("default", {
        text: "",
        maskColor: "rgba(127,127,127,0.08)",
      });
    try {
      const data = await loadStation(state.station, state.year);
      if (wanted !== `${state.station}|${state.year}`) return; // a newer selection won
      state.data = data;
      computeEpisodes(data);
      buildPeriodOptions(data);
      setStatus("");
      render();
    } catch (err) {
      setStatus(`無法載入資料：${err.message}`);
    } finally {
      if (chart) chart.hideLoading();
    }
  }

  function selectStation(name) {
    if (!state.index.stations.some((s) => s.name === name)) return;
    state.station = name;
    refresh();
  }

  function onThemeChange() {
    paintKeys();
    paintMarkers();
    render();
  }

  async function init() {
    if (!core) {
      setStatus("頁面元件載入失敗，請重新整理頁面。");
      return;
    }
    try {
      state.index = validateIndex(await fetchJson("index.json"));
    } catch (err) {
      setStatus(`無法載入資料：${err.message}`);
      return;
    }
    state.year = Math.max(...state.index.years);
    state.station = state.index.stations.some((s) => s.name === "前金")
      ? "前金"
      : state.index.stations[0].name;
    buildControls();
    buildMap();
    if (typeof echarts === "undefined") {
      $("chart").textContent = "圖表元件載入失敗，請重新整理頁面。";
    } else {
      chart = echarts.init($("chart"), null, { renderer: "canvas" });
      window.addEventListener("resize", () => chart.resize());
    }
    window
      .matchMedia("(prefers-color-scheme: dark)")
      .addEventListener("change", onThemeChange);
    refresh();
  }

  document.addEventListener("DOMContentLoaded", init);
})();
