/* PM2.5 forecast explorer - static page, data from data/*.json (same origin). */
"use strict";

(function () {
  const DATA_ROOT = "data/";
  const HOUR_MS = 3600 * 1000;
  const TAIPEI_OFFSET_MS = 8 * HOUR_MS; // shift instants so the UTC axis reads Taiwan time
  const THRESHOLD = 30.5;
  const DEFAULT_MONTH = 0; // January: autumn-winter episode season
  const DEFAULT_VISIBLE = ["chronos2_ft_nwp", "xgboost_nwp", "chronos2_uni"];
  const DASHED = new Set(["persistence"]);
  const FINETUNED = "chronos2_ft_nwp";
  const ZERO_SHOT = "chronos2_uni";
  const TRAINED = "xgboost_nwp";
  const OBSERVED_LABEL = "觀測值";
  const MONTHS = [
    "1 月",
    "2 月",
    "3 月",
    "4 月",
    "5 月",
    "6 月",
    "7 月",
    "8 月",
    "9 月",
    "10 月",
    "11 月",
    "12 月",
  ];

  const state = {
    index: null,
    station: null,
    year: null,
    lead: 24,
    month: DEFAULT_MONTH,
    band: FINETUNED,
    visible: new Set(DEFAULT_VISIBLE),
    data: null,
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

  // ---------- controls ----------

  function option(value, label) {
    const el = document.createElement("option");
    el.value = String(value);
    el.textContent = label;
    return el;
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

    const leadGroup = $("lead-group");
    index.leads.forEach((h) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "lead";
      input.value = String(h);
      input.checked = h === state.lead;
      input.addEventListener("change", () => {
        state.lead = h;
        render();
      });
      label.append(input, document.createTextNode(`${h} 小時`));
      leadGroup.append(label);
    });

    const monthSelect = $("month-select");
    monthSelect.append(option(-1, "全年"));
    MONTHS.forEach((m, i) => monthSelect.append(option(i, m)));
    monthSelect.value = String(state.month);
    monthSelect.addEventListener("change", () => {
      state.month = Number(monthSelect.value);
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

    const toggles = $("model-toggles");
    const observed = document.createElement("span");
    observed.className = "model-toggle";
    const observedKey = document.createElement("span");
    observedKey.className = "key observed";
    observed.append(observedKey, document.createTextNode(OBSERVED_LABEL));
    toggles.append(observed);
    index.models.forEach((m) => {
      const label = document.createElement("label");
      label.className = "model-toggle";
      const input = document.createElement("input");
      input.type = "checkbox";
      input.checked = state.visible.has(m.id);
      input.addEventListener("change", () => {
        if (input.checked) state.visible.add(m.id);
        else state.visible.delete(m.id);
        render();
      });
      const key = document.createElement("span");
      key.className = DASHED.has(m.id) ? "key dashed" : "key";
      key.dataset.model = m.id;
      label.append(input, key, document.createTextNode(m.label));
      toggles.append(label);
    });
    paintKeys();
  }

  function paintKeys() {
    document.querySelectorAll(".key[data-model]").forEach((el) => {
      el.style.setProperty("--key-color", modelColor(el.dataset.model));
    });
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

  function modelSeries(id, label, pred, startMs) {
    return {
      name: label,
      type: "line",
      symbol: "none",
      connectNulls: false,
      sampling: "lttb",
      z: 3,
      data: toPairs(pred.q50, startMs),
      lineStyle: {
        width: 2,
        color: modelColor(id),
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
    return `<div class="tt-time">${escapeHtml(formatTime(params[0].value[0]))}（台灣時間）</div>${lines.join("")}`;
  }

  function chartOption(data) {
    const startMs = Date.parse(data.start) + TAIPEI_OFFSET_MS;
    const lead = data.leads[String(state.lead)];
    const series = [];
    const bandPred = state.band && lead.models[state.band];
    if (bandPred)
      series.push(...bandSeries(bandPred, startMs, modelColor(state.band)));
    state.index.models
      .filter((m) => state.visible.has(m.id) && lead.models[m.id])
      .forEach((m) =>
        series.push(modelSeries(m.id, m.label, lead.models[m.id], startMs)),
      );
    series.push(observedSeries(lead.y, startMs));
    const text = cssVar("--text-secondary");
    const grid = cssVar("--grid");
    return {
      useUTC: true,
      animation: false,
      backgroundColor: "transparent",
      textStyle: { fontFamily: getComputedStyle(document.body).fontFamily },
      grid: { left: 48, right: 16, top: 16, bottom: 72 },
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
        axisLabel: { color: text, hideOverlap: true },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        name: "μg/m³",
        nameTextStyle: { color: text, align: "right" },
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

  function zoomRange() {
    if (state.month < 0 || !state.year) return null;
    return [
      Date.UTC(state.year, state.month, 1),
      Date.UTC(state.year, state.month + 1, 1),
    ];
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
      const swatch = document.createElement("span");
      swatch.className = DASHED.has(m.id) ? "key dashed" : "key";
      swatch.dataset.model = m.id;
      wrap.append(swatch, document.createTextNode(m.label));
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
    const label = (id) => state.index.models.find((m) => m.id === id).label;
    const sentence = (id, value) => {
      const gap = relativeGap(value, xgb);
      const span = document.createElement("span");
      const strong = document.createElement("strong");
      strong.textContent = `${gap > 0 ? "高" : "低"} ${Math.abs(gap).toFixed(1)}%`;
      span.append(
        document.createTextNode(`${label(id)} 的 MAE 比 ${label(TRAINED)} `),
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

  // ---------- flow ----------

  function render() {
    if (!state.data || !chart) return;
    const data = state.data;
    chart.setOption(chartOption(data), { notMerge: true });
    applyZoom();
    renderTable(data);
    renderVerdict(data);
    const entry = state.index.stations.find((s) => s.name === state.station);
    $("station-title").textContent = `${entry.name}（${entry.en}）`;
    $("station-sub").textContent =
      `${entry.region_zh}空品區 · 測試年 ${state.year} · 提前 ${state.lead} 小時`;
    $("chart-title").textContent =
      `${entry.name} 提前 ${state.lead} 小時的預測與觀測`;
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
