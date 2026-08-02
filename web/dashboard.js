"use strict";
/* Aegis dashboard. The local safety and financial-risk values are supplied
 * by the server:
 * every value drawn for the local-pack tracks/zones/alerts is a verbatim
 * field from the server's FrameDelta. The one addition
 * this file computes itself is the zoomed-out GLOBAL layer's screen
 * projection for live AIS positions streamed from data/global_ais.py --
 * those come through as bare lat/lon/course, same as any Leaflet marker.
 */

const TRAIL_LEN = 20;
const MAX_LOG_LINES = 400;

const els = {
  map: document.getElementById("map"),
  mapWrap: document.getElementById("map-wrap"),
  eventlog: document.getElementById("eventlog"),
  sidepanel: document.getElementById("sidepanel"),
  log: document.getElementById("log-entries"),
  idCount: document.getElementById("id-switch-count"),
  statsTracks: document.getElementById("stats-tracks"),
  statsDark: document.getElementById("stats-dark"),
  statsDarkLabel: document.getElementById("stats-dark-label"),
  statsTotal: document.getElementById("stats-total"),
  btnPlay: document.getElementById("btn-play"),
  btnReset: document.getElementById("btn-reset"),
  scrubber: document.getElementById("scrubber"),
  frameReadout: document.getElementById("frame-readout"),
  connStatus: document.getElementById("conn-status"),
  speedBtns: Array.from(document.querySelectorAll(".speed-btn")),
  packPicker: document.getElementById("pack-picker"),
  btnAssoc: document.getElementById("btn-assoc"),
  assocBadge: document.getElementById("assoc-badge"),
  btnRetract: document.getElementById("btn-retract"),
  btnGlobalLayer: document.getElementById("btn-global-layer"),
  btnBathymetry: document.getElementById("btn-bathymetry"),
  globalBadge: document.getElementById("global-badge"),
  fishingIntelligence: document.getElementById("fishing-intelligence"),
  fishingIntelligenceToggle: document.getElementById("fishing-intelligence-toggle"),
  fishingIntelligenceBody: document.getElementById("fishing-intelligence-body"),
  fishingIntelligenceNote: document.getElementById("fishing-intelligence-note"),
  banner: document.getElementById("banner"),
  briefList: document.getElementById("brief-list"),
  jtmsFacts: document.getElementById("jtms-facts"),
  jtmsConcls: document.getElementById("jtms-concls"),
  evalSummary: document.getElementById("eval-summary"),
  evalList: document.getElementById("eval-list"),
  riskSummary: document.getElementById("risk-summary"),
  riskList: document.getElementById("risk-list"),
  trajectoryPanel: document.getElementById("trajectory-panel"),
  trajectoryClose: document.getElementById("trajectory-close"),
  trajectoryName: document.getElementById("trajectory-name"),
  trajectoryMeta: document.getElementById("trajectory-meta"),
  trajectoryContext: document.getElementById("trajectory-context"),
  trajectoryGfw: document.getElementById("trajectory-gfw"),
  trajectoryActivity: document.getElementById("trajectory-activity"),
  trajectoryRisk: document.getElementById("trajectory-risk"),
  trajectoryEnvironment: document.getElementById("trajectory-environment"),
  trajectoryOptions: document.getElementById("trajectory-options"),
  trajectoryHeading: document.getElementById("trajectory-heading"),
  trajectoryNote: document.getElementById("trajectory-note"),
};

const state = {
  ws: null,
  totalFrames: 0,
  frameIntervalS: 30,
  playing: false,
  speed: 10,
  lastIdSwitches: 0,
  scrubDragging: false,
  zonesDrawn: false,
  assocMode: "global",
  packId: null,
  globalLayerOn: false,
  globalLive: false, // true once a real aisstream.io fix has arrived
  globalVessels: new Map(),
  globalRevision: 0,
  globalStatus: {},
  selectedMmsi: null,
  selectedVessel: null,
  trajectoryMode: "all",
  trajectoryAnimationFrame: null,
  trajectoryAnimationStartedAt: null,
  trajectoryRenderId: 0,
  predictionCache: new Map(),
  predictionPolls: new Set(),
  gfwCache: new Map(),
  gfwActivityCache: new Map(),
  gfwLayerMetadataLoaded: false,
  localView: null,
  preSelectionView: null,
  lastFrameRisk: null,
  bathymetryOn: true,
};
const SCENARIO_NAMES = {
  s01_dark_in_sanctuary: "Dark-vessel response",
  s02_mmsi_spoof: "Identity-conflict detection",
  s02_synthetic_demo: "Vessel tracking baseline",
  s03_ghost_fleet: "Fleet identity monitoring",
};

const tracks = new Map(); // track_id -> { marker, trailGroup, ellipseLayer, positions, color }
const globalMarkers = new Map(); // mmsi -> marker

// ------------------------------------------------------------------- map

const map = L.map(els.map, {
  zoomControl: false,
  attributionControl: true,
  worldCopyJump: true,
}).setView([20, 0], 3);

const AegisZoomControl = L.Control.extend({
  options: { position: "topright" },
  onAdd(targetMap) {
    const control = L.DomUtil.create("div", "aegis-zoom-control");
    const zoomIn = L.DomUtil.create("button", "aegis-zoom-button", control);
    const zoomOut = L.DomUtil.create("button", "aegis-zoom-button", control);
    zoomIn.type = "button";
    zoomOut.type = "button";
    zoomIn.title = "Zoom in";
    zoomOut.title = "Zoom out";
    zoomIn.setAttribute("aria-label", "Zoom in");
    zoomOut.setAttribute("aria-label", "Zoom out");
    zoomIn.innerHTML = '<span aria-hidden="true"></span>';
    zoomOut.innerHTML = '<span aria-hidden="true"></span>';
    L.DomEvent.disableClickPropagation(control);
    L.DomEvent.on(zoomIn, "click", () => targetMap.zoomIn());
    L.DomEvent.on(zoomOut, "click", () => targetMap.zoomOut());
    return control;
  },
});
new AegisZoomControl().addTo(map);
map.attributionControl.setPrefix(false);

const BoatCanvasRenderer = L.Canvas.extend({
  _updateCircle(layer) {
    if (!layer.options.boatShape) {
      return L.Canvas.prototype._updateCircle.call(this, layer);
    }
    if (!this._drawing || layer._empty()) return;
    const point = layer._point;
    const zoom = this._map.getZoom();
    const densitySize = zoom <= 3
      ? 0.9
      : zoom <= 4
        ? 1.25
        : zoom <= 6
          ? 1.8
          : zoom <= 9
            ? 3
            : 5.5;
    const size = layer.options.boatSelected ? 10 : densitySize;
    const bearing = (Number(layer.options.boatBearing) || 0) * Math.PI / 180;
    const ctx = this._ctx;
    ctx.save();
    ctx.translate(point.x, point.y);
    ctx.rotate(bearing);
    ctx.beginPath();
    ctx.moveTo(0, -size * 1.45);
    ctx.lineTo(size * 0.72, size * 0.35);
    ctx.lineTo(size * 0.6, size * 1.25);
    ctx.lineTo(0, size * 0.9);
    ctx.lineTo(-size * 0.6, size * 1.25);
    ctx.lineTo(-size * 0.72, size * 0.35);
    ctx.closePath();
    ctx.restore();
    this._fillStroke(ctx, layer);
  },
});

L.tileLayer(
  // CARTO retired the "dark_matter" path (404s now); "dark_all" is the
  // live equivalent -- confirmed by curl against basemaps.cartocdn.com.
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {
    subdomains: "abcd",
    maxZoom: 19,
    updateWhenZooming: true,
    updateWhenIdle: false,
    keepBuffer: 3,
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> · ' +
      '&copy; <a href="https://carto.com/attributions">CARTO</a>',
  }
).addTo(map);

let cameraInFlight = false;
let cameraSequence = 0;
function stopMapFlight() {
  cameraSequence += 1;
  cameraInFlight = false;
  map.stop();
  restoreVesselPane();
}
function flyMap(action, durationSeconds, settleAction = null) {
  map.stop();
  const sequence = ++cameraSequence;
  cameraInFlight = true;
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      if (sequence !== cameraSequence) {
        resolve();
        return;
      }
      cameraInFlight = false;
      if (settleAction) settleAction();
      map.invalidateSize({ pan: false });
      globalRenderer._update();
      restoreVesselPane();
      resolve();
    };
    action();
    // Leaflet can emit a stale moveend from a just-cancelled flyTo. Waiting for
    // the declared duration prevents paths and runners from rendering midway
    // through the replacement camera animation.
    window.setTimeout(finish, durationSeconds * 1000 + 120);
  });
}

const bathymetryLayer = L.tileLayer.wms("https://wms.gebco.net/mapserv?", {
  layers: "GEBCO_LATEST",
  format: "image/png",
  transparent: true,
  opacity: 0.3,
  attribution: "GEBCO 2026",
}).addTo(map);
els.btnBathymetry.addEventListener("click", () => {
  state.bathymetryOn = !state.bathymetryOn;
  els.btnBathymetry.classList.toggle("active", state.bathymetryOn);
  els.btnBathymetry.setAttribute(
    "aria-pressed",
    String(state.bathymetryOn)
  );
  if (state.bathymetryOn) bathymetryLayer.addTo(map);
  else map.removeLayer(bathymetryLayer);
});

const fishingIntelligenceLayers = {
  fishing: L.tileLayer("/api/gfw/tiles/fishing/{z}/{x}/{y}.png", {
    opacity: 0.62,
    maxZoom: 12,
    keepBuffer: 2,
    className: "gfw-heatmap-layer",
    attribution: "Global Fishing Watch",
  }),
  sar: L.tileLayer("/api/gfw/tiles/sar/{z}/{x}/{y}.png", {
    opacity: 0.72,
    maxZoom: 12,
    keepBuffer: 2,
    className: "gfw-sar-layer",
    attribution: "Global Fishing Watch · Copernicus Sentinel-1",
  }),
};

async function loadFishingLayerMetadata() {
  if (state.gfwLayerMetadataLoaded) return;
  try {
    const data = await getJson("/api/gfw/layers");
    state.gfwLayerMetadataLoaded = true;
    const available = new Map((data.layers || []).map((layer) => [layer.id, layer]));
    for (const button of document.querySelectorAll(".intel-layer")) {
      const layer = available.get(button.dataset.intelLayer);
      button.disabled = !layer?.available;
      const detail = button.querySelector("small");
      if (layer?.available && detail) {
        detail.textContent =
          `${layer.unit} · ${layer.from} to ${layer.to}`;
      }
    }
    els.fishingIntelligenceNote.textContent = data.note ||
      "Activity does not by itself establish illegality.";
  } catch (_err) {
    els.fishingIntelligenceNote.textContent =
      "Global Fishing Watch layers are temporarily unavailable.";
  }
}

els.fishingIntelligenceToggle.addEventListener("click", () => {
  const opening = els.fishingIntelligenceBody.classList.contains("hidden");
  els.fishingIntelligenceBody.classList.toggle("hidden", !opening);
  els.fishingIntelligenceToggle.setAttribute("aria-expanded", String(opening));
  if (opening) loadFishingLayerMetadata();
});
for (const button of document.querySelectorAll(".intel-layer")) {
  button.addEventListener("click", () => {
    const kind = button.dataset.intelLayer;
    const layer = fishingIntelligenceLayers[kind];
    const enabling = !map.hasLayer(layer);
    if (enabling) layer.addTo(map);
    else map.removeLayer(layer);
    button.classList.toggle("active", enabling);
    button.setAttribute("aria-pressed", String(enabling));
  });
}

const zonesLayer = L.featureGroup().addTo(map); // needs getBounds(); plain layerGroup lacks it
const localLayer = L.layerGroup().addTo(map);
const globalLayer = L.layerGroup();
const globalProjectionLayer = L.layerGroup().addTo(globalLayer);
const selectedVesselLayer = L.layerGroup().addTo(globalLayer);
const fishingActivityLayer = L.layerGroup().addTo(globalLayer);
map.createPane("vesselPane");
map.getPane("vesselPane").style.zIndex = 390;
map.getPane("vesselPane").style.opacity = "1";
map.getPane("vesselPane").style.willChange = "transform";
const globalRenderer = new BoatCanvasRenderer({
  padding: 0.5,
  tolerance: 8,
  pane: "vesselPane",
});
function restoreVesselPane() {
  map.getPane("vesselPane").style.opacity = "1";
}
map.on("zoomend", () => {
  globalRenderer._update();
  restoreVesselPane();
});

function drawZones(zones) {
  zonesLayer.clearLayers();
  state.zonesDrawn = false;
  for (const z of zones || []) {
    L.geoJSON(z.geojson, {
      style: {
        color: z.style.color,
        weight: 2,
        opacity: 0.9,
        fillOpacity: z.style.fillOpacity,
        dashArray: z.style.dashArray || null,
      },
    })
      .bindTooltip(`${z.name} (${z.kind})`)
      .addTo(zonesLayer);
  }
  if (zones && zones.length && !state.globalLayerOn) {
    state.zonesDrawn = true;
    map.fitBounds(zonesLayer.getBounds().pad(6), { animate: false });
  }
}

function triangleIcon(color, headingDeg) {
  return L.divIcon({
    className: "track-icon",
    html: `<div class="track-triangle" style="border-bottom-color:${color};transform:rotate(${headingDeg}deg);"></div>`,
    iconSize: [16, 18],
    iconAnchor: [8, 10],
  });
}

function ensureTrack(tp) {
  let t = tracks.get(tp.track_id);
  if (t) return t;
  const marker = L.marker([tp.lat, tp.lon], {
    icon: triangleIcon(tp.color, tp.heading_deg),
    riseOnHover: true,
  }).addTo(localLayer);
  const trailGroup = L.layerGroup().addTo(localLayer);
  const ellipseLayer = L.layerGroup().addTo(localLayer);
  t = { marker, trailGroup, ellipseLayer, positions: [], color: tp.color, heading: tp.heading_deg };
  tracks.set(tp.track_id, t);
  return t;
}

function redrawTrail(t) {
  t.trailGroup.clearLayers();
  const pos = t.positions;
  const segs = pos.length - 1;
  for (let i = 0; i < segs; i++) {
    const age = (i + 1) / segs;
    L.polyline([pos[i], pos[i + 1]], {
      color: t.color, weight: 2, opacity: 0.12 + 0.6 * age, interactive: false,
    }).addTo(t.trailGroup);
  }
}

function redrawEllipse(t, tp) {
  t.ellipseLayer.clearLayers();
  if (!tp.dark || !tp.ellipse_latlon || !tp.ellipse_latlon.length) return;
  L.polygon(tp.ellipse_latlon, {
    color: "#ffb020", weight: 1.5, fillColor: "#ffb020", fillOpacity: 0.08,
    dashArray: "3 3", interactive: false,
  }).addTo(t.ellipseLayer);
}

function updateTrackVisual(t, tp) {
  t.marker.setLatLng([tp.lat, tp.lon]);
  const el = t.marker.getElement();
  if (el) {
    const tri = el.querySelector(".track-triangle");
    if (tri) {
      tri.style.borderBottomColor = tp.color;
      tri.style.transform = `rotate(${tp.heading_deg}deg)`;
    }
  }
  t.marker.setTooltipContent(`${tp.label} · ${tp.status}`);
  t.color = tp.color;
}

function removeTrack(id) {
  const t = tracks.get(id);
  if (!t) return;
  localLayer.removeLayer(t.marker);
  localLayer.removeLayer(t.trailGroup);
  localLayer.removeLayer(t.ellipseLayer);
  tracks.delete(id);
}

function updateTracks(payloads) {
  const seen = new Set();
  for (const tp of payloads || []) {
    seen.add(tp.track_id);
    const t = ensureTrack(tp);
    if (!t.marker.getTooltip()) t.marker.bindTooltip("", { sticky: true });
    t.positions.push([tp.lat, tp.lon]);
    if (t.positions.length > TRAIL_LEN) t.positions.shift();
    redrawTrail(t);
    redrawEllipse(t, tp);
    updateTrackVisual(t, tp);
  }
  for (const id of Array.from(tracks.keys())) {
    if (!seen.has(id)) removeTrack(id);
  }
}

// -------------------------------------------------------------- event log

function escapeHtml(s) {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

const LOG_TAG_RE = /^\[(\d+):(\d{2}):(\d{2})\]\s*(EVENT|TRACK|ZONE)\s+(.*)$/;

function parseLogLine(line) {
  const m = LOG_TAG_RE.exec(line);
  if (!m) return { t: 0, cls: "log-log", html: escapeHtml(line) };
  const t = Number(m[1]) * 3600 + Number(m[2]) * 60 + Number(m[3]);
  const tag = m[4];
  return { t, cls: `log-${tag.toLowerCase()}`, html: `<span class="tag">[${tag}]</span>${escapeHtml(m[5])}` };
}

function alertEntry(a) {
  const who = a.track_id ? `${a.track_id}: ` : "";
  return { t: a.t, cls: `log-alert-${a.severity}`, html: `<span class="tag">[Alert]</span>${escapeHtml(who + a.headline)}` };
}

function fusionEntry(msg) {
  const tentative = (msg.tracks || []).filter((t) => t.status === "tentative").length;
  const s = msg.stats || {};
  return {
    t: msg.t, cls: "log-fusion",
    html: `<span class="tag">[Fusion]</span>f${msg.frame_idx} · ${s.n_meas ?? 0} meas → ${s.n_tracks ?? 0} tracks · ${tentative} hyp open`,
  };
}

function renderLogEntries(entries, { replace }) {
  entries.sort((a, b) => a.t - b.t);
  const frag = document.createDocumentFragment();
  for (let i = entries.length - 1; i >= 0; i--) {
    const e = entries[i];
    const div = document.createElement("div");
    div.className = `log-line ${e.cls}`;
    div.innerHTML = e.html;
    frag.appendChild(div);
  }
  if (replace) {
    els.log.replaceChildren(frag);
  } else {
    els.log.insertBefore(frag, els.log.firstChild);
    while (els.log.children.length > MAX_LOG_LINES) els.log.removeChild(els.log.lastChild);
  }
}

// ----------------------------------------------------------------- stats

function globalContactCounts() {
  const status = state.globalStatus;
  if (Number.isFinite(Number(status.total_contacts))) {
    return {
      active: Number(status.active_contacts) || 0,
      dark: Number(status.dark_contacts) || 0,
      total: Number(status.total_contacts) || 0,
    };
  }
  let dark = 0;
  for (const vessel of state.globalVessels.values()) {
    if (vessel.dark) dark += 1;
  }
  return { active: state.globalVessels.size - dark, dark, total: state.globalVessels.size };
}

function updateStats(msg) {
  if (state.globalLayerOn) {
    const counts = globalContactCounts();
    els.statsTracks.textContent = counts.active;
    els.statsDark.textContent = counts.dark;
    els.statsTotal.textContent = counts.total;
    els.idCount.textContent = state.globalStatus.identity_switches ?? 0;
    return;
  }
  const s = msg.stats || {};
  const total = Number(s.n_tracks ?? 0);
  const dark = Number(s.n_dark ?? 0);
  els.statsTracks.textContent = Math.max(0, total - dark);
  els.statsDark.textContent = dark;
  els.statsTotal.textContent = total;
  if (typeof msg.id_switches === "number") {
    els.idCount.textContent = msg.id_switches;
    if (msg.id_switches > state.lastIdSwitches) {
      els.idCount.classList.remove("flash");
      void els.idCount.offsetWidth;
      els.idCount.classList.add("flash");
    }
    state.lastIdSwitches = msg.id_switches;
  }
}

function formatUsd(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(Number(value || 0));
}

function aisSilenceThreshold() {
  return Math.max(1, Math.round(Number(state.globalStatus.dark_after_s) || 45));
}

function formatSilenceAge(value) {
  const seconds = Math.max(0, Number(value) || 0);
  if (seconds < 120) return `${Math.round(seconds)} seconds`;
  const totalMinutes = Math.floor(seconds / 60);
  if (totalMinutes >= 60) {
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h${minutes ? ` ${minutes}m` : ""}`;
  }
  const minutes = totalMinutes;
  return `${minutes} minute${minutes === 1 ? "" : "s"}`;
}

function updateFinancialRisk(msg) {
  const risk = msg.financial_risk || { low_usd: 0, high_usd: 0, items: [] };
  state.lastFrameRisk = risk;
  if (state.selectedMmsi !== null) return;
  renderRiskPanel(risk, "Current scenario step");
}

function renderRiskPanel(risk, scope) {
  const options = risk.items || [];
  const hasOptions = options.length > 0;
  const amount = hasOptions
    ? `${options.length} costed response option${options.length === 1 ? "" : "s"}`
    : "Monitoring only";
  const detail = hasOptions
    ? `${escapeHtml(scope)} · options are conditional and are not added together`
    : `${escapeHtml(scope)} · no response action indicated`;
  els.riskSummary.innerHTML =
    `<div class="risk-total"><span class="hint">Response planning</span>` +
    `<span class="amount">${amount}</span>` +
    `<span class="hint">${detail}</span></div>`;
  els.riskList.innerHTML = "";
  for (const item of options) {
    const div = document.createElement("div");
    div.className = `risk-card ${item.severity === "critical" ? "critical" : ""}`;
    div.innerHTML =
      `<div class="risk-range">${formatUsd(item.low_usd)}–${formatUsd(item.high_usd)}</div>` +
      `<div class="risk-label">${escapeHtml(item.label)}</div>` +
      `<div class="risk-basis">${escapeHtml(item.assumption || "")}</div>` +
      `<div class="risk-basis">${escapeHtml(item.basis || "")}</div>`;
    els.riskList.appendChild(div);
  }
  if (!hasOptions) {
    els.riskList.innerHTML =
      `<p class="hint">Monitoring live safety signals. No response action is currently triggered.</p>`;
  } else if (risk.source) {
    const source = document.createElement("p");
    source.className = "hint risk-source";
    source.textContent =
      `Source: ${risk.source.agency} · ${risk.source.rate_schedule} · effective ${risk.source.effective_date}. ` +
      "These are outside-government reimbursable rates.";
    els.riskList.appendChild(source);
  }
}

function updateScrubber(frameIdx) {
  if (!state.scrubDragging) els.scrubber.value = String(frameIdx);
  const t = frameIdx * state.frameIntervalS;
  const clock = new Date(t * 1000).toISOString().substr(11, 8);
  els.frameReadout.textContent = `Step ${frameIdx + 1} of ${state.totalFrames} · ${clock}`;
}

// ------------------------------------------------------------- transport ui

function setPlayingUi(playing) {
  state.playing = playing;
  els.btnPlay.textContent = playing ? "Pause" : "Play";
  els.btnPlay.classList.toggle("is-playing", playing);
}

function setSpeedUi(speed) {
  state.speed = speed;
  for (const b of els.speedBtns) b.classList.toggle("active", Number(b.dataset.speed) === speed);
}

function setAssocUi(mode) {
  state.assocMode = mode;
  const naive = mode === "greedy";
  els.btnAssoc.textContent = naive ? "SIMPLIFIED" : "STANDARD";
  els.btnAssoc.classList.toggle("naive", naive);
  if (state.globalLayerOn) {
    els.assocBadge.textContent = "";
    els.assocBadge.classList.remove("naive");
    return;
  }
  els.assocBadge.textContent = naive ? "SIMPLIFIED" : "STANDARD";
  els.assocBadge.classList.toggle("naive", naive);
}

async function responseJson(response) {
  const data = await response.json();
  if (!response.ok) {
    const error = new Error(data.error || "Request failed");
    error.status = response.status;
    throw error;
  }
  return data;
}
async function postJson(path, body) {
  const res = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  return responseJson(res);
}
async function getJson(path) {
  const res = await fetch(path);
  return responseJson(res);
}

els.btnPlay.addEventListener("click", async () => {
  const r = state.playing ? await postJson("/api/pause") : await postJson("/api/play", { speed: state.speed });
  setPlayingUi(r.playing);
});
els.btnReset.addEventListener("click", () => postJson("/api/reset"));
for (const b of els.speedBtns) {
  b.addEventListener("click", async () => {
    const r = await postJson("/api/speed", { speed: Number(b.dataset.speed) });
    setSpeedUi(r.speed);
  });
}
let seekDebounce = null;
els.scrubber.addEventListener("input", () => {
  state.scrubDragging = true;
  const frame = Number(els.scrubber.value);
  els.frameReadout.textContent = `Step ${frame + 1} of ${state.totalFrames} · Seeking…`;
  clearTimeout(seekDebounce);
  seekDebounce = setTimeout(() => postJson("/api/seek", { frame }), 35);
});
els.scrubber.addEventListener("change", () => {
  postJson("/api/seek", { frame: Number(els.scrubber.value) });
  state.scrubDragging = false;
});

async function switchPack(pack_id) {
  els.packPicker.disabled = true;
  els.connStatus.textContent = `Loading ${SCENARIO_NAMES[pack_id] || "scenario"}…`;
  await postJson("/api/switch", { pack_id });
  els.packPicker.disabled = false;
}
async function toggleAssoc() {
  const next = state.assocMode === "greedy" ? "global" : "greedy";
  els.btnAssoc.disabled = true;
  await postJson("/api/switch", { assoc_mode: next });
  els.btnAssoc.disabled = false;
}
els.btnAssoc.addEventListener("click", toggleAssoc);
els.packPicker.addEventListener("change", () => switchPack(els.packPicker.value));

// ------------------------------------------------------------------ tabs

for (const btn of document.querySelectorAll(".tab-btn")) {
  btn.addEventListener("click", () => {
    for (const b of document.querySelectorAll(".tab-btn")) b.classList.toggle("active", b === btn);
    for (const p of document.querySelectorAll(".tab-pane")) p.classList.toggle("active", p.id === `tab-${btn.dataset.tab}`);
  });
}

// -------------------------------------------------------------------- jtms

const signalLabels = {
  broadcast_a_mmsi: "Primary identity broadcast",
  broadcast_b_mmsi: "Secondary identity broadcast",
  physically_impossible: "Location conflict",
  spoof_detected: "Identity conflict alert",
};
const conclusionLabels = {
  identity_a_confirmed: "Primary identity assessment",
  identity_b_confirmed: "Secondary identity assessment",
  spoof_detected: "Possible identity spoofing",
};
function friendlySignalName(id) {
  return signalLabels[id] || String(id).replaceAll("_", " ");
}
function friendlyConclusionName(id) {
  return conclusionLabels[id] || String(id).replaceAll("_", " ");
}
function friendlyConclusionSummary(id, supported) {
  if (id === "spoof_detected") {
    return supported
      ? "Conflicting position and identity signals indicate possible MMSI spoofing."
      : "The available signals do not currently indicate MMSI spoofing.";
  }
  return supported
    ? "The available broadcasts support this vessel identity."
    : "The available evidence does not currently support this vessel identity.";
}

function renderJtms(data) {
  els.jtmsFacts.innerHTML = "<h3>Observed signals</h3>";
  for (const [fid, f] of Object.entries(data.facts || {})) {
    const div = document.createElement("div");
    div.className = "jtms-node";
    div.innerHTML =
      `<span class="node-id">${escapeHtml(friendlySignalName(fid))}</span>` +
      `${escapeHtml(String(f.label).replaceAll("vessel_a", "the primary vessel").replaceAll("vessel_b", "the secondary vessel"))}<br>` +
      `<span class="${f.believed ? "status-believed" : "status-disbelieved"}">` +
      `${f.believed ? "Observed" : "Not observed"}</span>`;
    els.jtmsFacts.appendChild(div);
  }
  els.jtmsConcls.innerHTML = "<h3>Assessment</h3>";
  for (const [cid, c] of Object.entries(data.conclusions || {})) {
    const div = document.createElement("div");
    div.className = "jtms-node";
    const flipped = (data.flipped || []).includes(cid);
    const supported = c.status === "IN";
    div.innerHTML =
      `<span class="node-id">${escapeHtml(friendlyConclusionName(cid))}${flipped ? " · Updated" : ""}</span>` +
      `${escapeHtml(friendlyConclusionSummary(cid, supported))}<br>` +
      `<span class="status-${c.status.toLowerCase()}">${supported ? "Supported" : "Not supported"}</span>`;
    els.jtmsConcls.appendChild(div);
  }
}

function liveAssessmentCards(v) {
  const ageSeconds = Math.max(0, Number(v.age_s) || 0);
  const ageText = formatSilenceAge(ageSeconds);
  const silenceThreshold = aisSilenceThreshold();
  const registry = state.gfwCache.get(v.mmsi);
  const context = v.context || {};
  const locationSignals = [
    context.in_port ? "inside a monitored port area" : "",
    context.in_sanctuary ? "inside protected waters" : "",
    context.on_land ? "position conflicts with coastline data" : "",
    (context.near_cables || []).length ? "near subsea infrastructure" : "",
  ].filter(Boolean);
  return [
    {
      title: "Position signal",
      detail: v.dark
        ? `No AIS position received for ${ageText}; Aegis marks a contact AIS-silent after ${silenceThreshold} seconds.`
        : "AIS position reports are arriving normally.",
      status: v.dark ? `Silent >${silenceThreshold}s` : "Reporting",
      alert: v.dark,
    },
    {
      title: "Last reported movement",
      detail:
        `${Number(v.speed_kn || 0).toFixed(1)} knots toward ` +
        `${Number(v.course || 0).toFixed(0)}°.`,
      status: "Observed",
      alert: false,
    },
    {
      title: "Location review",
      detail: locationSignals.length
        ? locationSignals.join("; ") + "."
        : "No monitored location conflicts at the last reported position.",
      status: locationSignals.length ? "Review" : "Clear",
      alert: locationSignals.length > 0,
    },
    {
      title: "Identity review",
      detail: registry?.matched
        ? `Independent registry match: ${registry.name || v.name || `MMSI ${v.mmsi}`}.`
        : "No independent registry confirmation is available.",
      status: registry?.matched ? "Matched" : "AIS only",
      alert: false,
    },
  ];
}

function renderLiveEvidence(v) {
  els.jtmsFacts.innerHTML = "<h3>Observed signals</h3>";
  for (const card of liveAssessmentCards(v)) {
    const div = document.createElement("div");
    div.className = "jtms-node";
    div.innerHTML =
      `<span class="node-id">${escapeHtml(card.title)}</span>` +
      `${escapeHtml(card.detail)}<br>` +
      `<span class="${card.alert ? "status-out" : "status-in"}">${escapeHtml(card.status)}</span>`;
    els.jtmsFacts.appendChild(div);
  }
  const riskItems = v.risk?.items || [];
  els.jtmsConcls.innerHTML =
    `<h3>Assessment</h3><div class="jtms-node">` +
    `<span class="node-id">${v.dark ? "Position uncertain" : "Position current"}</span>` +
    `${v.dark
      ? "The vessel may have moved beyond its last reported position. Predicted routes show where it is most likely to be."
      : "The vessel is reporting normally; no predicted route is needed."}` +
    `<br><span class="${riskItems.length ? "status-out" : "status-in"}">` +
    `${riskItems.length ? "Review recommended" : "Routine monitoring"}</span></div>`;
}

function renderLiveBrief(v) {
  const silenceAge = formatSilenceAge(v.age_s);
  const silenceThreshold = aisSilenceThreshold();
  const riskItems = v.risk?.items || [];
  const summary = v.dark
    ? `${v.name || `MMSI ${v.mmsi}`} has sent no AIS position for ${silenceAge}, ` +
      `exceeding Aegis's ${silenceThreshold}-second silence threshold. Its last reported speed was ` +
      `${Number(v.speed_kn || 0).toFixed(1)} knots. Review the predicted routes ` +
      `and uncertainty ranges before deciding whether to investigate.`
    : `${v.name || `MMSI ${v.mmsi}`} is reporting its position normally at ` +
      `${Number(v.speed_kn || 0).toFixed(1)} knots. Continue routine monitoring.`;
  els.briefList.innerHTML =
    `<div class="brief-card"><span class="concl-id">Vessel summary</span>` +
    `<span class="${v.dark ? "status-out" : "status-in"}">` +
    `${v.dark ? "Attention" : "Normal"}</span> — ${escapeHtml(summary)}` +
    `<div class="provenance">Based on live AIS movement, location context, and ` +
    `${riskItems.length ? `${riskItems.length} active safety signal${riskItems.length === 1 ? "" : "s"}` : "no active safety alerts"}.</div></div>`;
}

function renderLivePanelEmptyState() {
  els.briefList.innerHTML =
    `<div class="brief-card"><span class="concl-id">No vessel selected</span>` +
    `Select a vessel on the map to view a plain-language operational summary.</div>`;
  els.jtmsFacts.innerHTML =
    `<h3>Observed signals</h3><p class="hint">Select a vessel to review its position, movement, location, and identity signals.</p>`;
  els.jtmsConcls.innerHTML = "";
}

async function jtmsReset() { renderJtms(await postJson("/api/jtms/reset")); }
async function jtmsRetract(factId) { renderJtms(await postJson("/api/jtms/retract", { fact_id: factId })); await loadBrief(); }
async function jtmsReinstate(factId) { renderJtms(await postJson("/api/jtms/reinstate", { fact_id: factId })); await loadBrief(); }

els.btnRetract.addEventListener("click", () => jtmsRetract("broadcast_b_mmsi"));

// ------------------------------------------------------------------- brief

async function loadBrief() {
  const data = await getJson("/api/brief?concl_ids=identity_a_confirmed,identity_b_confirmed,spoof_detected&use_llm=1");
  els.briefList.innerHTML = "";
  for (const b of data.briefs || []) {
    const div = document.createElement("div");
    div.className = "brief-card";
    const supported = b.status === "IN";
    div.innerHTML =
      `<span class="concl-id">${escapeHtml(friendlyConclusionName(b.concl_id))}</span>` +
      `<span class="status-${b.status.toLowerCase()}">${supported ? "Supported" : "Not supported"}</span> — ` +
      `${escapeHtml(friendlyConclusionSummary(b.concl_id, supported))}` +
      `<div class="provenance">Evidence: ${b.source_ids.map(friendlySignalName).join(", ")}.</div>`;
    els.briefList.appendChild(div);
  }
}

// -------------------------------------------------------------------- eval

function renderLiveChecks() {
  const status = state.globalStatus || {};
  const counts = globalContactCounts();
  const threshold = aisSilenceThreshold();
  const latestAge = status.last_message_at
    ? Math.max(0, Date.now() / 1000 - Number(status.last_message_at))
    : null;
  const inconsistentSilence = Array.from(state.globalVessels.values()).filter((v) => {
    const shouldBeDark = Number(v.age_s || 0) >= threshold;
    return Boolean(v.dark) !== shouldBeDark;
  }).length;
  const checks = [
    {
      title: "Live AIS connection",
      passed: Boolean(status.connected && state.globalLive),
      detail: status.connected
        ? `AISStream connected; latest message received ${latestAge === null ? "just now" : `${formatSilenceAge(latestAge)} ago`}.`
        : "The live AIS connection is not currently receiving data.",
    },
    {
      title: "Position ingestion",
      passed: Number(status.position_reports || 0) > 0,
      detail:
        `${Number(status.position_reports || 0).toLocaleString()} real position reports and ` +
        `${Number(status.static_reports || 0).toLocaleString()} vessel-detail reports received this run.`,
    },
    {
      title: "Contact accounting",
      passed: counts.active + counts.dark === counts.total,
      detail:
        `${counts.active.toLocaleString()} transmitting + ${counts.dark.toLocaleString()} AIS-silent = ` +
        `${counts.total.toLocaleString()} displayed contacts.`,
    },
    {
      title: "AIS-silence classification",
      passed: inconsistentSilence === 0,
      detail: inconsistentSilence
        ? `${inconsistentSilence.toLocaleString()} contacts disagree with the ${threshold}-second silence threshold.`
        : `All displayed contacts agree with the ${threshold}-second threshold and their real last-report timestamps.`,
    },
  ];

  if (state.selectedVessel) {
    const vessel = state.selectedVessel;
    const expectedDark = Number(vessel.age_s || 0) >= threshold;
    const coordinatesValid =
      Number.isFinite(Number(vessel.lat))
      && Number.isFinite(Number(vessel.lon))
      && Math.abs(Number(vessel.lat)) <= 90
      && Math.abs(Number(vessel.lon)) <= 180;
    checks.push(
      {
        title: "Selected vessel timer",
        passed: Boolean(vessel.dark) === expectedDark,
        detail:
          `${vessel.name || `MMSI ${vessel.mmsi}`} last reported ${formatSilenceAge(vessel.age_s)} ago and is ` +
          `${vessel.dark ? "classified AIS-silent" : "classified as transmitting"}.`,
      },
      {
        title: "Selected vessel position",
        passed: coordinatesValid,
        detail: coordinatesValid
          ? `Last reported position is ${Number(vessel.lat).toFixed(4)}, ${Number(vessel.lon).toFixed(4)}.`
          : "The selected vessel does not have valid geographic coordinates.",
      },
    );
  }

  const passed = checks.filter((check) => check.passed).length;
  const review = checks.length - passed;
  els.evalSummary.innerHTML =
    `<div class="eval-summary-row">` +
    `<div class="chip pass">${passed} live checks passed</div>` +
    `${review ? `<div class="chip fail">${review} need review</div>` : ""}` +
    `</div>`;
  els.evalList.innerHTML = "";
  for (const check of checks) {
    const div = document.createElement("div");
    div.className = "eval-pack";
    div.innerHTML =
      `<span class="pack-id">${escapeHtml(check.title)}</span>` +
      `<div class="check-row"><span class="outcome-${check.passed ? "pass" : "fail"}">` +
      `${check.passed ? "Passed" : "Needs review"}</span> · ${escapeHtml(check.detail)}</div>`;
    els.evalList.appendChild(div);
  }
}

async function loadEval() {
  if (state.globalLayerOn) {
    renderLiveChecks();
    return;
  }
  els.evalSummary.textContent = "Running system checks…";
  els.evalList.innerHTML = "";
  const data = await getJson("/api/eval");
  const t = data.totals;
  els.evalSummary.innerHTML =
    `<div class="eval-summary-row">` +
    `<div class="chip pass">${t.passed} checks passed</div>` +
    `${t.failed ? `<div class="chip fail">${t.failed} need review</div>` : ""}` +
    `${t.skipped ? `<div class="chip skip">${t.skipped} scenario-specific checks not required</div>` : ""}` +
    `</div>`;
  for (const p of data.packs || []) {
    const div = document.createElement("div");
    div.className = "eval-pack";
    const applicable = p.passed + p.failed;
    const reviewText = p.failed
      ? ` · ${p.failed} need${p.failed === 1 ? "s" : ""} review`
      : " · no issues found";
    const scopeText = p.skipped
      ? ` · ${p.skipped} not required for this scenario`
      : "";
    div.innerHTML =
      `<span class="pack-id">${escapeHtml(SCENARIO_NAMES[p.pack_id] || "Tracking scenario")}</span>` +
      `<div class="check-row"><span class="outcome-${p.failed ? "fail" : "pass"}">` +
      `${p.passed} of ${applicable} safety checks passed</span>${reviewText}${scopeText}</div>`;
    els.evalList.appendChild(div);
  }
}
document.querySelector('[data-tab="eval"]').addEventListener("click", loadEval);
document.querySelector('[data-tab="brief"]').addEventListener("click", () => {
  if (state.globalLayerOn) {
    if (state.selectedVessel) renderLiveBrief(state.selectedVessel);
    else renderLivePanelEmptyState();
  } else {
    loadBrief();
  }
});
document.querySelector('[data-tab="jtms"]').addEventListener("click", () => {
  if (state.globalLayerOn) {
    if (state.selectedVessel) renderLiveEvidence(state.selectedVessel);
    else renderLivePanelEmptyState();
  }
});

// ----------------------------------------------------------- global layer

function globalMarkerStyle(v, stale = false) {
  const critical = (v.risk?.items || []).some((item) => item.severity === "critical");
  const selected = Number(v.mmsi) === state.selectedMmsi;
  const color = selected ? "#7dd3fc" : stale ? "#94a3b8" : critical ? "#ef4444" : "#ffb020";
  return {
    boatShape: true,
    boatSelected: false,
    boatBearing: Number(v.course) || 0,
    color,
    weight: selected ? 0 : v.dark ? 1.5 : 0.8,
    opacity: selected ? 0 : stale ? 0.45 : 0.95,
    fillColor: color,
    fillOpacity: selected ? 0 : v.dark ? 0.08 : 0.78,
    dashArray: selected ? null : v.dark ? "3 2" : null,
  };
}

function renderSelectedVesselOverlay(v) {
  selectedVesselLayer.clearLayers();
  if (!v) return;
  const bearing = Number(v.course) || Number(v.heading) || 0;
  L.marker([Number(v.lat), Number(v.lon)], {
    icon: L.divIcon({
      className: "selected-vessel-icon",
      html:
        `<svg viewBox="0 0 18 26" aria-hidden="true" ` +
        `style="transform:rotate(${bearing}deg)">` +
        `<path d="M9 1 L16 11 L14.5 23 L9 19 L3.5 23 L2 11 Z"></path>` +
        `</svg>`,
      iconSize: [18, 26],
      iconAnchor: [9, 13],
    }),
    interactive: false,
    keyboard: false,
    zIndexOffset: 1200,
  }).addTo(selectedVesselLayer);
}

function renderOceanCurrents(ocean) {
  if (!ocean?.available) return;
  const driftSeconds = 1800;
  for (const vector of ocean.vectors || []) {
    const lat = Number(vector.lat);
    const lon = Number(vector.lon);
    const end = [
      lat + Number(vector.north_mps) * driftSeconds / 111320,
      lon + Number(vector.east_mps) * driftSeconds /
        (111320 * Math.max(0.1, Math.cos(lat * Math.PI / 180))),
    ];
    L.polyline([[lat, lon], end], {
      color: "#38bdf8",
      weight: 1.5,
      opacity: 0.65,
      interactive: false,
      className: "ocean-current-vector",
    }).addTo(globalProjectionLayer);
    L.circleMarker(end, {
      radius: 2,
      color: "#7dd3fc",
      weight: 1,
      fillColor: "#7dd3fc",
      fillOpacity: 0.8,
      interactive: false,
    }).addTo(globalProjectionLayer);
    L.marker([lat, lon], {
      icon: L.divIcon({
        className: "current-arrow-icon",
        html:
          `<span style="transform:rotate(${Number(vector.bearing_deg) - 90}deg)">➤</span>`,
        iconSize: [14, 14],
        iconAnchor: [7, 7],
      }),
      interactive: true,
      keyboard: false,
    })
      .bindTooltip(
        `Copernicus current · ${Number(vector.speed_mps).toFixed(2)} m/s · ` +
        `${Number(vector.bearing_deg).toFixed(0)}°`
      )
      .addTo(globalProjectionLayer);
  }
}

function offsetPosition(origin, bearingDeg, distanceM) {
  const radiusM = 6371008.8;
  const angular = distanceM / radiusM;
  const bearing = bearingDeg * Math.PI / 180;
  const lat1 = Number(origin[0]) * Math.PI / 180;
  const lon1 = Number(origin[1]) * Math.PI / 180;
  const lat2 = Math.asin(
    Math.sin(lat1) * Math.cos(angular) +
    Math.cos(lat1) * Math.sin(angular) * Math.cos(bearing)
  );
  const lon2 = lon1 + Math.atan2(
    Math.sin(bearing) * Math.sin(angular) * Math.cos(lat1),
    Math.cos(angular) - Math.sin(lat1) * Math.sin(lat2)
  );
  return [lat2 * 180 / Math.PI, lon2 * 180 / Math.PI];
}

function renderEnvironmentalMetrics(prediction) {
  const current = prediction.ocean_conditions?.center;
  const wind = prediction.weather_conditions?.center;
  const wave = prediction.ocean_conditions?.wave;
  const rows = [];
  if (current) {
    rows.push(
      `<div class="environment-row">` +
      `<span class="environment-icon environment-current" aria-hidden="true"></span>` +
      `<span><strong>Ocean flow</strong><small>Copernicus current, tide and wave drift</small></span>` +
      `<b>${Number(current.speed_mps).toFixed(2)} m/s · ${Number(current.bearing_deg).toFixed(0)}°</b>` +
      `</div>`
    );
  }
  if (wind) {
    rows.push(
      `<div class="environment-row">` +
      `<span class="environment-icon environment-wind" aria-hidden="true"></span>` +
      `<span><strong>Wind</strong><small>NOAA GFS · 10 m above surface</small></span>` +
      `<b>${Number(wind.speed_mps).toFixed(1)} m/s · ${Number(wind.bearing_deg).toFixed(0)}°</b>` +
      `</div>`
    );
  }
  if (wave?.available) {
    rows.push(
      `<div class="environment-row">` +
      `<span class="environment-icon environment-wave" aria-hidden="true"></span>` +
      `<span><strong>Waves</strong><small>Copernicus · from ${Number(wave.from_direction_deg).toFixed(0)}°</small></span>` +
      `<b>${Number(wave.height_m).toFixed(1)} m · ${Number(wave.period_s).toFixed(1)} s</b>` +
      `</div>`
    );
  }
  if (!rows.length && (
    prediction.ocean_conditions?.pending ||
    prediction.weather_conditions?.pending
  )) {
    rows.push(
      `<div class="environment-row environment-loading">` +
      `<span class="environment-icon"></span>` +
      `<span><strong>Current, wind and waves</strong><small>Loading nearby observations</small></span>` +
      `<b>Pending</b></div>`
    );
  }
  els.trajectoryEnvironment.innerHTML = rows.length
    ? `<div class="trajectory-heading">Local conditions</div>${rows.join("")}`
    : "";
}

function renderEnvironmentalVisuals(prediction, start) {
  const wind = prediction.weather_conditions?.center;
  if (wind) {
    const bearing = Number(wind.bearing_deg);
    const startPoint = map.latLngToLayerPoint(start);
    const position = map.layerPointToLatLng(
      L.point(startPoint.x, startPoint.y - 48)
    );
    L.marker(position, {
      icon: L.divIcon({
        className: "environment-map-icon wind-map-icon",
        html:
          `<svg viewBox="0 0 34 24" aria-hidden="true" ` +
          `style="transform:rotate(${bearing - 90}deg)">` +
          `<path d="M3 7h20c5 0 6-6 1-6"></path>` +
          `<path d="M3 12h27"></path>` +
          `<path d="M3 17h16c5 0 6 6 1 6"></path>` +
          `</svg>`,
        iconSize: [34, 24],
        iconAnchor: [17, 12],
      }),
      interactive: true,
      keyboard: false,
    })
      .bindTooltip(
        `NOAA wind · ${Number(wind.speed_mps).toFixed(1)} m/s · ` +
        `${bearing.toFixed(0)}°`
      )
      .addTo(globalProjectionLayer);
  }

  const wave = prediction.ocean_conditions?.wave;
  if (wave?.available) {
    const travelBearing = (Number(wave.from_direction_deg) + 180) % 360;
    const position = offsetPosition(start, travelBearing + 90, 650);
    L.marker(position, {
      icon: L.divIcon({
        className: "environment-map-icon wave-map-icon",
        html:
          `<svg viewBox="0 0 34 24" aria-hidden="true" ` +
          `style="transform:rotate(${travelBearing - 90}deg)">` +
          `<path d="M2 7c4-4 8-4 12 0s8 4 12 0"></path>` +
          `<path d="M8 13c4-4 8-4 12 0s8 4 12 0"></path>` +
          `<path d="M2 19c4-4 8-4 12 0s8 4 12 0"></path>` +
          `</svg>`,
        iconSize: [34, 24],
        iconAnchor: [17, 12],
      }),
      interactive: true,
      keyboard: false,
    })
      .bindTooltip(
        `Copernicus waves · ${Number(wave.height_m).toFixed(1)} m · ` +
        `${Number(wave.period_s).toFixed(1)} s`
      )
      .addTo(globalProjectionLayer);
  }
}

function stopTrajectoryAnimation({ resetClock = true } = {}) {
  if (state.trajectoryAnimationFrame !== null) {
    cancelAnimationFrame(state.trajectoryAnimationFrame);
    state.trajectoryAnimationFrame = null;
  }
  if (resetClock) state.trajectoryAnimationStartedAt = null;
}

function trajectoryRunnerIcon(color, compact = false) {
  const width = compact ? 10 : 14;
  const height = compact ? 14 : 20;
  return L.divIcon({
    className: `trajectory-runner-icon${compact ? " compact" : ""}`,
    html:
      `<svg viewBox="0 0 14 20" aria-hidden="true" style="color:${color}">` +
      `<path d="M7 1 L12 9 L11 17 L7 14.5 L3 17 L2 9 Z"></path>` +
      `</svg>`,
    iconSize: [width, height],
    iconAnchor: [width / 2, height / 2],
  });
}

function trajectoryBearing(from, to, fallback = 0) {
  const north = Number(to[0]) - Number(from[0]);
  const east =
    (Number(to[1]) - Number(from[1])) *
    Math.cos(Number(from[0]) * Math.PI / 180);
  if (Math.hypot(east, north) < 1e-9) return fallback;
  return Math.atan2(east, north) * 180 / Math.PI;
}

function startTrajectoryAnimation(runners, { preservePhase = false } = {}) {
  stopTrajectoryAnimation({ resetClock: !preservePhase });
  let readinessFrames = 0;
  const begin = () => {
    if (
      runners.some((runner) => runner.marker.getElement() === null)
      && readinessFrames < 4
    ) {
      readinessFrames += 1;
      state.trajectoryAnimationFrame = requestAnimationFrame(begin);
      return;
    }
    const startedAt = preservePhase && state.trajectoryAnimationStartedAt !== null
      ? state.trajectoryAnimationStartedAt
      : performance.now();
    state.trajectoryAnimationStartedAt = startedAt;
    const durationMs = 4800;
    const travelEnd = 0.88;
    const fadeStart = 0.8;
    const animate = (now) => {
      const phase = ((now - startedAt) % durationMs) / durationMs;
      for (const runner of runners) {
        const runnerPhase = (phase + (runner.phaseOffset || 0)) % 1;
        const progress = Math.min(1, runnerPhase / travelEnd);
        const eased = progress * progress * (3 - 2 * progress);
        const opacity = runnerPhase < fadeStart
          ? 1
          : Math.max(
            0,
            (travelEnd - runnerPhase) / (travelEnd - fadeStart)
          );
        const scaled = eased * (runner.path.length - 1);
        const segment = Math.min(runner.path.length - 2, Math.floor(scaled));
        const fraction = scaled - segment;
        const from = runner.path[segment];
        const to = runner.path[segment + 1];
        const rawBearing = runner.shortRoute
          ? runner.stableBearing
          : trajectoryBearing(from, to, runner.stableBearing);
        if (runner.displayBearing === undefined) {
          runner.displayBearing = rawBearing;
        } else {
          const bearingDelta =
            (rawBearing - runner.displayBearing + 540) % 360 - 180;
          runner.displayBearing += bearingDelta * 0.16;
        }
        runner.marker.setLatLng([
          from[0] + (to[0] - from[0]) * fraction,
          from[1] + (to[1] - from[1]) * fraction,
        ]);
        const element = runner.marker.getElement();
        if (element) {
          element.style.opacity = String(opacity);
          const boat = element.querySelector("svg");
          if (boat) {
            boat.style.transform =
              `rotate(${runner.displayBearing}deg)`;
          }
        }
      }
      state.trajectoryAnimationFrame = requestAnimationFrame(animate);
    };
    state.trajectoryAnimationFrame = requestAnimationFrame(animate);
  };
  state.trajectoryAnimationFrame = requestAnimationFrame(begin);
}

function hideTrajectory({ restoreView = true } = {}) {
  stopMapFlight();
  stopTrajectoryAnimation();
  setTrajectoryModesDisabled(false);
  state.trajectoryRenderId += 1;
  const returnView = state.preSelectionView;
  state.preSelectionView = null;
  if (state.selectedMmsi !== null) {
    postJson("/api/global/pin", { mmsi: null }).catch(() => {});
  }
  const selectedMarker = globalMarkers.get(state.selectedMmsi);
  state.selectedMmsi = null;
  selectedVesselLayer.clearLayers();
  if (selectedMarker) {
    selectedMarker.setStyle(globalMarkerStyle(selectedMarker._fix || {}));
    selectedMarker.closeTooltip();
  }
  state.selectedVessel = null;
  if (state.globalLayerOn) {
    renderLivePanelEmptyState();
    if (document.getElementById("tab-eval").classList.contains("active")) {
      renderLiveChecks();
    }
  }
  globalProjectionLayer.clearLayers();
  fishingActivityLayer.clearLayers();
  els.trajectoryPanel.classList.add("hidden");
  if (els.trajectoryPanel.parentElement !== els.mapWrap) {
    els.mapWrap.appendChild(els.trajectoryPanel);
  }
  map.invalidateSize({ pan: false });
  if (state.lastFrameRisk) renderRiskPanel(state.lastFrameRisk, "Current scenario step");
  if (restoreView && returnView && state.globalLayerOn) {
    flyMap(
      () => map.flyTo(returnView.center, returnView.zoom, {
        animate: true,
        duration: 1.45,
        easeLinearity: 0.18,
      }),
      1.45,
      () => map.setView(returnView.center, returnView.zoom, { animate: false })
    );
  }
}

function renderGfwIdentity(data) {
  if (!data.configured) {
    els.trajectoryGfw.innerHTML =
      `<div class="trajectory-heading">Vessel registry</div>` +
      `<div>Independent registry details are not available for this contact.</div>`;
    return;
  }
  if (!data.matched) {
    els.trajectoryGfw.innerHTML =
      `<div class="trajectory-heading">Vessel registry</div>` +
      `<div>No independent registry match was found.</div>`;
    return;
  }
  const details = [
    data.name,
    data.flag ? `flag ${data.flag}` : "",
    data.imo && data.imo !== "0" ? `IMO ${data.imo}` : "",
    (data.ship_types || []).join(", "),
    Number.isFinite(Number(data.positions_count))
      ? `${Number(data.positions_count).toLocaleString()} historical positions`
      : "",
  ].filter(Boolean);
  els.trajectoryGfw.innerHTML =
    `<div class="trajectory-heading">Vessel registry match</div>` +
    `<div>${details.map(escapeHtml).join(" · ")}</div>`;
}

async function loadGfwIdentity(mmsi) {
  if (state.gfwCache.has(mmsi)) {
    renderGfwIdentity(state.gfwCache.get(mmsi));
    return;
  }
  els.trajectoryGfw.innerHTML =
    `<div class="trajectory-heading">Vessel registry</div><div>Checking identity records…</div>`;
  try {
    const data = await getJson(`/api/global/${encodeURIComponent(mmsi)}/gfw`);
    state.gfwCache.set(mmsi, data);
    if (state.selectedMmsi === mmsi) {
      renderGfwIdentity(data);
      renderLiveBrief(state.selectedVessel);
      renderLiveEvidence(state.selectedVessel);
    }
  } catch (err) {
    if (state.selectedMmsi === mmsi) {
      renderGfwIdentity({ configured: true, matched: false, error: "request_failed" });
    }
  }
}

const activityLabels = {
  FISHING: "Apparent fishing",
  ENCOUNTER: "Vessel encounter",
  LOITERING: "Loitering",
  PORT_VISIT: "Port visit",
  GAP: "AIS gap",
};

function eventRegionText(event) {
  const regions = event.regions || {};
  const parts = [];
  const protectedNames = (event.protected_areas || [])
    .map((area) => area.name)
    .filter(Boolean);
  if (protectedNames.length) {
    parts.push(protectedNames.slice(0, 1).join(", "));
  } else if ((regions.mpaNoTake || []).length) {
    parts.push("no-take MPA reference");
  } else if ((regions.mpa || []).length) {
    parts.push("marine protected area");
  }
  if ((regions.eez || []).length) parts.push("EEZ");
  if ((regions.rfmo || []).length) {
    parts.push(`RFMO ${regions.rfmo.slice(0, 2).join(", ")}`);
  }
  return parts.join(" · ");
}

function activityMarkerIcon(type) {
  const kind = String(type || "").toLowerCase().replace("_", "-");
  return L.divIcon({
    className: "gfw-event-marker-wrap",
    html: `<span class="gfw-event-marker gfw-event-${escapeHtml(kind)}">` +
      `<span aria-hidden="true"></span></span>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

function renderGfwActivity(data) {
  fishingActivityLayer.clearLayers();
  if (!data.configured) {
    els.trajectoryActivity.innerHTML = "";
    return;
  }
  const events = data.events || [];
  if (!data.matched || !events.length) {
    els.trajectoryActivity.innerHTML =
      `<div class="trajectory-heading">Activity history</div>` +
      `<div class="activity-empty">No Global Fishing Watch events were found for this vessel in the last year.</div>`;
    return;
  }

  const counts = new Map();
  for (const event of events) {
    counts.set(event.type, (counts.get(event.type) || 0) + 1);
    const regionText = eventRegionText(event);
    const start = event.start ? new Date(event.start) : null;
    const dateText = start && !Number.isNaN(start.getTime())
      ? start.toISOString().slice(0, 10)
      : "Date unavailable";
    const marker = L.marker([event.lat, event.lon], {
      icon: activityMarkerIcon(event.type),
      zIndexOffset: 420,
    });
    marker.bindTooltip(
      `<strong>${escapeHtml(activityLabels[event.type] || event.type)}</strong><br>` +
      `${escapeHtml(dateText)}` +
      `${regionText ? `<br>${escapeHtml(regionText)}` : ""}` +
      `<br><span>Global Fishing Watch</span>`,
      { direction: "top", opacity: 0.96 }
    );
    marker.addTo(fishingActivityLayer);
  }

  const chips = Array.from(counts.entries())
    .map(([type, count]) =>
      `<span class="activity-chip activity-chip-${escapeHtml(type.toLowerCase())}">` +
      `<b>${count}</b> ${escapeHtml(activityLabels[type] || type)}</span>`
    )
    .join("");
  const rows = events.slice(0, 6).map((event) => {
    const regionText = eventRegionText(event);
    const when = event.start
      ? new Date(event.start).toISOString().slice(0, 10)
      : "Date unavailable";
    const duration = Number.isFinite(Number(event.duration_hours))
      ? `${Number(event.duration_hours).toFixed(1)}h`
      : "";
    return (
      `<button class="activity-row" type="button" data-event-id="${escapeHtml(event.id || "")}">` +
      `<span class="activity-dot activity-dot-${escapeHtml(event.type.toLowerCase())}"></span>` +
      `<span><strong>${escapeHtml(activityLabels[event.type] || event.type)}</strong>` +
      `<small>${escapeHtml([when, duration, regionText].filter(Boolean).join(" · "))}</small></span>` +
      `</button>`
    );
  }).join("");
  els.trajectoryActivity.innerHTML =
    `<div class="trajectory-heading">Activity history · past year</div>` +
    `<div class="activity-chips">${chips}</div>` +
    `<div class="activity-list">${rows}</div>` +
    `<p class="activity-caveat">${escapeHtml(data.caveat || "")}</p>`;
  for (const row of els.trajectoryActivity.querySelectorAll(".activity-row")) {
    row.addEventListener("click", () => {
      const event = events.find((item) => String(item.id || "") === row.dataset.eventId);
      if (event) map.flyTo([event.lat, event.lon], Math.max(map.getZoom(), 8), {
        duration: 0.8,
      });
    });
  }
}

async function loadGfwActivity(mmsi) {
  if (state.gfwActivityCache.has(mmsi)) {
    renderGfwActivity(state.gfwActivityCache.get(mmsi));
    return;
  }
  els.trajectoryActivity.innerHTML =
    `<div class="trajectory-heading">Activity history</div>` +
    `<div class="activity-loading"><span></span>Reviewing fishing, encounter, port, loitering and AIS-gap records…</div>`;
  try {
    const data = await getJson(
      `/api/global/${encodeURIComponent(mmsi)}/gfw/activity`
    );
    state.gfwActivityCache.set(mmsi, data);
    if (state.selectedMmsi === mmsi) renderGfwActivity(data);
  } catch (_err) {
    if (state.selectedMmsi === mmsi) {
      els.trajectoryActivity.innerHTML =
        `<div class="trajectory-heading">Activity history</div>` +
        `<div class="activity-empty">Activity records are temporarily unavailable.</div>`;
    }
  }
}

function refreshSelectedVessel(v) {
  state.selectedVessel = v;
  const age = Math.max(0, Number(v.age_s) || 0);
  const lastFix = new Date((Number(v.last_seen) || Date.now() / 1000) * 1000);
  const risk = v.risk || { low_usd: 0, high_usd: 0, items: [] };
  els.trajectoryName.textContent = v.name || `MMSI ${v.mmsi}`;
  els.trajectoryMeta.innerHTML =
    `Last fix: ${escapeHtml(lastFix.toISOString().slice(11, 19))} UTC<br>` +
    `${v.dark
      ? `No AIS report for: ${escapeHtml(formatSilenceAge(age))}<br>` +
        `AIS-silent threshold: ${aisSilenceThreshold()} seconds<br>`
      : ""}` +
    `MMSI: ${escapeHtml(v.mmsi)}${v.imo ? ` · IMO: ${escapeHtml(v.imo)}` : ""}<br>` +
    `Track: ${Number(v.course || 0).toFixed(0)}° · ${Number(v.speed_kn || 0).toFixed(1)} kn` +
    `${v.destination ? `<br>Destination: ${escapeHtml(v.destination)}` : ""}` +
    `${v.call_sign ? ` · Call sign: ${escapeHtml(v.call_sign)}` : ""}`;
  els.trajectoryRisk.innerHTML =
    `<div class="trajectory-heading">Response planning</div>` +
    `<div class="trajectory-cost">${(risk.items || []).length
      ? `${risk.items.length} conditional option${risk.items.length === 1 ? "" : "s"}`
      : "Monitoring only"}</div>`;
  renderRiskPanel(risk, v.name || `MMSI ${v.mmsi}`);
  if (state.globalLayerOn) {
    renderLiveBrief(v);
    renderLiveEvidence(v);
    if (document.getElementById("tab-eval").classList.contains("active")) {
      renderLiveChecks();
    }
  }
}

function trajectoryBounds(scenarios, start) {
  const bounds = L.latLngBounds(
    scenarios.flatMap((scenario) => scenario.path).concat([start])
  );
  for (const scenario of scenarios) {
    const end = scenario.path[scenario.path.length - 1];
    const radiusM = Math.max(0, Number(scenario.uncertainty_radius_m) || 0);
    const latRadius = radiusM / 111320;
    const lonRadius = latRadius /
      Math.max(0.1, Math.cos(Number(end[0]) * Math.PI / 180));
    bounds.extend([Number(end[0]) - latRadius, Number(end[1]) - lonRadius]);
    bounds.extend([Number(end[0]) + latRadius, Number(end[1]) + lonRadius]);
  }
  return bounds;
}

function setTrajectoryModesDisabled(disabled) {
  for (const button of document.querySelectorAll(".trajectory-mode")) {
    button.disabled = disabled;
  }
}

function predictionEnvironmentPending(prediction) {
  return Boolean(
    prediction.ocean_conditions?.pending ||
    prediction.weather_conditions?.pending
  );
}

async function pollPredictionEnvironment(v, cacheKey) {
  if (state.predictionPolls.has(cacheKey)) return;
  state.predictionPolls.add(cacheKey);
  try {
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await new Promise((resolve) => window.setTimeout(resolve, 2000));
      const latest = state.globalVessels.get(v.mmsi);
      if (
        state.selectedMmsi !== v.mmsi ||
        !latest?.dark ||
        `${latest.mmsi}:${latest.last_seen}` !== cacheKey
      ) {
        return;
      }
      let refreshed;
      try {
        refreshed = await getJson(
          `/api/global/${encodeURIComponent(v.mmsi)}/prediction`
        );
      } catch (_err) {
        continue;
      }
      if (predictionEnvironmentPending(refreshed)) continue;
      state.predictionCache.set(cacheKey, refreshed);
      await showTrajectory(latest, {
        adjustCamera: false,
        refreshEnvironment: true,
      });
      return;
    }
  } finally {
    state.predictionPolls.delete(cacheKey);
  }
}

async function showTrajectory(
  v,
  { adjustCamera = true, refreshEnvironment = false } = {}
) {
  const inPlaceRefresh =
    refreshEnvironment && state.selectedMmsi === v.mmsi;
  const renderId = ++state.trajectoryRenderId;
  if (adjustCamera) stopMapFlight();
  if (state.selectedMmsi === null) {
    const center = map.getCenter();
    state.preSelectionView = {
      center: [center.lat, center.lng],
      zoom: map.getZoom(),
    };
  }
  const previousMarker = globalMarkers.get(state.selectedMmsi);
  if (state.selectedMmsi !== v.mmsi) {
    postJson("/api/global/pin", { mmsi: v.mmsi }).catch(() => {});
  }
  state.selectedMmsi = v.mmsi;
  if (previousMarker && previousMarker !== globalMarkers.get(v.mmsi)) {
    previousMarker.setStyle(globalMarkerStyle(previousMarker._fix || {}));
  }
  const selectedMarker = globalMarkers.get(v.mmsi);
  if (selectedMarker) selectedMarker.setStyle(globalMarkerStyle(v));
  renderSelectedVesselOverlay(v);
  if (!inPlaceRefresh) {
    stopTrajectoryAnimation();
    globalProjectionLayer.clearLayers();
  }
  const start = [Number(v.lat), Number(v.lon)];
  const context = v.context || {};
  refreshSelectedVessel(v);

  const contextLines = [];
  if (context.ofac) {
    contextLines.push(
      `<strong class="context-critical">OFAC match:</strong> ${escapeHtml(context.ofac.name)} ` +
      `(${escapeHtml(context.ofac.program)}, via ${escapeHtml(context.ofac.match_basis)})`
    );
  }
  if (context.in_sanctuary) contextLines.push("Inside Monterey Bay sanctuary");
  if (context.in_port) {
    const portLabel = context.port?.name
      ? `${escapeHtml(context.port.name)} port area`
      : "Monitored port area";
    contextLines.push(
      `${portLabel} · ${Number(context.port?.distance_km || 0).toFixed(1)} km away`
    );
  }
  if (context.on_land) contextLines.push("Position intersects coastline data");
  for (const cable of context.near_cables || []) {
    contextLines.push(`${escapeHtml(cable.name)} cable · ${Number(cable.distance_km).toFixed(1)} km`);
  }
  els.trajectoryContext.innerHTML = contextLines.length
    ? `<div class="trajectory-heading">Location alerts</div>${contextLines.map((line) => `<div>${line}</div>`).join("")}`
    : `<div class="context-clear">No monitored location or identity-list alerts at this position.</div>`;
  if (!inPlaceRefresh) {
    loadGfwIdentity(v.mmsi);
    loadGfwActivity(v.mmsi);
  }
  if (els.trajectoryPanel.parentElement !== els.eventlog) {
    els.eventlog.appendChild(els.trajectoryPanel);
  }
  els.trajectoryPanel.classList.remove("hidden");
  const modeSwitch = els.trajectoryPanel.querySelector(".trajectory-mode-switch");
  modeSwitch.classList.toggle("hidden", !v.dark);
  if (!v.dark) {
    setTrajectoryModesDisabled(false);
    els.trajectoryEnvironment.innerHTML = "";
    els.trajectoryHeading.textContent = "AIS transmitting · live contact";
    els.trajectoryOptions.innerHTML =
      `<div class="trajectory-option"><span class="trajectory-swatch"></span>` +
      `<span>Current reported position</span><span class="trajectory-distance">LIVE</span></div>`;
    els.trajectoryNote.textContent =
      "No predicted trajectory is shown while this vessel is actively transmitting AIS.";
    if (adjustCamera) {
      const targetZoom = Math.max(map.getZoom(), 14);
      await flyMap(
        () => map.flyTo(start, targetZoom, {
          animate: true,
          duration: 1.05,
          easeLinearity: 0.18,
        }),
        1.05,
        () => map.setView(start, targetZoom, { animate: false })
      );
    }
    return;
  }
  if (!inPlaceRefresh) {
    els.trajectoryHeading.textContent = "Calculating likely routes…";
    setTrajectoryModesDisabled(true);
    els.trajectoryEnvironment.innerHTML =
      `<div class="trajectory-heading">Local conditions</div>` +
      `<div class="environment-row environment-loading">` +
      `<span class="environment-icon"></span>` +
      `<span><strong>Current, wind and waves</strong><small>Loading nearby observations</small></span>` +
      `<b>Pending</b></div>`;
    els.trajectoryOptions.innerHTML = "";
    els.trajectoryNote.textContent =
      "Reviewing the vessel's last reported movement and nearby conditions.";
  }
  const cacheKey = `${v.mmsi}:${v.last_seen}`;
  let prediction = state.predictionCache.get(cacheKey);
  let shouldPollEnvironment = false;
  try {
    if (!prediction) {
      prediction = await getJson(`/api/global/${encodeURIComponent(v.mmsi)}/prediction`);
      if (!predictionEnvironmentPending(prediction)) {
        state.predictionCache.set(cacheKey, prediction);
      } else {
        shouldPollEnvironment = true;
      }
    }
  } catch (err) {
    if (state.trajectoryRenderId === renderId) {
      setTrajectoryModesDisabled(false);
      if (err.status === 409) {
        els.trajectoryHeading.textContent = "Position reporting resumed";
        els.trajectoryNote.textContent =
          "This vessel is transmitting again, so a predicted route is no longer needed.";
      } else if (err.status === 404) {
        els.trajectoryHeading.textContent = "Contact left the live feed";
        els.trajectoryNote.textContent =
          "The vessel is no longer available in the current live coverage area.";
      } else {
        els.trajectoryHeading.textContent = "Route estimate unavailable";
        els.trajectoryNote.textContent =
          "There is not enough recent movement data to estimate likely routes for this contact.";
      }
    }
    return;
  }
  if (state.trajectoryRenderId !== renderId || state.selectedMmsi !== v.mmsi) return;
  if (inPlaceRefresh) {
    stopTrajectoryAnimation({ resetClock: false });
    globalProjectionLayer.clearLayers();
  }

  const colors = ["#6fc9e8", "#4ea8c7", "#8bd8ee", "#a9c7d1", "#3f819c",
    "#d5edf5"];
  const allScenarios = prediction.scenarios || [];
  const scenarios = state.trajectoryMode === "single"
    ? allScenarios.slice(0, 1)
    : allScenarios;
  els.trajectoryHeading.textContent =
    `Predicted routes · ${prediction.horizon_minutes}-minute outlook`;
  els.trajectoryOptions.innerHTML = "";
  const behaviorLabels = {
    maintain_course: "Course held",
    maneuver: "Course change",
    slow_maneuver: "Slow turn",
    course_reversal: "Turnaround",
    drift: "Unpowered drift",
  };
  scenarios.forEach((scenario, index) => {
    const color = colors[index % colors.length];
    const probability = Math.max(0, Number(scenario.probability) * 100);
    const spreadKm = Math.max(0, Number(scenario.uncertainty_radius_m) / 1000);
    const row = document.createElement("div");
    row.className = "trajectory-option";
    row.innerHTML =
      `<span class="trajectory-swatch" style="background:${color}"></span>` +
      `<span>${behaviorLabels[scenario.behavior] || `Route ${index + 1}`} · ` +
      `±${spreadKm.toFixed(1)} km spread</span>` +
      `<span class="trajectory-distance">${probability.toFixed(1)}% likely</span>`;
    els.trajectoryOptions.appendChild(row);
  });
  const bounds = trajectoryBounds(scenarios, start);
  if (adjustCamera && bounds.isValid()) {
    const cameraOptions = {
      padding: [64, 64],
      maxZoom: 12,
    };
    await flyMap(
      () => map.flyToBounds(bounds, {
        ...cameraOptions,
        animate: true,
        duration: 1.15,
        easeLinearity: 0.18,
      }),
      1.15,
      () => map.fitBounds(bounds, {
        ...cameraOptions,
        animate: false,
      })
    );
  }
  if (state.trajectoryRenderId !== renderId || state.selectedMmsi !== v.mmsi) return;
  const runners = [];
  renderOceanCurrents(prediction.ocean_conditions);
  renderEnvironmentalMetrics(prediction);
  renderEnvironmentalVisuals(prediction, start);

  if ((v.history || []).length > 1) {
    L.polyline(v.history, {
      color: "#ffb020",
      weight: 2,
      opacity: 0.65,
      interactive: false,
    }).addTo(globalProjectionLayer);
  }
  scenarios.forEach((scenario, index) => {
    const color = colors[index % colors.length];
    const path = scenario.path;
    const end = path[path.length - 1];
    const phaseOffset = index / Math.max(1, scenarios.length);
    const initialProgress = Math.min(1, phaseOffset / 0.88);
    const initialEased =
      initialProgress * initialProgress * (3 - 2 * initialProgress);
    const initialScaled = initialEased * (path.length - 1);
    const initialSegment = Math.min(
      path.length - 2,
      Math.floor(initialScaled)
    );
    const initialFraction = initialScaled - initialSegment;
    const initialFrom = path[initialSegment];
    const initialTo = path[initialSegment + 1];
    const initialPosition = [
      initialFrom[0] + (initialTo[0] - initialFrom[0]) * initialFraction,
      initialFrom[1] + (initialTo[1] - initialFrom[1]) * initialFraction,
    ];
    L.polyline(path, {
      color,
      weight: 2,
      opacity: 0.9,
      dashArray: "6 5",
      interactive: false,
      className: `trajectory-path trajectory-path-${index}`,
    }).addTo(globalProjectionLayer);
    L.circle(end, {
      radius: Number(scenario.uncertainty_radius_m) || 100,
      color,
      weight: 1,
      opacity: 0.75,
      fillColor: color,
      fillOpacity: 0.08,
      dashArray: "3 4",
      interactive: false,
      className: "trajectory-radius",
    }).addTo(globalProjectionLayer);
    const routeOriginPoint = map.latLngToLayerPoint(path[0]);
    const routeSpanPx = path.reduce(
      (largest, point) => Math.max(
        largest,
        routeOriginPoint.distanceTo(map.latLngToLayerPoint(point))
      ),
      0
    );
    const shortRoute =
      Number(scenario.distance_nm || 0) < 0.4 ||
      routeSpanPx < 10;
    const stableBearing = trajectoryBearing(
      path[0],
      end,
      Number(v.course) || Number(v.heading) || 0
    );
    const runner = L.marker(initialPosition, {
      icon: trajectoryRunnerIcon(color, shortRoute),
      interactive: false,
      keyboard: false,
      zIndexOffset: 1000,
    }).addTo(globalProjectionLayer);
    runner._trajectoryRunner = true;
    runners.push({
      marker: runner,
      path,
      phaseOffset,
      shortRoute,
      stableBearing,
      displayBearing: stableBearing,
    });
  });
  const drivers = prediction.uncertainty_drivers || [];
  const current = prediction.ocean_conditions?.center;
  const wave = prediction.ocean_conditions?.wave;
  const wind = prediction.weather_conditions?.center;
  const silenceMinutes = Math.max(1, Math.round(Number(v.age_s) / 60));
  const currentText = current
    ? `Nearby ocean flow is ${Number(current.speed_mps).toFixed(2)} m/s ` +
      `toward ${Number(current.bearing_deg).toFixed(0)}° and is included in these routes. `
    : prediction.ocean_conditions?.pending
      ? "Ocean-current data is still loading; these routes currently follow the vessel's reported movement. "
      : "No ocean-current reading was available for this location. ";
  const waveText = wave?.available
    ? `Copernicus reports ${Number(wave.height_m).toFixed(1)} m waves, including wave drift. `
    : "";
  const windText = wind
    ? `NOAA wind is ${Number(wind.speed_mps).toFixed(1)} m/s toward ` +
      `${Number(wind.bearing_deg).toFixed(0)}° and contributes to estimated leeway. `
    : prediction.weather_conditions?.pending
      ? "Wind forcing is loading. "
      : "";
  const timingText =
    `The routes begin at the last reported position and include ` +
    `${silenceMinutes} minute${silenceMinutes === 1 ? "" : "s"} without a signal. `;
  const driverLabels = {
    course_over_ground: "recent direction",
    true_heading: "vessel heading",
    speed_over_ground: "recent speed",
    rate_of_turn: "turning rate",
    track_history: "movement history",
  };
  const qualityText = drivers.length
    ? `The estimate is less certain because ${drivers.map((name) => driverLabels[name] || name).join(", ")} ` +
      `${drivers.length === 1 ? "was" : "were"} not available.`
    : "All core movement readings were available.";
  els.trajectoryNote.textContent =
    currentText +
    waveText +
    windText +
    timingText +
    qualityText;
  startTrajectoryAnimation(runners, { preservePhase: inPlaceRefresh });
  setTrajectoryModesDisabled(false);
  if (shouldPollEnvironment) {
    pollPredictionEnvironment(v, cacheKey);
  }
}

function applyGlobalFix(v) {
  let m = globalMarkers.get(v.mmsi);
  if (!m) {
    m = L.circleMarker([v.lat, v.lon], {
      renderer: globalRenderer,
      radius: 8,
      ...globalMarkerStyle(v),
      interactive: true,
      bubblingMouseEvents: false,
    }).addTo(globalLayer);
    m.on("click", () => showTrajectory(m._fix));
    m.bindTooltip("", { sticky: true });
    globalMarkers.set(v.mmsi, m);
  } else {
    m.setLatLng([v.lat, v.lon]);
    m.setStyle(globalMarkerStyle(v));
  }
  m._stale = false;
  m._fix = v;
  m._dark = !!v.dark;
  m._riskHigh = Number(v.risk?.high_usd || 0);
  const contextFlags = [
    v.context?.ofac ? "SANCTIONS LIST MATCH" : "",
    v.context?.in_sanctuary ? "PROTECTED WATERS" : "",
    (v.context?.near_cables || []).length ? "NEAR SUBSEA INFRASTRUCTURE" : "",
  ].filter(Boolean);
  m.setTooltipContent(
    `${escapeHtml(v.name || `MMSI ${v.mmsi}`)} · ${Number(v.course || 0).toFixed(0)}° · ` +
    `${Number(v.speed_kn || 0).toFixed(1)} kn` +
    `${v.dark ? ` · NO AIS REPORT FOR ${escapeHtml(formatSilenceAge(v.age_s).toUpperCase())}` : ""}` +
    `${contextFlags.length ? ` · ${contextFlags.join(" · ")}` : ""}`
  );
  if (state.selectedMmsi === v.mmsi) {
    renderSelectedVesselOverlay(v);
    const resumedAis = state.selectedVessel?.dark && !v.dark;
    if (resumedAis) showTrajectory(v);
    else refreshSelectedVessel(v);
  }
}

els.trajectoryClose.addEventListener("click", hideTrajectory);
for (const button of document.querySelectorAll(".trajectory-mode")) {
  button.addEventListener("click", () => {
    if (button.disabled) return;
    state.trajectoryMode = button.dataset.trajectoryMode;
    for (const other of document.querySelectorAll(".trajectory-mode")) {
      other.classList.toggle("active", other === button);
    }
    if (state.selectedVessel) {
      showTrajectory(state.selectedVessel, { adjustCamera: false });
    }
  });
}

function globalStatusText() {
  if (state.globalLive) {
    return `Vessel feed · ${globalContactCounts().total.toLocaleString()} contacts`;
  }
  if (!state.globalStatus.configured) {
    return "Vessel feed unavailable";
  }
  if (state.globalStatus.connected) {
    return "Connected · waiting for positions";
  }
  return "Reconnecting to vessel feed";
}

function renderLiveRail(status) {
  if (!document.getElementById("live-rail-contacts")) {
    els.log.innerHTML =
      `<div class="log-line log-track"><span class="tag">Vessels</span><span id="live-rail-contacts"></span></div>` +
      `<div class="log-line log-fusion"><span class="tag">Updates</span><span id="live-rail-messages"></span></div>` +
      `<div class="log-line log-event"><span class="tag">Identity</span><span id="live-rail-identities"></span></div>` +
      `<div class="log-line log-zone"><span class="tag">Reference</span>protected waters · ports · coastline · infrastructure</div>` +
      `<div class="log-line log-log"><span class="tag">Registry</span>independent identity review on selection</div>`;
  }
  const counts = globalContactCounts();
  const silenceThreshold = aisSilenceThreshold();
  document.getElementById("live-rail-contacts").textContent =
    `${counts.active.toLocaleString()} transmitting · ` +
    `${counts.dark.toLocaleString()} no AIS report >${silenceThreshold}s · ` +
    `${counts.total.toLocaleString()} total vessels`;
  document.getElementById("live-rail-messages").textContent =
    `${Number(status.position_reports || 0).toLocaleString()} positions · ` +
    `${Number(status.static_reports || 0).toLocaleString()} vessel details`;
  document.getElementById("live-rail-identities").textContent =
    `${Number(status.identity_switches || 0).toLocaleString()} observed changes`;
}

function setGlobalLayer(on) {
  state.globalLayerOn = on;
  document.body.classList.toggle("live-mode", on);
  els.btnGlobalLayer.classList.toggle("active", on);
  if (on) {
    state.localView = { center: map.getCenter(), zoom: map.getZoom() };
    map.removeLayer(localLayer);
    map.removeLayer(zonesLayer);
    globalLayer.addTo(map);
    const counts = globalContactCounts();
    els.statsTracks.textContent = counts.active;
    els.statsDark.textContent = counts.dark;
    els.statsTotal.textContent = counts.total;
    els.idCount.textContent = state.globalStatus.identity_switches ?? 0;
    els.assocBadge.textContent = "LIVE";
    els.assocBadge.classList.remove("naive");
    renderLiveRail(state.globalStatus);
    if (state.selectedVessel) {
      renderLiveBrief(state.selectedVessel);
      renderLiveEvidence(state.selectedVessel);
    } else {
      renderLivePanelEmptyState();
    }
    els.globalBadge.classList.remove("hidden");
    els.globalBadge.textContent = globalStatusText();
  } else {
    hideTrajectory({ restoreView: false });
    map.removeLayer(globalLayer);
    localLayer.addTo(map);
    zonesLayer.addTo(map);
    setAssocUi(state.assocMode);
    if (state.localView) {
      map.setView(state.localView.center, state.localView.zoom, { animate: false });
      state.localView = null;
    }
    els.globalBadge.classList.add("hidden");
  }
  if (document.getElementById("tab-eval").classList.contains("active")) {
    loadEval();
  }
}
els.btnGlobalLayer.addEventListener("click", () => setGlobalLayer(!state.globalLayerOn));

let globalPollTimer = null;
async function pollGlobal() {
  try {
    const data = await getJson(`/api/global?since=${state.globalRevision}`);
    state.globalLive = !!data.live;
    state.globalStatus = data.status || {};
    els.statsDarkLabel.textContent = `no AIS report >${aisSilenceThreshold()}s`;
    if (data.full) {
      for (const marker of globalMarkers.values()) globalLayer.removeLayer(marker);
      globalMarkers.clear();
      state.globalVessels.clear();
    }
    for (const v of data.vessels || []) {
      state.globalVessels.set(v.mmsi, v);
      applyGlobalFix(v);
    }
    for (const mmsi of data.removed || []) {
      const marker = globalMarkers.get(mmsi);
      if (!marker) continue;
      if (state.selectedMmsi === mmsi) {
        marker._stale = true;
        marker.setStyle(globalMarkerStyle(marker._fix || {}, true));
        marker.setTooltipContent(
          `${escapeHtml(marker._fix?.name || `MMSI ${mmsi}`)} · last report retained`
        );
        continue;
      }
      globalLayer.removeLayer(marker);
      globalMarkers.delete(mmsi);
      state.globalVessels.delete(mmsi);
    }
    state.globalRevision = Number(data.revision) || state.globalRevision;
    if (state.globalLayerOn) {
      const counts = globalContactCounts();
      els.statsTracks.textContent = counts.active;
      els.statsDark.textContent = counts.dark;
      els.statsTotal.textContent = counts.total;
      const status = data.status || {};
      els.idCount.textContent = status.identity_switches ?? 0;
      renderLiveRail(status);
      if (document.getElementById("tab-eval").classList.contains("active")) {
        renderLiveChecks();
      }
      els.globalBadge.textContent = state.globalLive
        ? `Vessel feed · ${counts.total.toLocaleString()} contacts · ` +
          `${counts.dark.toLocaleString()} silent >${aisSilenceThreshold()}s`
        : globalStatusText();
    }
  } catch (err) {
    // Global layer is best-effort; local pack streaming must never depend on it.
  }
}
globalPollTimer = setInterval(pollGlobal, 4000);
pollGlobal();

// ---------------------------------------------------------------- websocket

function applyMessage(msg) {
  if (msg.type === "init") {
    state.totalFrames = msg.total_frames;
    state.frameIntervalS = msg.frame_interval_s;
    state.packId = msg.pack_id;
    els.scrubber.max = String(Math.max(0, msg.total_frames - 1));
    if (msg.assoc_mode) setAssocUi(msg.assoc_mode);
    if (msg.known_packs) {
      els.packPicker.innerHTML = "";
      for (const p of msg.known_packs) {
        const opt = document.createElement("option");
        opt.value = p;
        opt.textContent = SCENARIO_NAMES[p] || "Tracking scenario";
        if (p === msg.pack_id) opt.selected = true;
        els.packPicker.appendChild(opt);
      }
    } else if (msg.pack_id) {
      els.packPicker.value = msg.pack_id;
    }
    drawZones(msg.zones);
    return;
  }

  if (msg.type === "state") {
    setPlayingUi(msg.playing);
    setSpeedUi(msg.speed);
    return;
  }

  if (msg.type === "frame" || msg.type === "sync") {
    updateTracks(msg.tracks);
    updateStats(msg);
    updateFinancialRisk(msg);
    updateScrubber(msg.frame_idx);
    setPlayingUi(msg.playing);
    setSpeedUi(msg.speed);
    if (msg.assoc_mode) setAssocUi(msg.assoc_mode);

    if (msg.type === "sync") {
      drawZones(msg.zones && msg.zones.length ? msg.zones : null);
      if (!state.globalLayerOn) {
        const entries = [fusionEntry(msg)].concat((msg.history_alerts || []).map(alertEntry)).concat((msg.history_log || []).map(parseLogLine));
        renderLogEntries(entries, { replace: true });
      }
    } else {
      if (!state.globalLayerOn) {
        const entries = [fusionEntry(msg)].concat((msg.alerts || []).map(alertEntry)).concat((msg.log || []).map(parseLogLine));
        renderLogEntries(entries, { replace: false });
      }
      const crit = (msg.alerts || []).find((a) => a.severity === "critical");
      if (crit) {
        els.banner.textContent = `Priority · ${crit.headline}`;
        els.banner.classList.remove("hidden");
      }
    }
  }
}

function setConn(status) {
  els.connStatus.textContent = status;
  els.connStatus.className = "conn-status " + (status === "connected" ? "ok" : status === "connecting…" ? "" : "bad");
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  state.ws = ws;
  setConn("connecting…");
  ws.onopen = () => setConn("connected");
  ws.onmessage = (ev) => {
    try { applyMessage(JSON.parse(ev.data)); } catch (err) { console.error("bad message", err, ev.data); }
  };
  ws.onclose = () => { setConn("disconnected — retrying"); setTimeout(connect, 1000); };
  ws.onerror = () => ws.close();
}

// ------------------------------------------------------------- keyboard

document.addEventListener("keydown", (ev) => {
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "SELECT") return;
  if (ev.code === "Space") { ev.preventDefault(); els.btnPlay.click(); }
  else if (ev.code === "ArrowRight") postJson("/api/seek", { frame: Number(els.scrubber.value) + 1 });
  else if (ev.code === "ArrowLeft") postJson("/api/seek", { frame: Math.max(0, Number(els.scrubber.value) - 1) });
  else if (ev.key === "m" || ev.key === "M") toggleAssoc();
  else if (ev.key === "x" || ev.key === "X") jtmsRetract("broadcast_b_mmsi");
  else if (ev.key === "r" || ev.key === "R") jtmsReinstate("broadcast_b_mmsi");
  else if (ev.key === "Escape" && state.selectedMmsi !== null) hideTrajectory();
  else if (ev.key === "Escape") els.btnReset.click();
});

setGlobalLayer(true);
connect();
jtmsReset().then(() => {
  if (state.globalLayerOn) renderLivePanelEmptyState();
  else loadBrief();
});
