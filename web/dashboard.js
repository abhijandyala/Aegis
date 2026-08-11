"use strict";
/* Aegis dashboard. The local safety and response-planning values are supplied
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
  vesselVisibilityBtns: Array.from(
    document.querySelectorAll("[data-vessel-visibility]")
  ),
  btnBathymetry: document.getElementById("btn-bathymetry"),
  fishingIntelligence: document.getElementById("fishing-intelligence"),
  fishingIntelligenceToggle: document.getElementById("fishing-intelligence-toggle"),
  fishingIntelligenceBody: document.getElementById("fishing-intelligence-body"),
  fishingIntelligenceNote: document.getElementById("fishing-intelligence-note"),
  worldSearch: document.getElementById("world-search"),
  worldSearchToggle: document.getElementById("world-search-toggle"),
  worldSearchForm: document.getElementById("world-search-form"),
  worldSearchInput: document.getElementById("world-search-input"),
  worldSearchResults: document.getElementById("world-search-results"),
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
  trajectoryNearby: document.getElementById("trajectory-nearby"),
  trajectoryGfw: document.getElementById("trajectory-gfw"),
  trajectoryActivity: document.getElementById("trajectory-activity"),
  trajectoryRisk: document.getElementById("trajectory-risk"),
  trajectoryEnvironment: document.getElementById("trajectory-environment"),
  trajectoryOptions: document.getElementById("trajectory-options"),
  trajectoryHeading: document.getElementById("trajectory-heading"),
  trajectoryNote: document.getElementById("trajectory-note"),
  trajectoryLoading: document.getElementById("trajectory-loading"),
  trajectoryLoadingVessel: document.getElementById("trajectory-loading-vessel"),
  trajectoryLoadingStages: Array.from(
    document.querySelectorAll("[data-loading-stage]")
  ),
  overviewSelection: document.getElementById("overview-selection"),
  tabMoreToggle: document.getElementById("tab-more-toggle"),
  tabMoreMenu: document.getElementById("tab-more-menu"),
  simulationViewer: document.getElementById("simulation-viewer"),
  simulationViewerOpen: document.getElementById("simulation-viewer-open"),
  simulationViewerClose: document.getElementById("simulation-viewer-close"),
  simulationViewerMap: document.getElementById("simulation-viewer-map"),
  simulationViewerCanvas: document.getElementById("simulation-viewer-canvas"),
  simulationViewerZoomIn: document.getElementById("simulation-viewer-zoom-in"),
  simulationViewerZoomOut: document.getElementById("simulation-viewer-zoom-out"),
  simulationViewerScale: document.getElementById("simulation-viewer-scale"),
  simulationViewerLoading: document.getElementById("simulation-viewer-loading"),
  simulationViewerTooltip: document.getElementById("simulation-viewer-tooltip"),
  simulationViewerTitle: document.getElementById("simulation-viewer-title"),
  simulationViewerContext: document.getElementById("simulation-viewer-context"),
  simulationViewerPhase: document.getElementById("simulation-viewer-phase"),
  simulationViewerTime: document.getElementById("simulation-viewer-time"),
  simulationViewerTimeline: document.getElementById("simulation-viewer-timeline"),
  simulationViewerPlay: document.getElementById("simulation-viewer-play"),
  simulationViewerRestart: document.getElementById("simulation-viewer-restart"),
  simulationViewerCount: document.getElementById("simulation-viewer-count"),
  simulationViewerLegend: document.getElementById("simulation-viewer-legend"),
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
  vesselVisibility: "both",
  globalLive: false, // true once the selected real AIS provider has a fix
  globalVessels: new Map(),
  globalRevision: 0,
  globalStatus: {},
  globalProviderFramed: null,
  selectedMmsi: null,
  selectedVessel: null,
  trajectoryMode: "all",
  trajectoryAnimationFrame: null,
  trajectoryAnimationStartedAt: null,
  trajectoryRenderId: 0,
  predictionCache: new Map(),
  predictionPolls: new Set(),
  selectedPredictionKey: null,
  predictionLoadingSince: 0,
  predictionLoadingMmsi: null,
  gfwCache: new Map(),
  gfwActivityCache: new Map(),
  gfwLayerMetadataLoaded: false,
  nearbyContext: null,
  contextRequestId: 0,
  contextRefreshTimer: null,
  searchRequestId: 0,
  searchTimer: null,
  localView: null,
  preSelectionView: null,
  lastFrameRisk: null,
  bathymetryOn: true,
  simulationCache: new Map(),
  simulationData: null,
  simulationFrame: null,
  simulationLastFrameAt: null,
  simulationLastDrawAt: null,
  simulationProgress: 0,
  simulationPlaying: true,
  simulationProjection: null,
  simulationBackdrop: null,
  simulationDensityCanvas: null,
  simulationMap: null,
  simulationZoomMin: 2,
  simulationZoomMax: 19,
  simulationPositions: [],
  simulationRequestId: 0,
  simulationViewerKey: null,
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
  minZoom: 2,
  maxBounds: [[-85, -180], [85, 180]],
  maxBoundsViscosity: 1,
  worldCopyJump: false,
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
    const updateDisabledState = () => {
      zoomIn.disabled = targetMap.getZoom() >= targetMap.getMaxZoom();
      zoomOut.disabled = targetMap.getZoom() <= targetMap.getMinZoom();
    };
    this._updateDisabledState = updateDisabledState;
    L.DomEvent.disableClickPropagation(control);
    L.DomEvent.on(zoomIn, "click", () => targetMap.zoomIn());
    L.DomEvent.on(zoomOut, "click", () => targetMap.zoomOut());
    targetMap.on("zoomend", updateDisabledState);
    updateDisabledState();
    return control;
  },
  onRemove(targetMap) {
    targetMap.off("zoomend", this._updateDisabledState);
  },
});
new AegisZoomControl().addTo(map);
map.attributionControl.setPrefix(false);
map.attributionControl.setPosition("bottomleft");

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
    minZoom: 2,
    maxZoom: 19,
    noWrap: true,
    bounds: [[-85, -180], [85, 180]],
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

function setWorldSearchOpen(open) {
  els.worldSearch.classList.toggle("expanded", open);
  els.worldSearchToggle.setAttribute("aria-expanded", String(open));
  if (open) {
    window.setTimeout(() => els.worldSearchInput.focus(), 180);
  } else {
    window.clearTimeout(state.searchTimer);
    els.worldSearchInput.value = "";
    els.worldSearchResults.innerHTML = "";
    els.worldSearchResults.classList.add("hidden");
  }
}

function flyToSearchResult(place) {
  if (state.selectedMmsi !== null) {
    hideTrajectory({ restoreView: false });
  }
  const bounds = place.bounds && L.latLngBounds(place.bounds);
  if (bounds?.isValid()) {
    map.flyToBounds(bounds, {
      animate: true,
      duration: 1.45,
      easeLinearity: 0.18,
      padding: [70, 70],
      maxZoom: 11,
    });
  } else {
    map.flyTo([place.lat, place.lon], 9, {
      animate: true,
      duration: 1.45,
      easeLinearity: 0.18,
    });
  }
  setWorldSearchOpen(false);
}

function renderSearchResults(places) {
  if (!places.length) {
    els.worldSearchResults.innerHTML =
      `<p class="world-search-empty">No matching place found.</p>`;
    els.worldSearchResults.classList.remove("hidden");
    return;
  }
  els.worldSearchResults.innerHTML = places.map((place, index) => {
    const parts = String(place.name || "Place").split(",");
    const primary = parts.shift()?.trim() || "Place";
    const secondary = parts.join(",").trim();
    return (
      `<button type="button" class="world-search-result" data-result="${index}">` +
      `<span class="world-search-pin" aria-hidden="true"></span>` +
      `<span><strong>${escapeHtml(primary)}</strong>` +
      `<small>${escapeHtml(secondary || place.type || "")}</small></span></button>`
    );
  }).join("");
  els.worldSearchResults.classList.remove("hidden");
  for (const button of els.worldSearchResults.querySelectorAll(".world-search-result")) {
    button.addEventListener("click", () => {
      const place = places[Number(button.dataset.result)];
      if (place) flyToSearchResult(place);
    });
  }
}

async function searchWorld() {
  const query = els.worldSearchInput.value.trim();
  const requestId = ++state.searchRequestId;
  if (query.length < 2) {
    els.worldSearchResults.classList.add("hidden");
    els.worldSearchResults.innerHTML = "";
    return;
  }
  els.worldSearchResults.innerHTML =
    `<div class="world-search-loading"><span></span>Searching places…</div>`;
  els.worldSearchResults.classList.remove("hidden");
  try {
    const data = await getJson(`/api/geocode?q=${encodeURIComponent(query)}`);
    if (requestId === state.searchRequestId) {
      renderSearchResults(data.places || []);
    }
  } catch (_err) {
    if (requestId === state.searchRequestId) {
      els.worldSearchResults.innerHTML =
        `<p class="world-search-empty">Place search is temporarily unavailable.</p>`;
    }
  }
}

L.DomEvent.disableClickPropagation(els.worldSearch);
els.worldSearchToggle.addEventListener("click", () => {
  setWorldSearchOpen(!els.worldSearch.classList.contains("expanded"));
});
els.worldSearchInput.addEventListener("input", () => {
  window.clearTimeout(state.searchTimer);
  state.searchTimer = window.setTimeout(searchWorld, 380);
});
els.worldSearchInput.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    setWorldSearchOpen(false);
  }
});
els.worldSearchForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const first = els.worldSearchResults.querySelector(".world-search-result");
  if (first) first.click();
  else searchWorld();
});

const zonesLayer = L.featureGroup().addTo(map); // needs getBounds(); plain layerGroup lacks it
const localLayer = L.layerGroup().addTo(map);
const globalLayer = L.layerGroup();
const globalProjectionLayer = L.layerGroup().addTo(globalLayer);
const selectedVesselLayer = L.layerGroup().addTo(globalLayer);
const fishingActivityLayer = L.layerGroup().addTo(globalLayer);
const nearbyContextLayer = L.layerGroup().addTo(globalLayer);
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
map.on("moveend", () => {
  if (state.globalLayerOn) refreshNearbyContext();
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

function formatLatitude(value) {
  const latitude = Number(value);
  if (!Number.isFinite(latitude)) return null;
  return `${Math.abs(latitude).toFixed(4)}° ${latitude >= 0 ? "N" : "S"}`;
}

function formatLongitude(value) {
  const longitude = Number(value);
  if (!Number.isFinite(longitude)) return null;
  return `${Math.abs(longitude).toFixed(4)}° ${longitude >= 0 ? "E" : "W"}`;
}

function formatMaritimeDistanceMeters(value) {
  const meters = Math.max(0, Number(value) || 0);
  const nauticalMiles = meters / 1852;
  if (nauticalMiles < 1) {
    return `${Math.round(meters).toLocaleString()} m`;
  }
  const digits = nauticalMiles >= 100 ? 0 : 1;
  return `${nauticalMiles.toFixed(digits)} NM`;
}

function formatMaritimeDistanceKm(value) {
  return formatMaritimeDistanceMeters((Number(value) || 0) * 1000);
}

function aisShipTypeLabel(value) {
  const code = Number(value);
  if (!Number.isFinite(code) || code <= 0) return null;
  if (code === 30) return "Fishing";
  if (code === 31 || code === 32) return "Towing";
  if (code === 33) return "Dredging";
  if (code === 34) return "Diving operations";
  if (code === 35) return "Military operations";
  if (code === 36) return "Sailing";
  if (code === 37) return "Pleasure craft";
  if (code >= 40 && code <= 49) return "High-speed craft";
  if (code === 50) return "Pilot vessel";
  if (code === 51) return "Search and rescue";
  if (code === 52) return "Tug";
  if (code === 53) return "Port tender";
  if (code === 55) return "Law enforcement";
  if (code === 58) return "Medical transport";
  if (code >= 60 && code <= 69) return "Passenger";
  if (code >= 70 && code <= 79) return "Cargo";
  if (code >= 80 && code <= 89) return "Tanker";
  if (code >= 90 && code <= 99) return "Other";
  return `AIS type ${code}`;
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
    ? `${options.length} response option${options.length === 1 ? "" : "s"}`
    : "Monitoring only";
  const detail = hasOptions
    ? `${escapeHtml(scope)} · review each option independently`
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
      `<div class="risk-priority">${item.severity === "critical" ? "Urgent review" : "Response option"}</div>` +
      `<div class="risk-label">${escapeHtml(item.label)}</div>` +
      `<div class="risk-basis">${escapeHtml(item.assumption || "")}</div>`;
    els.riskList.appendChild(div);
  }
  if (!hasOptions) {
    els.riskList.innerHTML =
      `<p class="hint">Monitoring live safety signals. No response action is currently triggered.</p>`;
  } else {
    const reference = document.createElement("details");
    reference.className = "risk-reference";
    reference.innerHTML =
      `<summary>Reference cost assumptions</summary>` +
      `<p>Optional planning references—not incurred, quoted, or predicted costs.</p>` +
      options.map((item) =>
        `<div class="risk-reference-row">` +
        `<strong>${escapeHtml(item.label)}</strong>` +
        `<span>${formatUsd(item.low_usd)}–${formatUsd(item.high_usd)}</span>` +
        `<small>${escapeHtml(item.basis || item.assumption || "")}</small>` +
        `</div>`
      ).join("") +
      (risk.source
        ? `<p>Source: ${escapeHtml(risk.source.agency)} · ` +
          `${escapeHtml(risk.source.rate_schedule)} · effective ` +
          `${escapeHtml(risk.source.effective_date)}. Outside-government reimbursable rates.</p>`
        : "");
    els.riskList.appendChild(reference);
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

function activateSidebarPanel(panelId) {
  const detailPanels = new Set(["tab-risk", "tab-brief", "tab-jtms", "tab-eval"]);
  for (const button of document.querySelectorAll(".tab-btn, .tab-menu-option")) {
    const targetId = button.dataset.panel || `tab-${button.dataset.tab}`;
    button.classList.toggle("active", targetId === panelId);
  }
  els.tabMoreToggle.classList.toggle("active", detailPanels.has(panelId));
  for (const panel of document.querySelectorAll(".tab-pane")) {
    panel.classList.toggle("active", panel.id === panelId);
  }
}

function setMoreMenuOpen(open) {
  els.tabMoreMenu.classList.toggle("hidden", !open);
  els.tabMoreToggle.setAttribute("aria-expanded", String(open));
}

for (const btn of document.querySelectorAll(".tab-btn, .tab-menu-option")) {
  btn.addEventListener("click", () => {
    activateSidebarPanel(btn.dataset.panel || `tab-${btn.dataset.tab}`);
    setMoreMenuOpen(false);
  });
}
els.tabMoreToggle.addEventListener("click", (event) => {
  event.stopPropagation();
  setMoreMenuOpen(els.tabMoreMenu.classList.contains("hidden"));
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".tab-more")) setMoreMenuOpen(false);
});

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

function vesselMatchesVisibility(v) {
  if (state.vesselVisibility === "silent") return Boolean(v?.dark);
  if (state.vesselVisibility === "live") return !v?.dark;
  return true;
}

function syncGlobalMarkerVisibility(marker, vessel) {
  const shouldShow = vesselMatchesVisibility(vessel);
  const isShown = globalLayer.hasLayer(marker);
  if (shouldShow && !isShown) globalLayer.addLayer(marker);
  else if (!shouldShow && isShown) globalLayer.removeLayer(marker);
}

function setVesselVisibility(mode) {
  if (!["silent", "live", "both"].includes(mode)) return;
  state.vesselVisibility = mode;
  try {
    localStorage.setItem("aegis-vessel-visibility", mode);
  } catch (_error) {
    // Filtering still works if browser storage is unavailable.
  }
  for (const button of els.vesselVisibilityBtns) {
    const active = button.dataset.vesselVisibility === mode;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
  }
  for (const marker of globalMarkers.values()) {
    syncGlobalMarkerVisibility(marker, marker._fix || {});
  }
  if (
    state.selectedVessel
    && !vesselMatchesVisibility(state.selectedVessel)
  ) {
    hideTrajectory();
  }
}

function renderSelectedVesselOverlay(v) {
  selectedVesselLayer.clearLayers();
  if (!v) return;
  const bearing = Number(v.course) || Number(v.heading) || 0;
  L.marker([Number(v.lat), Number(v.lon)], {
    icon: L.divIcon({
      className: `selected-vessel-icon ${v.dark ? "dark" : "active"}`,
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
      weight: 1,
      opacity: 0.42,
      interactive: false,
      className: "ocean-current-vector",
    }).addTo(globalProjectionLayer);
    L.circleMarker(end, {
      radius: 2,
      color: "#7dd3fc",
      weight: 1,
      fillColor: "#7dd3fc",
      fillOpacity: 0.5,
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
        const opacity = runnerPhase < fadeStart
          ? 1
          : Math.max(
            0,
            (travelEnd - runnerPhase) / (travelEnd - fadeStart)
          );
        const scaled = progress * (runner.path.length - 1);
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
          element.style.opacity = String(
            opacity * (runner.opacityScale ?? 1)
          );
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

const simulationBehaviorColors = {
  maintain_course: "#9bb8c3",
  maneuver: "#7899a5",
  slow_maneuver: "#c0c8cc",
  course_reversal: "#c3a66e",
  drift: "#9b92aa",
};
const simulationBehaviorLabels = {
  maintain_course: "Course held",
  maneuver: "Course change",
  slow_maneuver: "Slow turn",
  course_reversal: "Turnaround",
  drift: "Drift / stopped",
};

function stopSimulationAnimation() {
  if (state.simulationFrame !== null) {
    cancelAnimationFrame(state.simulationFrame);
    state.simulationFrame = null;
  }
  state.simulationLastFrameAt = null;
  state.simulationLastDrawAt = null;
}

function renderSimulationPlayControl() {
  const action = state.simulationPlaying ? "Pause" : "Play";
  els.simulationViewerPlay.setAttribute("aria-label", `${action} simulation`);
  els.simulationViewerPlay.title = `${action} simulation`;
  els.simulationViewerPlay.innerHTML =
    `<span class="simulation-control-icon icon-${action.toLowerCase()}" ` +
    `aria-hidden="true"></span>`;
}

function simulationProjector(bounds, width, height) {
  const south = Number(bounds?.[0]?.[0]) || 0;
  const west = Number(bounds?.[0]?.[1]) || 0;
  const north = Number(bounds?.[1]?.[0]) || south;
  const east = Number(bounds?.[1]?.[1]) || west;
  const latSpan = Math.max(0.0001, north - south);
  const longitudeScale = Math.max(
    0.1,
    Math.cos(((south + north) / 2) * Math.PI / 180)
  );
  const lonSpan = Math.max(0.0001, (east - west) * longitudeScale);
  const padding = 24;
  const scale = Math.min(
    (width - padding * 2) / lonSpan,
    (height - padding * 2) / latSpan
  );
  const drawnWidth = lonSpan * scale;
  const drawnHeight = latSpan * scale;
  const offsetX = (width - drawnWidth) / 2;
  const offsetY = (height - drawnHeight) / 2;
  return (lat, lon) => [
    offsetX + (lon - west) * longitudeScale * scale,
    offsetY + (north - lat) * scale,
  ];
}

function syncSimulationZoomControls() {
  const zoom = state.simulationMap?.getZoom();
  els.simulationViewerZoomIn.disabled =
    !Number.isFinite(zoom) || zoom >= state.simulationZoomMax;
  els.simulationViewerZoomOut.disabled =
    !Number.isFinite(zoom) || zoom <= state.simulationZoomMin;
  updateSimulationScale();
}

function updateSimulationScale() {
  const simulationMap = state.simulationMap;
  if (!simulationMap || !els.simulationViewerScale) return;
  const maximumWidth = 78;
  const size = simulationMap.getSize();
  if (!size.x || !size.y) return;
  const y = size.y / 2;
  const maximumMeters = simulationMap.distance(
    simulationMap.containerPointToLatLng([0, y]),
    simulationMap.containerPointToLatLng([maximumWidth, y])
  );
  const candidates = [
    50, 100, 200, 500,
    1852, 3704, 9260, 18520, 37040, 92600,
    185200, 370400, 926000, 1852000,
  ];
  let distanceMeters = candidates[0];
  for (const candidate of candidates) {
    if (candidate > maximumMeters) break;
    distanceMeters = candidate;
  }
  const width = Math.max(
    20,
    Math.min(maximumWidth, distanceMeters / maximumMeters * maximumWidth)
  );
  els.simulationViewerScale.style.width = `${width}px`;
  els.simulationViewerScale.textContent =
    formatMaritimeDistanceMeters(distanceMeters);
}

function refreshSimulationMapProjection() {
  if (
    !state.simulationData
    || els.simulationViewer.classList.contains("hidden")
  ) {
    return;
  }
  const rect = els.simulationViewerCanvas.getBoundingClientRect();
  buildSimulationBackdrop(
    Math.max(240, Math.round(rect.width)),
    Math.max(200, Math.round(rect.height))
  );
  drawSimulationFrame();
  syncSimulationZoomControls();
}

function changeSimulationZoom(delta, anchor = null) {
  const simulationMap = state.simulationMap;
  if (!simulationMap) return;
  const target = Math.max(
    state.simulationZoomMin,
    Math.min(state.simulationZoomMax, simulationMap.getZoom() + delta)
  );
  if (target === simulationMap.getZoom()) return;
  if (anchor) simulationMap.setZoomAround(anchor, target);
  else simulationMap.setZoom(target, { animate: false });
}

function prepareSimulationMap(bounds) {
  if (state.simulationMap === null) {
    state.simulationMap = L.map(els.simulationViewerMap, {
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
      boxZoom: false,
      keyboard: false,
      tap: false,
      zoomSnap: 0,
      zoomAnimation: false,
      fadeAnimation: false,
      markerZoomAnimation: false,
    });
    L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
      {
        subdomains: "abcd",
        minZoom: 2,
        maxZoom: 19,
        noWrap: true,
        bounds: [[-85, -180], [85, 180]],
      }
    ).addTo(state.simulationMap);
    state.simulationMap.on("zoomend", refreshSimulationMapProjection);
  }
  state.simulationMap.setMinZoom(2);
  state.simulationMap.setMaxZoom(19);
  state.simulationMap.invalidateSize({ pan: false });
  const fitBounds = L.latLngBounds(bounds);
  if (fitBounds.isValid()) {
    state.simulationMap.fitBounds(fitBounds, {
      animate: false,
      padding: [24, 24],
      maxZoom: 14,
    });
    const fittedZoom = state.simulationMap.getZoom();
    state.simulationZoomMin = Math.max(2, fittedZoom - 1);
    state.simulationZoomMax = Math.min(19, fittedZoom + 3);
    state.simulationMap.setMinZoom(state.simulationZoomMin);
    state.simulationMap.setMaxZoom(state.simulationZoomMax);
  }
  syncSimulationZoomControls();
}

function confidenceRegionPolygons(region) {
  if (Array.isArray(region?.water_polygons)) {
    return region.water_polygons
      .map((polygon) => [
        polygon.exterior || [],
        ...(polygon.holes || []),
      ])
      .filter((polygon) => polygon[0]?.length >= 3);
  }
  const polygon = region?.polygon || [];
  return polygon.length >= 3 ? [[polygon]] : [];
}

function drawSimulationRegion(ctx, region, project, color, alpha) {
  const polygons = confidenceRegionPolygons(region);
  if (!polygons.length) return;
  ctx.beginPath();
  for (const polygon of polygons) {
    for (const ring of polygon) {
      ring.forEach((point, index) => {
        const [x, y] = project(Number(point[0]), Number(point[1]));
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
    }
  }
  ctx.fillStyle = color;
  ctx.globalAlpha = alpha;
  ctx.fill("evenodd");
  ctx.globalAlpha = 1;
}

function buildSimulationBackdrop(width, height) {
  const prediction = state.simulationData;
  const ensemble = prediction?.simulation_ensemble;
  if (!ensemble) return;
  const backdrop = document.createElement("canvas");
  backdrop.width = width;
  backdrop.height = height;
  const ctx = backdrop.getContext("2d");
  const project = state.simulationMap
    ? (lat, lon) => {
      const point = state.simulationMap.latLngToContainerPoint([lat, lon]);
      return [point.x, point.y];
    }
    : simulationProjector(ensemble.bounds, width, height);
  state.simulationProjection = project;

  ctx.clearRect(0, 0, width, height);

  [...(prediction.forecast_confidence_regions || [])]
    .reverse()
    .forEach((region) => {
      const alpha = Number(region.level) === 50 ? 0.055 :
        Number(region.level) === 80 ? 0.032 : 0.018;
      drawSimulationRegion(ctx, region, project, "#6fc9e8", alpha);
    });
  [...(prediction.elapsed_confidence_regions || [])]
    .reverse()
    .forEach((region) => {
      const alpha = Number(region.level) === 50 ? 0.04 :
        Number(region.level) === 80 ? 0.024 : 0.014;
      drawSimulationRegion(ctx, region, project, "#dca956", alpha);
    });

  ctx.lineWidth = 0.55;
  for (const sample of ensemble.paths) {
    const flat = sample.path;
    ctx.beginPath();
    for (let index = 0; index < flat.length; index += 2) {
      const [x, y] = project(Number(flat[index]), Number(flat[index + 1]));
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.globalAlpha = 0.018;
    ctx.strokeStyle =
      simulationBehaviorColors[sample.behavior] || "#6fc9e8";
    ctx.stroke();
  }
  ctx.globalAlpha = 1;

  const origin = ensemble.paths[0]?.path;
  if (origin?.length >= 2) {
    const [x, y] = project(Number(origin[0]), Number(origin[1]));
    ctx.strokeStyle = "#ffffff";
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x - 5, y);
    ctx.lineTo(x + 5, y);
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
  state.simulationBackdrop = backdrop;
}

function drawSimulationDensity(ctx, positions, forecast, width, height) {
  const resolution = 4;
  const densityWidth = Math.max(1, Math.ceil(width / resolution));
  const densityHeight = Math.max(1, Math.ceil(height / resolution));
  const counts = new Uint16Array(densityWidth * densityHeight);
  let maximum = 0;
  for (const position of positions) {
    const x = Math.max(
      0,
      Math.min(densityWidth - 1, Math.floor(position.x / resolution))
    );
    const y = Math.max(
      0,
      Math.min(densityHeight - 1, Math.floor(position.y / resolution))
    );
    const index = y * densityWidth + x;
    counts[index] += 1;
    maximum = Math.max(maximum, counts[index]);
  }
  if (!maximum) return;
  const density = state.simulationDensityCanvas || document.createElement("canvas");
  state.simulationDensityCanvas = density;
  if (density.width !== densityWidth || density.height !== densityHeight) {
    density.width = densityWidth;
    density.height = densityHeight;
  }
  const densityCtx = density.getContext("2d");
  const image = densityCtx.createImageData(densityWidth, densityHeight);
  const color = forecast ? [125, 165, 178] : [178, 157, 112];
  for (let index = 0; index < counts.length; index += 1) {
    if (!counts[index]) continue;
    const intensity = Math.pow(counts[index] / maximum, 0.55);
    const offset = index * 4;
    image.data[offset] = color[0];
    image.data[offset + 1] = color[1];
    image.data[offset + 2] = color[2];
    image.data[offset + 3] = Math.round(150 * intensity);
  }
  densityCtx.putImageData(image, 0, 0);
  ctx.save();
  ctx.globalAlpha = 0.82;
  ctx.filter = "blur(8px)";
  ctx.drawImage(density, 0, 0, width, height);
  ctx.restore();
}

function drawSimulationTrails(
  ctx,
  ensemble,
  segment,
  fraction,
  project,
  forecast
) {
  ctx.save();
  ctx.lineWidth = 0.7;
  ctx.globalAlpha = forecast ? 0.12 : 0.085;
  for (const sample of ensemble.paths) {
    const flat = sample.path;
    const firstSegment = Math.max(0, segment - 5);
    const firstOffset = Math.min(firstSegment * 2, flat.length - 2);
    ctx.beginPath();
    const [startX, startY] = project(
      Number(flat[firstOffset]),
      Number(flat[firstOffset + 1])
    );
    ctx.moveTo(startX, startY);
    for (let trailSegment = firstSegment + 1; trailSegment <= segment; trailSegment += 1) {
      const offset = Math.min(trailSegment * 2, flat.length - 2);
      const [x, y] = project(
        Number(flat[offset]),
        Number(flat[offset + 1])
      );
      ctx.lineTo(x, y);
    }
    const currentOffset = Math.min(segment * 2, flat.length - 2);
    const nextOffset = Math.min(currentOffset + 2, flat.length - 2);
    const lat =
      Number(flat[currentOffset])
      + (Number(flat[nextOffset]) - Number(flat[currentOffset])) * fraction;
    const lon =
      Number(flat[currentOffset + 1])
      + (Number(flat[nextOffset + 1]) - Number(flat[currentOffset + 1])) * fraction;
    const [currentX, currentY] = project(lat, lon);
    ctx.lineTo(currentX, currentY);
    ctx.strokeStyle = forecast
      ? "#9bb8c3"
      : simulationBehaviorColors[sample.behavior] || "#9bb8c3";
    ctx.stroke();
  }
  ctx.restore();
}

function resizeSimulationCanvas() {
  if (els.simulationViewer.classList.contains("hidden")) return;
  if (state.simulationMap) {
    state.simulationMap.invalidateSize({ pan: false });
  }
  const rect = els.simulationViewerCanvas.getBoundingClientRect();
  const width = Math.max(240, Math.round(rect.width));
  const height = Math.max(200, Math.round(rect.height));
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
  els.simulationViewerCanvas.width = Math.round(width * pixelRatio);
  els.simulationViewerCanvas.height = Math.round(height * pixelRatio);
  buildSimulationBackdrop(width, height);
  drawSimulationFrame();
}

function simulationTimelinePosition(progress) {
  const timeline =
    state.simulationData?.simulation_ensemble?.timeline_minutes || [0];
  const totalMinutes = Number(timeline[timeline.length - 1]) || 1;
  const minute = Math.max(0, Math.min(totalMinutes, progress * totalMinutes));
  let segment = 0;
  while (
    segment < timeline.length - 2 &&
    Number(timeline[segment + 1]) < minute
  ) {
    segment += 1;
  }
  const startMinute = Number(timeline[segment]) || 0;
  const endMinute = Number(timeline[segment + 1]) || startMinute + 1;
  const fraction = Math.max(
    0,
    Math.min(1, (minute - startMinute) / Math.max(0.0001, endMinute - startMinute))
  );
  return { minute, segment, fraction };
}

function drawSimulationFrame() {
  const prediction = state.simulationData;
  const ensemble = prediction?.simulation_ensemble;
  const project = state.simulationProjection;
  if (!ensemble || !project || !state.simulationBackdrop) return;
  const canvas = els.simulationViewerCanvas;
  const pixelRatio = Math.min(2, window.devicePixelRatio || 1);
  const width = canvas.width / pixelRatio;
  const height = canvas.height / pixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(state.simulationBackdrop, 0, 0);

  const { minute, segment, fraction } = simulationTimelinePosition(
    state.simulationProgress
  );
  const currentMinute = Number(
    ensemble.timeline_minutes[ensemble.current_path_index]
  ) || 0;
  const isForecast = minute > currentMinute;
  els.simulationViewerPhase.textContent = isForecast
    ? "Forward outlook"
    : "Possible movement since last fix";
  els.simulationViewerTime.textContent = isForecast
    ? `+${formatSilenceAge(Math.max(0, minute - currentMinute) * 60)}`
    : formatSilenceAge(minute * 60);
  els.simulationViewerTimeline.value = String(
    Math.round(state.simulationProgress * 1000)
  );

  const positions = [];
  for (let sampleIndex = 0; sampleIndex < ensemble.paths.length; sampleIndex += 1) {
    const sample = ensemble.paths[sampleIndex];
    const flat = sample.path;
    const startOffset = Math.min(segment * 2, flat.length - 2);
    const endOffset = Math.min(startOffset + 2, flat.length - 2);
    const lat =
      Number(flat[startOffset])
      + (Number(flat[endOffset]) - Number(flat[startOffset])) * fraction;
    const lon =
      Number(flat[startOffset + 1])
      + (Number(flat[endOffset + 1]) - Number(flat[startOffset + 1])) * fraction;
    const [x, y] = project(lat, lon);
    positions.push({
      x,
      y,
      lat,
      lon,
      behavior: sample.behavior,
      index: sampleIndex,
    });
  }
  state.simulationPositions = positions;
  drawSimulationDensity(ctx, positions, isForecast, width, height);
  drawSimulationTrails(
    ctx,
    ensemble,
    segment,
    fraction,
    project,
    isForecast
  );

  for (const [behavior, color] of Object.entries(simulationBehaviorColors)) {
    ctx.beginPath();
    for (const position of positions) {
      if (position.behavior !== behavior) continue;
      ctx.moveTo(position.x + 1.25, position.y);
      ctx.arc(position.x, position.y, 1.25, 0, Math.PI * 2);
    }
    ctx.fillStyle = color;
    ctx.globalAlpha = isForecast ? 0.82 : 0.72;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function animateSimulationViewer(timestamp) {
  if (
    els.simulationViewer.classList.contains("hidden") ||
    !state.simulationData
  ) {
    state.simulationFrame = null;
    return;
  }
  if (
    state.simulationLastDrawAt !== null
    && timestamp - state.simulationLastDrawAt < 32
  ) {
    state.simulationFrame = requestAnimationFrame(animateSimulationViewer);
    return;
  }
  state.simulationLastDrawAt = timestamp;
  if (state.simulationPlaying) {
    if (state.simulationLastFrameAt !== null) {
      state.simulationProgress +=
        (timestamp - state.simulationLastFrameAt) / 14000;
      if (state.simulationProgress > 1) state.simulationProgress %= 1;
    }
    state.simulationLastFrameAt = timestamp;
    drawSimulationFrame();
  } else {
    state.simulationLastFrameAt = null;
  }
  state.simulationFrame = requestAnimationFrame(animateSimulationViewer);
}

function initializeSimulationViewer(prediction, vessel) {
  const ensemble = prediction.simulation_ensemble;
  state.simulationData = prediction;
  state.simulationProgress = 0;
  state.simulationPlaying = true;
  state.simulationLastFrameAt = null;
  state.simulationLastDrawAt = null;
  els.simulationViewerTitle.textContent = "Monte Carlo Simulation";
  els.simulationViewerContext.textContent =
    vessel.name || `MMSI ${vessel.mmsi}`;
  els.simulationViewerCount.textContent =
    `${ensemble.count.toLocaleString()} paths`;
  const totalMinutes =
    Number(ensemble.timeline_minutes[ensemble.timeline_minutes.length - 1]) || 1;
  const currentMinute =
    Number(ensemble.timeline_minutes[ensemble.current_path_index]) || 0;
  els.simulationViewerTimeline.style.setProperty(
    "--current-boundary",
    `${Math.max(0, Math.min(100, currentMinute / totalMinutes * 100))}%`
  );
  renderSimulationPlayControl();
  const counts = {};
  for (const sample of ensemble.paths) {
    counts[sample.behavior] = (counts[sample.behavior] || 0) + 1;
  }
  els.simulationViewerLegend.innerHTML = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([behavior, count]) => {
      const color = simulationBehaviorColors[behavior] || "#6fc9e8";
      return `<span style="color:${color}"><i></i>` +
        `${escapeHtml(simulationBehaviorLabels[behavior] || behavior)} ` +
        `${count}</span>`;
    })
    .join("");
  els.simulationViewerLoading.classList.add("hidden");
  prepareSimulationMap(ensemble.bounds);
  resizeSimulationCanvas();
  stopSimulationAnimation();
  state.simulationFrame = requestAnimationFrame(animateSimulationViewer);
}

async function openSimulationViewer() {
  const vessel = state.selectedVessel;
  if (!vessel?.dark) return;
  const requestId = ++state.simulationRequestId;
  const key = predictionCacheKey(vessel);
  state.simulationViewerKey = key;
  els.simulationViewer.classList.remove("hidden");
  els.simulationViewerLoading.innerHTML =
    `<span></span>Running the navigation model…`;
  els.simulationViewerLoading.classList.remove("hidden");
  els.simulationViewerTooltip.classList.add("hidden");
  stopSimulationAnimation();
  let prediction = state.simulationCache.get(key);
  try {
    if (!prediction) {
      prediction = await getJson(
        `/api/global/${encodeURIComponent(vessel.mmsi)}/prediction?include_samples=1`
      );
      for (const existingKey of state.simulationCache.keys()) {
        if (
          existingKey.startsWith(`${vessel.mmsi}:`) &&
          existingKey !== key
        ) {
          state.simulationCache.delete(existingKey);
        }
      }
      state.simulationCache.set(key, prediction);
    }
  } catch (_err) {
    if (requestId === state.simulationRequestId) {
      els.simulationViewerLoading.innerHTML =
        "The simulation ensemble is temporarily unavailable.";
    }
    return;
  }
  if (
    requestId !== state.simulationRequestId ||
    state.selectedMmsi !== vessel.mmsi
  ) {
    return;
  }
  initializeSimulationViewer(prediction, vessel);
}

function closeSimulationViewer() {
  state.simulationRequestId += 1;
  state.simulationViewerKey = null;
  state.simulationData = null;
  state.simulationPositions = [];
  state.simulationBackdrop = null;
  stopSimulationAnimation();
  els.simulationViewer.classList.add("hidden");
  els.simulationViewerTooltip.classList.add("hidden");
}

function hideTrajectory({ restoreView = true } = {}) {
  stopMapFlight();
  stopTrajectoryAnimation();
  closeSimulationViewer();
  cancelPredictionLoading();
  setTrajectoryModesDisabled(false);
  state.trajectoryRenderId += 1;
  const returnView = state.preSelectionView;
  state.preSelectionView = null;
  if (state.selectedMmsi !== null) {
    postJson("/api/global/pin", { mmsi: null }).catch(() => {});
  }
  const selectedMarker = globalMarkers.get(state.selectedMmsi);
  state.selectedMmsi = null;
  state.selectedPredictionKey = null;
  selectedVesselLayer.clearLayers();
  if (selectedMarker) {
    selectedMarker.setStyle(globalMarkerStyle(selectedMarker._fix || {}));
    selectedMarker.closeTooltip();
  }
  state.selectedVessel = null;
  if (state.globalLayerOn) renderLiveRail(state.globalStatus);
  els.overviewSelection.classList.add("hidden");
  els.overviewSelection.innerHTML = "";
  els.trajectoryNearby.innerHTML = "";
  if (state.globalLayerOn) {
    renderLivePanelEmptyState();
    refreshNearbyContext();
    if (document.getElementById("tab-eval").classList.contains("active")) {
      renderLiveChecks();
    }
  }
  globalProjectionLayer.clearLayers();
  fishingActivityLayer.clearLayers();
  els.trajectoryPanel.classList.add("hidden");
  document.getElementById("tab-vessel-button").classList.add("hidden");
  activateSidebarPanel("tab-overview");
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
  const age = vesselSilenceAge(v);
  state.selectedVessel = { ...v, age_s: age };
  if (state.globalLayerOn) renderLiveRail(state.globalStatus);
  const lastFix = new Date((Number(v.last_seen) || Date.now() / 1000) * 1000);
  const risk = v.risk || { low_usd: 0, high_usd: 0, items: [] };
  const latitude = formatLatitude(v.lat);
  const longitude = formatLongitude(v.lon);
  els.trajectoryName.textContent = v.name || `MMSI ${v.mmsi}`;
  els.trajectoryMeta.innerHTML =
    `<div class="vessel-status-line">` +
    `<strong class="${v.dark ? "is-silent" : "is-reporting"}">` +
    `${v.dark
      ? `AIS silent · ${escapeHtml(formatSilenceAge(age))}`
      : "AIS reporting"}</strong>` +
    `<span>Last fix ${escapeHtml(lastFix.toISOString().slice(11, 19))} UTC</span></div>` +
    `<div class="vessel-motion">${Number(v.speed_kn || 0).toFixed(1)} kn · ` +
    `${Number(v.course || 0).toFixed(0)}°` +
    `${v.destination
      ? ` · ${escapeHtml(String(v.destination).replaceAll("<>", " → "))}`
      : ""}</div>` +
    `<div class="vessel-identifiers">MMSI ${escapeHtml(v.mmsi)}` +
    `${v.imo ? ` · IMO ${escapeHtml(v.imo)}` : ""}` +
    `${v.call_sign ? ` · ${escapeHtml(v.call_sign)}` : ""}</div>` +
    `<div class="vessel-position">` +
    `<div><span>Latitude</span><strong>${escapeHtml(latitude || "Unavailable")}</strong></div>` +
    `<div><span>Longitude</span><strong>${escapeHtml(longitude || "Unavailable")}</strong></div>` +
    `</div>`;
  els.overviewSelection.classList.remove("hidden");
  els.overviewSelection.innerHTML =
    `<span>Selected vessel</span>` +
    `<div><strong>${escapeHtml(v.name || `MMSI ${v.mmsi}`)}</strong>` +
    `<b>${Number(v.speed_kn || 0).toFixed(1)} kn</b></div>` +
    `<small>${v.dark
      ? `AIS silent · ${escapeHtml(formatSilenceAge(age))}`
      : "AIS reporting"} · ${Number(v.course || 0).toFixed(0)}° course</small>` +
    `<small class="overview-position">${escapeHtml(latitude || "Latitude unavailable")} · ` +
    `${escapeHtml(longitude || "Longitude unavailable")}</small>`;
  els.trajectoryRisk.innerHTML =
    `<div class="trajectory-heading">Response planning</div>` +
    `<div class="trajectory-cost">${(risk.items || []).length
      ? `${risk.items.length} response option${risk.items.length === 1 ? "" : "s"}`
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

function vesselSilenceAge(v) {
  const reportedAge = Math.max(0, Number(v?.age_s) || 0);
  const lastSeen = Number(v?.last_seen);
  if (!v?.dark || !Number.isFinite(lastSeen)) return reportedAge;
  return Math.max(reportedAge, Date.now() / 1000 - lastSeen);
}

function predictionCacheKey(v) {
  const bucket = Math.floor(vesselSilenceAge(v) / 300);
  return `${v.mmsi}:${v.last_seen}:${bucket}`;
}

function trajectoryBounds(scenarios, start, confidenceRegions = []) {
  const bounds = L.latLngBounds(
    scenarios.flatMap((scenario) => scenario.path).concat([start])
  );
  for (const region of confidenceRegions) {
    for (const polygon of confidenceRegionPolygons(region)) {
      for (const ring of polygon) {
        for (const point of ring) bounds.extend(point);
      }
    }
  }
  return bounds;
}

function trajectoryCameraOptions(bounds) {
  const diagonalM = map.distance(
    bounds.getSouthWest(),
    bounds.getNorthEast()
  );
  const maxZoom =
    diagonalM < 1000 ? 16 :
    diagonalM < 5000 ? 15 :
    diagonalM < 25000 ? 13 :
    diagonalM < 100000 ? 11 :
    diagonalM < 500000 ? 9 : 7;
  const viewport = map.getSize();
  const paddingPx = Math.max(
    48,
    Math.min(
      96,
      Math.round(Math.min(viewport.x, viewport.y) * (
        diagonalM >= 100000 ? 0.16 : 0.11
      ))
    )
  );
  return {
    padding: [paddingPx, paddingPx],
    maxZoom,
  };
}

function setTrajectoryModesDisabled(disabled) {
  for (const button of document.querySelectorAll(".trajectory-mode")) {
    button.disabled = disabled;
  }
}

function setPredictionLoadingStage(name, status, label) {
  const row = els.trajectoryLoadingStages.find(
    (stage) => stage.dataset.loadingStage === name
  );
  if (!row) return;
  row.classList.toggle("is-loading", status === "loading");
  row.classList.toggle("is-ready", status === "ready");
  const statusLabel = row.querySelector("b");
  if (statusLabel) {
    statusLabel.textContent = label || (status === "ready" ? "Ready" : "Loading");
  }
}

function showPredictionLoading(v) {
  state.predictionLoadingSince = performance.now();
  state.predictionLoadingMmsi = v.mmsi;
  els.trajectoryLoadingVessel.textContent =
    `${v.name || `MMSI ${v.mmsi}`} · ${formatLatitude(v.lat)} · ${formatLongitude(v.lon)}`;
  for (const name of ["terrain", "ocean", "weather"]) {
    setPredictionLoadingStage(name, "loading", "Loading");
  }
  setPredictionLoadingStage("routes", "loading", "Calculating");
  els.trajectoryLoading.classList.add("is-visible");
  els.trajectoryLoading.setAttribute("aria-hidden", "false");
}

async function hidePredictionLoading(mmsi = state.predictionLoadingMmsi) {
  if (mmsi !== state.predictionLoadingMmsi) return;
  const visibleFor = performance.now() - state.predictionLoadingSince;
  if (visibleFor < 520) {
    await new Promise((resolve) => window.setTimeout(resolve, 520 - visibleFor));
  }
  if (mmsi !== state.predictionLoadingMmsi) return;
  setPredictionLoadingStage("routes", "ready", "Ready");
  els.trajectoryLoading.classList.remove("is-visible");
  els.trajectoryLoading.setAttribute("aria-hidden", "true");
  state.predictionLoadingMmsi = null;
}

function cancelPredictionLoading() {
  els.trajectoryLoading.classList.remove("is-visible");
  els.trajectoryLoading.setAttribute("aria-hidden", "true");
  state.predictionLoadingMmsi = null;
}

async function warmPredictionConditions(v, renderId) {
  for (let attempt = 0; attempt < 24; attempt += 1) {
    if (state.selectedMmsi !== v.mmsi || state.trajectoryRenderId !== renderId) return;
    let conditions;
    try {
      conditions = await getJson(
        `/api/global/${encodeURIComponent(v.mmsi)}/conditions`
      );
    } catch (_err) {
      await new Promise((resolve) => window.setTimeout(resolve, 1200));
      continue;
    }
    if (state.selectedMmsi !== v.mmsi || state.trajectoryRenderId !== renderId) return;
    renderEnvironmentalMetrics(conditions);
    const oceanReady = Boolean(conditions.ocean_conditions?.available);
    const oceanPending = Boolean(conditions.ocean_conditions?.pending);
    const weatherReady = Boolean(conditions.weather_conditions?.available);
    const weatherPending = Boolean(conditions.weather_conditions?.pending);
    const terrainReady = Boolean(conditions.terrain?.available);
    const terrainPending = Boolean(conditions.terrain?.pending);
    setPredictionLoadingStage(
      "ocean",
      oceanReady ? "ready" : oceanPending ? "loading" : "idle",
      oceanReady ? "Ready" : oceanPending ? "Loading" : "Unavailable"
    );
    setPredictionLoadingStage(
      "weather",
      weatherReady ? "ready" : weatherPending ? "loading" : "idle",
      weatherReady ? "Ready" : weatherPending ? "Loading" : "Unavailable"
    );
    setPredictionLoadingStage(
      "terrain",
      terrainReady ? "ready" : terrainPending ? "loading" : "idle",
      terrainReady ? "Ready" : terrainPending ? "Loading" : "Unavailable"
    );
    if (!oceanPending && !weatherPending && !terrainPending) return;
    await new Promise((resolve) => window.setTimeout(resolve, 1200));
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
        predictionCacheKey(latest) !== cacheKey
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
    closeSimulationViewer();
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
  refreshNearbyContext();

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
      `${portLabel} · ${formatMaritimeDistanceKm(context.port?.distance_km)} away`
    );
  }
  if (context.on_land) contextLines.push("Position intersects coastline data");
  for (const cable of context.near_cables || []) {
    contextLines.push(
      `${escapeHtml(cable.name)} cable · ${formatMaritimeDistanceKm(cable.distance_km)}`
    );
  }
  els.trajectoryContext.innerHTML = contextLines.length
    ? `<div class="trajectory-heading">Location alerts</div>${contextLines.map((line) => `<div>${line}</div>`).join("")}`
    : `<div class="context-clear">No monitored location or identity-list alerts at this position.</div>`;
  if (!inPlaceRefresh) {
    loadGfwIdentity(v.mmsi);
    loadGfwActivity(v.mmsi);
  }
  const trajectoryWasHidden = els.trajectoryPanel.classList.contains("hidden");
  els.trajectoryPanel.classList.remove("hidden");
  document.getElementById("tab-vessel-button").classList.remove("hidden");
  if (adjustCamera || trajectoryWasHidden) {
    activateSidebarPanel("trajectory-panel");
  }
  const modeSwitch = els.trajectoryPanel.querySelector(".trajectory-mode-switch");
  modeSwitch.classList.toggle("hidden", !v.dark);
  els.simulationViewerOpen.classList.toggle("hidden", !v.dark);
  if (
    !els.simulationViewer.classList.contains("hidden") &&
    state.simulationViewerKey !== predictionCacheKey(v)
  ) {
    closeSimulationViewer();
  }
  if (!v.dark) {
    closeSimulationViewer();
    cancelPredictionLoading();
    state.selectedPredictionKey = null;
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
  const cacheKey = predictionCacheKey(v);
  let prediction = state.predictionCache.get(cacheKey);
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
    warmPredictionConditions(v, renderId).catch(() => {});
    if (!prediction) showPredictionLoading(v);
  }
  state.selectedPredictionKey = cacheKey;
  let shouldPollEnvironment = false;
  try {
    if (!prediction) {
      prediction = await getJson(`/api/global/${encodeURIComponent(v.mmsi)}/prediction`);
      if (!predictionEnvironmentPending(prediction)) {
        for (const existingKey of state.predictionCache.keys()) {
          if (
            existingKey.startsWith(`${v.mmsi}:`) &&
            existingKey !== cacheKey
          ) {
            state.predictionCache.delete(existingKey);
          }
        }
        state.predictionCache.set(cacheKey, prediction);
      } else {
        shouldPollEnvironment = true;
      }
    }
  } catch (err) {
    if (state.trajectoryRenderId === renderId) {
      hidePredictionLoading(v.mmsi);
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
  if (prediction.simulation_ensemble) {
    state.simulationCache.set(cacheKey, prediction);
  }
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
  const elapsedRegions = prediction.elapsed_confidence_regions || [];
  const forecastRegions = prediction.forecast_confidence_regions ||
    prediction.confidence_regions || [];
  const modeledSilenceSeconds =
    Math.max(0, Number(prediction.modeled_silence_minutes) || 0) * 60;
  els.trajectoryHeading.textContent =
    `Possible movement since last fix · ${formatSilenceAge(modeledSilenceSeconds)}`;
  const confidenceLabel = prediction.confidence?.label || "unknown";
  els.trajectoryOptions.innerHTML =
    `<div class="trajectory-confidence">` +
    `<span><strong>50%</strong> likely region</span>` +
    `<span><strong>80%</strong> likely region</span>` +
    `<span><strong>95%</strong> possible region</span>` +
    `<small>Forward outlook · next ${Number(prediction.horizon_minutes) || 30} minutes · ` +
    `${escapeHtml(confidenceLabel)} confidence</small></div>`;
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
    const spread = formatMaritimeDistanceMeters(scenario.uncertainty_radius_m);
    const endpoint = scenario.path?.[scenario.path.length - 1];
    const endpointLatitude = endpoint ? formatLatitude(endpoint[0]) : null;
    const endpointLongitude = endpoint ? formatLongitude(endpoint[1]) : null;
    const row = document.createElement("div");
    row.className = "trajectory-option";
    row.innerHTML =
      `<span class="trajectory-swatch" style="background:${color}"></span>` +
      `<span class="trajectory-route-summary">` +
      `<strong>${behaviorLabels[scenario.behavior] || `Route ${index + 1}`}</strong>` +
      `<small>±${spread} uncertainty</small></span>` +
      `<span class="trajectory-distance">${probability.toFixed(1)}% confidence</span>` +
      `<small class="trajectory-endpoint">Predicted endpoint · ` +
      `${escapeHtml(endpointLatitude || "Latitude unavailable")} · ` +
      `${escapeHtml(endpointLongitude || "Longitude unavailable")}</small>`;
    els.trajectoryOptions.appendChild(row);
  });
  const bounds = trajectoryBounds(
    scenarios,
    start,
    elapsedRegions.concat(forecastRegions)
  );
  if (adjustCamera && bounds.isValid()) {
    const cameraOptions = trajectoryCameraOptions(bounds);
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
  for (const [phase, regions] of [
    ["forecast", forecastRegions],
    ["elapsed", elapsedRegions],
  ]) {
    [...regions].reverse().forEach((region) => {
      const level = Number(region.level) || 95;
      const fillOpacity = level === 50 ? 0.16 : level === 80 ? 0.095 : 0.05;
      confidenceRegionPolygons(region).forEach((polygon) => {
        L.polygon(polygon, {
          color: phase === "elapsed" ? "#ffb020" : "#6fc9e8",
          weight: 0,
          opacity: 0,
          fillColor: phase === "elapsed" ? "#ffb020" : "#6fc9e8",
          fillOpacity,
          fillRule: "evenodd",
          interactive: false,
          className: `confidence-region confidence-${phase} confidence-${level}`,
        }).addTo(globalProjectionLayer);
      });
    });
  }
  renderOceanCurrents(prediction.ocean_conditions);
  renderEnvironmentalMetrics(prediction);
  renderEnvironmentalVisuals(prediction, start);

  for (const segment of prediction.observed_history?.segments || []) {
    const path = segment.path || [];
    if (path.length < 2) continue;
    const durationMinutes = Number(segment.duration_minutes);
    const durationLabel = Number.isFinite(durationMinutes)
      ? formatSilenceAge(durationMinutes * 60)
      : "reported fixes";
    L.polyline(path, {
      color: "#ffb020",
      weight: 2,
      opacity: 0.65,
      interactive: true,
      className: "ais-observed-trail",
    })
      .bindTooltip(
        `AIS trail · ${escapeHtml(durationLabel)} before signal loss`,
        { sticky: true, direction: "top", opacity: 0.94 }
      )
      .addTo(globalProjectionLayer);
  }
  scenarios.forEach((scenario, index) => {
    const prominent = index < 3;
    const color = colors[index % colors.length];
    const path = scenario.path;
    const end = path[path.length - 1];
    const currentIndex = Math.max(
      0,
      Math.min(
        path.length - 1,
        Number(scenario.current_path_index ?? prediction.current_path_index) || 0
      )
    );
    const elapsedPath = path.slice(0, currentIndex + 1);
    const forecastPath = path.slice(currentIndex);
    const animationPath = forecastPath.length > 1
      ? forecastPath
      : path.slice(Math.max(0, path.length - 2));
    const phaseOffset = index / Math.max(1, scenarios.length);
    const initialProgress = Math.min(1, phaseOffset / 0.88);
    const initialScaled = initialProgress * (animationPath.length - 1);
    const initialSegment = Math.min(
      animationPath.length - 2,
      Math.floor(initialScaled)
    );
    const initialFraction = initialScaled - initialSegment;
    const initialFrom = animationPath[initialSegment];
    const initialTo = animationPath[initialSegment + 1];
    const initialPosition = [
      initialFrom[0] + (initialTo[0] - initialFrom[0]) * initialFraction,
      initialFrom[1] + (initialTo[1] - initialFrom[1]) * initialFraction,
    ];
    if (elapsedPath.length > 1) {
      const elapsedLine = L.polyline(elapsedPath, {
        color: "#ffb020",
        weight: prominent ? 1.6 : 0.75,
        opacity: prominent ? 0.68 : 0.2,
        dashArray: "3 5",
        interactive: prominent,
        className: `trajectory-path trajectory-elapsed trajectory-path-${index}`,
      });
      if (prominent) {
        elapsedLine.bindTooltip(
          `Modeled movement since last AIS fix · ${formatSilenceAge(modeledSilenceSeconds)}`,
          { sticky: true, direction: "top", opacity: 0.94 }
        );
      }
      elapsedLine.addTo(globalProjectionLayer);
    }
    if (forecastPath.length > 1) {
      const forecastLine = L.polyline(forecastPath, {
        color,
        weight: prominent ? 2 : 0.75,
        opacity: prominent ? 0.86 : 0.2,
        dashArray: "6 5",
        interactive: prominent,
        className: `trajectory-path trajectory-forecast trajectory-path-${index}`,
      });
      if (prominent) {
        forecastLine.bindTooltip(
          `Forward outlook · next ${Number(prediction.horizon_minutes) || 30} minutes`,
          { sticky: true, direction: "top", opacity: 0.94 }
        );
      }
      forecastLine.addTo(globalProjectionLayer);
    }
    L.circleMarker(path[currentIndex], {
      radius: prominent ? 3 : 1.5,
      color: "#ffcf66",
      weight: 1,
      opacity: prominent ? 0.9 : 0.3,
      fillColor: "#ffb020",
      fillOpacity: prominent ? 0.9 : 0.3,
      interactive: false,
      className: "trajectory-current-position",
    }).addTo(globalProjectionLayer);
    const routeOriginPoint = map.latLngToLayerPoint(animationPath[0]);
    const routeSpanPx = animationPath.reduce(
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
      animationPath[0],
      animationPath[animationPath.length - 1],
      Number(v.course) || Number(v.heading) || 0
    );
    const runner = L.marker(initialPosition, {
      icon: trajectoryRunnerIcon(color, shortRoute || !prominent),
      interactive: false,
      keyboard: false,
      zIndexOffset: 1000,
    }).addTo(globalProjectionLayer);
    runner._trajectoryRunner = true;
    runners.push({
      marker: runner,
      path: animationPath,
      phaseOffset,
      shortRoute,
      opacityScale: prominent ? 1 : 0.35,
      stableBearing,
      displayBearing: stableBearing,
    });
  });
  const drivers = prediction.uncertainty_drivers || [];
  const current = prediction.ocean_conditions?.center;
  const wave = prediction.ocean_conditions?.wave;
  const wind = prediction.weather_conditions?.center;
  const silenceSeconds = Math.max(
    0,
    Number(prediction.modeled_silence_minutes) * 60 || vesselSilenceAge(v)
  );
  const currentText = current
    ? `Nearby ocean flow is ${Number(current.speed_mps).toFixed(2)} m/s ` +
      `toward ${Number(current.bearing_deg).toFixed(0)}° and is applied only to the forward outlook. `
    : prediction.ocean_conditions?.pending
      ? "Ocean-current data for the forward outlook is still loading. "
      : "No current ocean-flow reading was available for the forward outlook. ";
  const waveText = wave?.available
    ? `Copernicus reports ${Number(wave.height_m).toFixed(1)} m waves for the forward outlook. `
    : "";
  const windText = wind
    ? `NOAA wind is ${Number(wind.speed_mps).toFixed(1)} m/s toward ` +
      `${Number(wind.bearing_deg).toFixed(0)}° and contributes to forward estimated leeway. `
    : prediction.weather_conditions?.pending
      ? "Wind forcing is loading. "
      : "";
  const timingText =
    `Possible movement covers the full ${formatSilenceAge(silenceSeconds)} since the last fix, ` +
    `followed by a separate ${Number(prediction.horizon_minutes) || 30}-minute outlook. `;
  const historicalText = prediction.environment_evidence?.historical_gap?.available
    ? "Historical environmental observations were applied across the AIS gap. "
    : "Historical current, wind and wave observations were unavailable, so live conditions were not projected backward across the AIS gap. ";
  const terrainText = prediction.signal_availability?.global_terrain
    ? "Routes and confidence areas are constrained to mapped navigable water, including coastal channels and harbor passages. "
    : "A detailed navigable-water mask was unavailable for this area. ";
  const driverLabels = {
    course_over_ground: "recent direction",
    true_heading: "vessel heading",
    speed_over_ground: "recent speed",
    rate_of_turn: "turning rate",
    track_history: "movement history",
    historical_environment_unavailable: "historical environmental evidence",
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
    historicalText +
    terrainText +
    `${prediction.confidence?.reason || "Confidence reflects the AIS gap and available navigation evidence"}. ` +
    qualityText;
  startTrajectoryAnimation(runners, { preservePhase: inPlaceRefresh });
  if (!inPlaceRefresh) hidePredictionLoading(v.mmsi);
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
      // Keep the rendered boat compact while giving it a forgiving click
      // target, especially at regional zoom levels with dense traffic.
      radius: 12,
      ...globalMarkerStyle(v),
      interactive: true,
      bubblingMouseEvents: false,
    });
    m.on("click", () => {
      m.closeTooltip();
      const latest = state.globalVessels.get(m._fix.mmsi) || m._fix;
      showTrajectory(latest);
    });
    m.bindTooltip("", { sticky: true });
    globalMarkers.set(v.mmsi, m);
  } else {
    m.setLatLng([v.lat, v.lon]);
    m.setStyle(globalMarkerStyle(v));
  }
  m._stale = false;
  m._fix = v;
  m._dark = !!v.dark;
  syncGlobalMarkerVisibility(m, v);
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
els.simulationViewerOpen.addEventListener("click", openSimulationViewer);
els.simulationViewerClose.addEventListener("click", closeSimulationViewer);
els.simulationViewerZoomIn.addEventListener("click", () => {
  changeSimulationZoom(1);
});
els.simulationViewerZoomOut.addEventListener("click", () => {
  changeSimulationZoom(-1);
});
els.simulationViewerPlay.addEventListener("click", () => {
  state.simulationPlaying = !state.simulationPlaying;
  state.simulationLastFrameAt = null;
  renderSimulationPlayControl();
});
els.simulationViewerRestart.addEventListener("click", () => {
  state.simulationProgress = 0;
  state.simulationLastFrameAt = null;
  drawSimulationFrame();
});
els.simulationViewerTimeline.addEventListener("input", () => {
  state.simulationProgress =
    Math.max(0, Math.min(1, Number(els.simulationViewerTimeline.value) / 1000));
  state.simulationLastFrameAt = null;
  drawSimulationFrame();
});
els.simulationViewerCanvas.addEventListener("pointermove", (event) => {
  if (!state.simulationPositions.length) return;
  const rect = els.simulationViewerCanvas.getBoundingClientRect();
  const pointerX = event.clientX - rect.left;
  const pointerY = event.clientY - rect.top;
  let nearest = null;
  let nearestDistance = 64;
  for (const position of state.simulationPositions) {
    const distance =
      (position.x - pointerX) ** 2 + (position.y - pointerY) ** 2;
    if (distance < nearestDistance) {
      nearest = position;
      nearestDistance = distance;
    }
  }
  if (!nearest) {
    els.simulationViewerTooltip.classList.add("hidden");
    return;
  }
  els.simulationViewerTooltip.classList.remove("hidden");
  els.simulationViewerTooltip.style.left =
    `${Math.min(rect.width - 175, pointerX + 10)}px`;
  els.simulationViewerTooltip.style.top =
    `${Math.max(8, pointerY - 52)}px`;
  els.simulationViewerTooltip.innerHTML =
    `<strong>Simulation ${nearest.index + 1}</strong>` +
    `${escapeHtml(simulationBehaviorLabels[nearest.behavior] || nearest.behavior)}<br>` +
    `${nearest.lat.toFixed(4)}, ${nearest.lon.toFixed(4)}`;
});
els.simulationViewerCanvas.addEventListener("pointerleave", () => {
  els.simulationViewerTooltip.classList.add("hidden");
});
els.simulationViewerCanvas.addEventListener("wheel", (event) => {
  if (!state.simulationMap) return;
  event.preventDefault();
  const rect = els.simulationViewerCanvas.getBoundingClientRect();
  changeSimulationZoom(
    event.deltaY < 0 ? 0.5 : -0.5,
    L.point(event.clientX - rect.left, event.clientY - rect.top)
  );
}, { passive: false });
window.addEventListener("resize", resizeSimulationCanvas);
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

function portDetail(port, selectedScope) {
  const details = [
    port.country,
    selectedScope && Number.isFinite(Number(port.distance_km))
      ? formatMaritimeDistanceKm(port.distance_km)
      : "",
    port.port_of_entry ? "port of entry" : "",
    port.pilot_required ? "pilot required" : "",
    port.tug_assist ? "tug assistance" : "",
  ].filter(Boolean);
  return details.join(" · ");
}

function nearbyPortRows(ports, selectedScope) {
  if (!ports.length) {
    return `<p class="nearby-empty">No indexed ports in this area.</p>`;
  }
  return ports.slice(0, selectedScope ? 8 : 6).map((port) =>
    `<button class="nearby-row" type="button" ` +
    `data-lat="${Number(port.lat)}" data-lon="${Number(port.lon)}">` +
    `<span class="nearby-symbol nearby-symbol-port" aria-hidden="true"></span>` +
    `<span><strong>${escapeHtml(port.name)}</strong>` +
    `<small>${escapeHtml(portDetail(port, selectedScope))}</small></span>` +
    `</button>`
  ).join("");
}

function bindNearbyRows(container) {
  for (const row of container.querySelectorAll(".nearby-row")) {
    row.addEventListener("click", () => {
      const lat = Number(row.dataset.lat);
      const lon = Number(row.dataset.lon);
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        map.flyTo([lat, lon], Math.max(map.getZoom(), 8), {
          duration: 0.8,
        });
      }
    });
  }
}

function renderNearbyMapFeatures(ports) {
  nearbyContextLayer.clearLayers();
  const permanentLabels = map.getZoom() >= 6;
  for (const port of ports.slice(0, 12)) {
    const marker = L.circleMarker([port.lat, port.lon], {
      radius: 4,
      color: "#b8f1dd",
      weight: 1.5,
      fillColor: "#163e35",
      fillOpacity: 0.9,
    });
    marker.bindTooltip(escapeHtml(port.name), {
      permanent: permanentLabels,
      direction: "right",
      offset: [7, 0],
      className: "port-map-label",
    });
    marker.on("click", () => {
      map.flyTo([port.lat, port.lon], Math.max(map.getZoom(), 9), {
        duration: 0.8,
      });
    });
    marker.addTo(nearbyContextLayer);
  }
}

function renderNearbyContext(data) {
  state.nearbyContext = data;
  const selectedScope = data.scope === "selected_vessel";
  const ports = data.ports || [];
  renderNearbyMapFeatures(ports);
  const content =
    `<div class="nearby-heading">${selectedScope ? "Nearby context" : "Map context"}</div>` +
    `<div class="nearby-section-title">` +
    `${selectedScope
      ? `Ports within ${formatMaritimeDistanceKm(data.radius_km || 150)}`
      : "Ports in view"}` +
    `<span>${ports.length}</span></div>` +
    `<div class="nearby-list">${nearbyPortRows(ports, selectedScope)}</div>` +
    `<p class="nearby-source">NGA World Port Index · click a port to locate it</p>`;
  if (selectedScope && state.selectedVessel) {
    els.trajectoryNearby.innerHTML = content;
    bindNearbyRows(els.trajectoryNearby);
    return;
  }
  const rail = document.getElementById("live-rail-context");
  if (rail) {
    rail.innerHTML = content;
    bindNearbyRows(rail);
  }
}

function refreshNearbyContext() {
  window.clearTimeout(state.contextRefreshTimer);
  state.contextRefreshTimer = window.setTimeout(async () => {
    const requestId = ++state.contextRequestId;
    let url;
    if (state.selectedVessel) {
      const vessel = state.selectedVessel;
      url =
        `/api/context/nearby?lat=${encodeURIComponent(vessel.lat)}` +
        `&lon=${encodeURIComponent(vessel.lon)}&radius_km=150`;
      els.trajectoryNearby.innerHTML =
        `<div class="nearby-heading">Nearby context</div>` +
        `<div class="nearby-loading"><span></span>Loading nearby ports…</div>`;
    } else {
      const bounds = map.getBounds();
      url =
        `/api/context/nearby?west=${encodeURIComponent(bounds.getWest())}` +
        `&south=${encodeURIComponent(bounds.getSouth())}` +
        `&east=${encodeURIComponent(bounds.getEast())}` +
        `&north=${encodeURIComponent(bounds.getNorth())}`;
    }
    try {
      const data = await getJson(url);
      if (requestId === state.contextRequestId) renderNearbyContext(data);
    } catch (_err) {
      if (requestId !== state.contextRequestId) return;
      const target = state.selectedVessel
        ? els.trajectoryNearby
        : document.getElementById("live-rail-context");
      if (target) {
        target.innerHTML =
          `<div class="nearby-heading">Map context</div>` +
          `<p class="nearby-empty">Port context is temporarily unavailable.</p>`;
      }
    }
  }, 220);
}

function renderLiveRail(status) {
  if (!document.getElementById("live-rail-live-count")) {
    els.log.innerHTML =
      `<section class="ops-status">` +
      `<div class="ops-section-label">Operational status</div>` +
      `<div class="ops-status-line">` +
      `<span id="live-rail-status-dot" class="ops-status-dot"></span>` +
      `<div><strong id="live-rail-status">Establishing picture</strong>` +
      `<small id="live-rail-status-detail">Waiting for current feed status</small></div>` +
      `</div></section>` +
      `<section class="intel-counters" aria-label="Maritime contact counters">` +
      `<article><span>Live contacts</span><strong id="live-rail-live-count">0</strong></article>` +
      `<article><span>Dark contacts</span><strong id="live-rail-dark-count">0</strong></article>` +
      `<article><span>Risk alerts</span><strong id="live-rail-risk-count">0</strong></article>` +
      `<article><span>Tracked</span><strong id="live-rail-total-count">0</strong></article>` +
      `</section>` +
      `<section class="ops-section">` +
      `<header><span>Active alerts</span><b id="live-rail-alert-count">0</b></header>` +
      `<div id="live-rail-alerts" class="ops-alerts"></div>` +
      `</section>` +
      `<section class="ops-section">` +
      `<header><span>Feed status</span><b id="live-rail-provider">—</b></header>` +
      `<div class="feed-status-grid">` +
      `<div><span>AIS feed</span><strong id="live-rail-feed-health">Checking</strong></div>` +
      `<div><span>Last update</span><strong id="live-rail-last-update">—</strong></div>` +
      `<div><span>Position reports</span><strong id="live-rail-position-reports">0</strong></div>` +
      `<div><span>Vessel records</span><strong id="live-rail-static-reports">0</strong></div>` +
      `<div><span>Identity conflicts</span><strong id="live-rail-identities">0</strong></div>` +
      `</div></section>` +
      `<section class="ops-section registry-section">` +
      `<header><span>Vessel registry</span><b id="live-rail-registry-state">Awaiting selection</b></header>` +
      `<dl id="live-rail-registry" class="registry-grid"></dl>` +
      `</section>` +
      `<div id="live-rail-context" class="live-rail-context"></div>`;
    refreshNearbyContext();
  }
  const counts = globalContactCounts();
  const silenceThreshold = aisSilenceThreshold();
  const vessels = Array.from(state.globalVessels.values());
  const riskContacts = vessels.filter((vessel) =>
    (vessel.risk?.items || []).some((item) =>
      String(item.severity).toLowerCase() === "critical"
    )
  ).length;
  const protectedContacts = vessels.filter(
    (vessel) => vessel.context?.in_sanctuary
  ).length;
  const identityConflicts = Number(status.identity_switches || 0);
  const feedHealthy = Boolean(status.connected && state.globalLive);
  const feedConnected = Boolean(status.connected);
  const pictureElevated = riskContacts > 0 || identityConflicts > 0;
  const statusTone = !feedConnected
    ? "offline"
    : !feedHealthy || pictureElevated
      ? "degraded"
      : "normal";
  const statusLabel = !feedConnected
    ? "OFFLINE"
    : !feedHealthy
      ? "DEGRADED"
      : pictureElevated
        ? "ELEVATED"
        : "NORMAL";
  const statusDetail = !feedConnected
    ? "Primary AIS feed is unavailable"
    : !feedHealthy
      ? "Connected; awaiting current position reports"
      : pictureElevated
        ? "Priority contact review is required"
        : "Live maritime picture is current";
  const lastMessageAt = Number(status.last_message_at);
  const lastUpdateAge = Number.isFinite(lastMessageAt)
    ? Math.max(0, Date.now() / 1000 - lastMessageAt)
    : null;
  const lastUpdateText = lastUpdateAge === null
    ? "No message"
    : lastUpdateAge < 1
      ? "<1 second ago"
      : `${formatSilenceAge(lastUpdateAge)} ago`;
  const alerts = [];
  if (!feedHealthy) {
    alerts.push({
      tone: feedConnected ? "warning" : "critical",
      title: feedConnected ? "AIS position stream delayed" : "AIS feed disconnected",
      detail: statusDetail,
    });
  }
  if (counts.dark > 0) {
    alerts.push({
      tone: "warning",
      title: `${counts.dark.toLocaleString()} contacts without AIS`,
      detail: `No position report for more than ${silenceThreshold} seconds`,
    });
  }
  if (riskContacts > 0) {
    alerts.push({
      tone: "critical",
      title: `${riskContacts.toLocaleString()} contacts require review`,
      detail: "Operational risk signals are active",
    });
  }
  if (identityConflicts > 0) {
    alerts.push({
      tone: "warning",
      title: `${identityConflicts.toLocaleString()} identity changes observed`,
      detail: "Identity evidence requires correlation",
    });
  }
  if (protectedContacts > 0) {
    alerts.push({
      tone: "neutral",
      title: `${protectedContacts.toLocaleString()} contacts in protected waters`,
      detail: "Location context only; not evidence of wrongdoing",
    });
  }
  if (!alerts.length) {
    alerts.push({
      tone: "verified",
      title: "No priority alerts",
      detail: "Current contact picture remains within monitoring thresholds",
    });
  }

  document.getElementById("live-rail-status").textContent = statusLabel;
  document.getElementById("live-rail-status-detail").textContent = statusDetail;
  document.getElementById("live-rail-status-dot").className =
    `ops-status-dot ${statusTone}`;
  document.getElementById("live-rail-live-count").textContent =
    counts.active.toLocaleString();
  document.getElementById("live-rail-dark-count").textContent =
    counts.dark.toLocaleString();
  document.getElementById("live-rail-risk-count").textContent =
    riskContacts.toLocaleString();
  document.getElementById("live-rail-total-count").textContent =
    counts.total.toLocaleString();
  document.getElementById("live-rail-alert-count").textContent =
    String(alerts.length);
  document.getElementById("live-rail-alerts").innerHTML = alerts.slice(0, 4).map((alert) =>
    `<div class="ops-alert ${alert.tone}"><span></span><div>` +
    `<strong>${escapeHtml(alert.title)}</strong>` +
    `<small>${escapeHtml(alert.detail)}</small></div></div>`
  ).join("");
  document.getElementById("live-rail-provider").textContent =
    String(status.provider || "AIS").replaceAll("_", " ").toUpperCase();
  document.getElementById("live-rail-feed-health").textContent =
    feedHealthy ? "Healthy" : feedConnected ? "Delayed" : "Offline";
  document.getElementById("live-rail-feed-health").className =
    feedHealthy ? "normal" : feedConnected ? "degraded" : "offline";
  document.getElementById("live-rail-last-update").textContent = lastUpdateText;
  document.getElementById("live-rail-position-reports").textContent =
    Number(status.position_reports || 0).toLocaleString();
  document.getElementById("live-rail-static-reports").textContent =
    Number(status.static_reports || 0).toLocaleString();
  document.getElementById("live-rail-identities").textContent =
    identityConflicts.toLocaleString();

  const vessel = state.selectedVessel;
  document.getElementById("live-rail-registry-state").textContent =
    vessel ? "Contact selected" : "Awaiting selection";
  const registryRows = [
    ["MMSI", vessel?.mmsi],
    ["IMO", vessel?.imo],
    ["Call sign", vessel?.call_sign],
    ["Flag", vessel?.flag || vessel?.country],
    ["Type", aisShipTypeLabel(vessel?.ship_type)],
    ["Destination", vessel?.destination
      ? String(vessel.destination).replaceAll("<>", " → ")
      : null],
    ["Latitude", formatLatitude(vessel?.lat)],
    ["Longitude", formatLongitude(vessel?.lon)],
    ["Course", vessel ? `${Number(vessel.course || 0).toFixed(0)}°` : null],
    ["Last AIS", vessel
      ? formatSilenceAge(vesselSilenceAge(vessel))
      : null],
  ];
  document.getElementById("live-rail-registry").innerHTML = registryRows.map(
    ([label, value]) =>
      `<div><dt>${escapeHtml(label)}</dt><dd>${value
        ? escapeHtml(value)
        : "—"}</dd></div>`
  ).join("");
}

function frameRegionalProvider() {
  const provider = state.globalStatus?.provider;
  if (
    provider !== "digitraffic" ||
    state.globalProviderFramed === provider ||
    !state.globalVessels.size
  ) {
    return;
  }
  const bounds = L.latLngBounds(
    Array.from(state.globalVessels.values(), (vessel) => [
      Number(vessel.lat),
      Number(vessel.lon),
    ])
  );
  if (!bounds.isValid()) return;
  map.fitBounds(bounds.pad(0.08), {
    animate: false,
    maxZoom: 7,
  });
  state.globalProviderFramed = provider;
  refreshNearbyContext();
}

function setGlobalLayer() {
  state.globalLayerOn = true;
  document.body.classList.add("live-mode");
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
  frameRegionalProvider();
  if (document.getElementById("tab-eval").classList.contains("active")) {
    loadEval();
  }
}

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
    if (state.selectedMmsi !== null) {
      const selected = state.globalVessels.get(state.selectedMmsi) ||
        state.selectedVessel;
      if (selected) {
        const agedSelected = {
          ...selected,
          age_s: vesselSilenceAge(selected),
        };
        refreshSelectedVessel(agedSelected);
        if (
          agedSelected.dark &&
          predictionCacheKey(agedSelected) !== state.selectedPredictionKey
        ) {
          showTrajectory(agedSelected, {
            adjustCamera: false,
            refreshEnvironment: true,
          });
        }
      }
    }
    if (state.globalLayerOn) {
      frameRegionalProvider();
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

for (const button of els.vesselVisibilityBtns) {
  button.addEventListener("click", () => {
    setVesselVisibility(button.dataset.vesselVisibility);
  });
}
let initialVesselVisibility = "both";
try {
  initialVesselVisibility =
    localStorage.getItem("aegis-vessel-visibility") || "both";
} catch (_error) {
  // Use the default when browser storage is unavailable.
}
setVesselVisibility(initialVesselVisibility);
setGlobalLayer();
connect();
jtmsReset().then(() => {
  if (state.globalLayerOn) renderLivePanelEmptyState();
  else loadBrief();
});
