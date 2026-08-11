"""Aegis web server: precomputed replay, REST transport, WS stream.

It runs the pure-Python runtime pipeline once at startup, caches
every frame's render-ready delta, and then just plays the cache back. That is
what makes "scrubbing instant" true: a seek is an array index, never a
re-run of the tracker.

Identity discipline continues past the tracker boundary: the id-switch
counter is computed here from ground truth (tracker.metrics), because the
frontend must never see which track_id belongs to which vessel -- only the
scalar count crosses the wire.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line, '#' comments) so a real
    aisstream.io key can be picked up without adding a new dependency.
    Never overwrites a variable already set in the real environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")

import numpy as np
from aiohttp import web, WSMsgType

from data.scenario import load_scenario
from data.global_ais import create_global_feed, global_snapshot
from data.dark_prediction import predict_dark_vessel
from data.global_fishing_watch import GlobalFishingWatchClient
from data.maritime_context import maritime_context
from data.ocean_conditions import OceanConditionsClient
from data.weather_conditions import WeatherConditionsClient
from data.world_ports import WorldPortIndexClient
from data.geocoding import GeocodingClient
from aegis.graph import build_mission
from aegis.main import run_frame_scored
from aegis.fusion import assoc_provenance as main_assoc_source
from aegis import jtms
from aegis.brief import panel_payload
from tracker.metrics import TrackingMetrics
from tracker import eval as tracker_eval

HISTORY_CAP = 150  # log/alert lines sent on a sync; the rail is a scroller, not an archive

# Real packs on disk, for the dashboard's scenario picker. Order matters --
# this is display order, s01 (the headline dark-vessel demo) first.
KNOWN_PACKS = ["s01_dark_in_sanctuary", "s02_synthetic_demo", "s02_mmsi_spoof", "s03_ghost_fleet"]


# --------------------------------------------------------------------- precompute

def precompute(pack_id: str, max_frames: int = -1, assoc_mode: str = "global") -> dict:
    """Run the full Aegis pipeline once and cache everything the server needs.

    Returns a dict with the per-frame delta cache plus flat, prefix-summed
    log/alert history so a seek can slice "everything up to here" in O(1)
    lookup + O(k) slice, never by re-deriving it. `assoc_mode` ("global" or
    "greedy") selects the stage-1 associator.
    """
    scenario = load_scenario(pack_id)
    mission = build_mission(scenario)
    mission.meta["rng"] = np.random.default_rng(
        int(scenario.expected.get("seed", 0)) + 20260726 + 1
    )
    metrics = TrackingMetrics()

    frames = []
    all_log: list[str] = []
    all_alerts: list[dict] = []
    log_end: list[int] = []
    alert_end: list[int] = []

    for frame in scenario.frames:
        if max_frames >= 0 and frame.idx >= max_frames:
            break
        delta, pairs = run_frame_scored(mission, scenario, frame, assoc_mode=assoc_mode)
        metrics.update(pairs)
        delta["id_switches"] = metrics.id_switches

        all_log.extend(delta["log"])
        all_alerts.extend(delta["alerts"])
        log_end.append(len(all_log))
        alert_end.append(len(all_alerts))
        frames.append(delta)

    return {
        "pack_id": scenario.pack_id,
        "assoc_mode": assoc_mode,
        "frame_interval_s": scenario.frame_interval_s,
        "zones": frames[0]["zones"] if frames else [],
        "frames": frames,
        "all_log": all_log,
        "all_alerts": all_alerts,
        "log_end": log_end,
        "alert_end": alert_end,
        "expected": scenario.expected,
    }


# ------------------------------------------------------------------------ jtms

def _jtms_state(flipped: list | None = None) -> dict:
    """JSON-serialisable snapshot of the live MMSI-spoof evidence graph:
    every Fact's believed flag, every Conclusion's status and
    a fresh brief_for() sentence, and (if this call followed a
    retract/reinstate) which conclusion ids just flipped."""
    facts = {fid: {"label": jtms.FACTS[fid].label, "believed": jtms.fact_believed(fid)}
              for fid in jtms.fact_ids()}
    concls = {cid: {"label": jtms.CONCLUSIONS[cid].label,
                     "status": jtms.conclusion_status(cid),
                     "brief": jtms.brief_for(cid)}
               for cid in jtms.conclusion_ids()}
    return {"facts": facts, "conclusions": concls, "flipped": flipped or []}


def history_for(cache: dict, frame_idx: int) -> tuple[list[str], list[dict]]:
    log_upto = cache["log_end"][frame_idx]
    alert_upto = cache["alert_end"][frame_idx]
    log = cache["all_log"][max(0, log_upto - HISTORY_CAP):log_upto]
    alerts = cache["all_alerts"][max(0, alert_upto - HISTORY_CAP):alert_upto]
    return log, alerts


# ------------------------------------------------------------------- playback

def build_app(cache: dict) -> web.Application:
    app = web.Application()
    app["caches"] = {(cache["pack_id"], cache["assoc_mode"]): cache}
    app["active_key"] = (cache["pack_id"], cache["assoc_mode"])
    app["state"] = {"frame_idx": 0, "playing": False, "speed": 10.0}
    app["clients"] = set()

    def active() -> dict:
        return app["caches"][app["active_key"]]

    def n_frames() -> int:
        return len(active()["frames"])

    async def broadcast(message: dict) -> None:
        dead = set()
        payload = json.dumps(message)
        for ws in app["clients"]:
            try:
                await ws.send_str(payload)
            except ConnectionResetError:
                dead.add(ws)
        app["clients"] -= dead

    app["broadcast"] = broadcast

    async def broadcast_state() -> None:
        await broadcast({"type": "state", **app["state"]})

    async def broadcast_sync(frame_idx: int) -> None:
        cache = active()
        log, alerts = history_for(cache, frame_idx)
        await broadcast({
            "type": "sync",
            **cache["frames"][frame_idx],
            "history_log": log,
            "history_alerts": alerts,
            "pack_id": cache["pack_id"],
            "assoc_mode": cache["assoc_mode"],
            "assoc_source": main_assoc_source(cache["assoc_mode"]),
            "total_frames": len(cache["frames"]),
            "zones": cache["zones"],
            **app["state"],
        })

    app["broadcast_state"] = broadcast_state
    app["broadcast_sync"] = broadcast_sync

    async def player_loop(app: web.Application) -> None:
        state = app["state"]
        while True:
            cache = active()
            n = len(cache["frames"])
            if state["playing"] and state["frame_idx"] < n - 1:
                await asyncio.sleep(cache["frame_interval_s"] / max(state["speed"], 1e-3))
                if not state["playing"]:
                    continue  # paused mid-sleep
                state["frame_idx"] += 1
                await app["broadcast"]({
                    "type": "frame",
                    **cache["frames"][state["frame_idx"]],
                    "playing": state["playing"],
                    "speed": state["speed"],
                })
                if state["frame_idx"] >= n - 1:
                    state["playing"] = False
                    await app["broadcast_state"]()
            else:
                await asyncio.sleep(0.05)

    async def start_player(app: web.Application) -> None:
        app["player_task"] = asyncio.create_task(player_loop(app))

    app.on_startup.append(start_player)

    api_key = os.environ.get("AISSTREAM_API_KEY", "")
    ais_provider = os.environ.get("AEGIS_AIS_PROVIDER", "aisstream")
    ais_state_path = Path(os.environ.get(
        "AEGIS_AIS_STATE_PATH",
        REPO_ROOT / ".aegis" / "ais_state.json.gz",
    ))
    provider_state_path = (
        ais_state_path
        if ais_provider.strip().lower() == "aisstream"
        else ais_state_path.with_name(
            f"{ais_state_path.stem}.{ais_provider.strip().lower()}"
            f"{ais_state_path.suffix}"
        )
    )
    app["global_feed"] = create_global_feed(
        ais_provider,
        aisstream_api_key=api_key,
        state_path=provider_state_path,
    )
    gfw_token = os.environ.get("GFW_API_TOKEN", "")
    wdpa_path = os.environ.get("AEGIS_WDPA_CSV", "")
    if not wdpa_path:
        supplied_wdpa = (
            Path.home()
            / "Downloads"
            / "WDPA_WDOECM_Aug2026_Public_marine_csv"
            / "WDPA_WDOECM_Aug2026_Public_marine_csv.csv"
        )
        if supplied_wdpa.is_file():
            wdpa_path = str(supplied_wdpa)
    app["gfw_client"] = (
        GlobalFishingWatchClient(gfw_token, protected_areas_path=wdpa_path)
        if gfw_token else None
    )
    app["ocean_client"] = OceanConditionsClient()
    app["weather_client"] = WeatherConditionsClient()
    app["ports_client"] = WorldPortIndexClient()
    app["geocoding_client"] = GeocodingClient()
    app["ocean_tasks"] = set()
    app["ocean_pending"] = set()
    app["weather_pending"] = set()

    async def start_global_feed(app: web.Application) -> None:
        if app["global_feed"] is not None:
            app["global_feed_task"] = asyncio.create_task(app["global_feed"].run())

    async def stop_global_feed(app: web.Application) -> None:
        task = app.get("global_feed_task")
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        feed = app.get("global_feed")
        if feed is not None:
            feed.save_state()

    app.on_startup.append(start_global_feed)
    app.on_cleanup.append(stop_global_feed)

    # ------------------------------------------------------------------ REST

    async def api_scenario(request: web.Request) -> web.Response:
        cache = active()
        return web.json_response({
            "pack_id": cache["pack_id"],
            "assoc_mode": cache["assoc_mode"],
            "frame_interval_s": cache["frame_interval_s"],
            "total_frames": len(cache["frames"]),
            "zones": cache["zones"],
            "expected": cache["expected"],
            "known_packs": KNOWN_PACKS,
            "warm": [{"pack_id": k[0], "assoc_mode": k[1]} for k in app["caches"]],
        })

    async def api_state(request: web.Request) -> web.Response:
        return web.json_response({"total_frames": n_frames(), **app["state"]})

    async def api_play(request: web.Request) -> web.Response:
        body = await _body(request)
        state = app["state"]
        if "speed" in body:
            state["speed"] = max(float(body["speed"]), 0.1)
        if state["frame_idx"] >= n_frames() - 1:
            state["frame_idx"] = 0  # replaying past the end restarts
        state["playing"] = True
        await app["broadcast_state"]()
        return web.json_response(state)

    async def api_pause(request: web.Request) -> web.Response:
        app["state"]["playing"] = False
        await app["broadcast_state"]()
        return web.json_response(app["state"])

    async def api_speed(request: web.Request) -> web.Response:
        body = await _body(request)
        app["state"]["speed"] = max(float(body.get("speed", 1.0)), 0.1)
        await app["broadcast_state"]()
        return web.json_response(app["state"])

    async def api_seek(request: web.Request) -> web.Response:
        body = await _body(request)
        frame = max(0, min(n_frames() - 1, int(body.get("frame", 0))))
        app["state"]["frame_idx"] = frame
        await app["broadcast_sync"](frame)
        return web.json_response(app["state"])

    async def api_reset(request: web.Request) -> web.Response:
        app["state"]["frame_idx"] = 0
        app["state"]["playing"] = False
        await app["broadcast_sync"](0)
        return web.json_response(app["state"])

    async def api_switch(request: web.Request) -> web.Response:
        """Switch the active scenario pack and/or association mode. Blocks
        while the (pack_id, assoc_mode) combination is precomputed for the
        first time -- there is no partial/streaming precompute -- then it is
        cached for every later switch back. Real runtime pipeline run, same as
        startup; s01's 326-vessel window can take a couple of minutes the
        first time, this is not a demo shortcut."""
        body = await _body(request)
        cur_pack, cur_mode = app["active_key"]
        pack_id = body.get("pack_id", cur_pack)
        assoc_mode = body.get("assoc_mode", cur_mode)
        if pack_id not in KNOWN_PACKS:
            return web.json_response({"error": f"unknown pack_id: {pack_id}"}, status=400)
        if assoc_mode not in ("global", "greedy"):
            return web.json_response({"error": f"unknown assoc_mode: {assoc_mode}"}, status=400)
        key = (pack_id, assoc_mode)
        if key not in app["caches"]:
            app["caches"][key] = precompute(pack_id, assoc_mode=assoc_mode)
        app["active_key"] = key
        app["state"]["frame_idx"] = 0
        app["state"]["playing"] = False
        cache = app["caches"][key]
        await app["broadcast"]({
            "type": "init",
            "pack_id": cache["pack_id"],
            "assoc_mode": cache["assoc_mode"],
            "assoc_source": main_assoc_source(cache["assoc_mode"]),
            "frame_interval_s": cache["frame_interval_s"],
            "total_frames": len(cache["frames"]),
            "zones": cache["zones"],
        })
        await app["broadcast_sync"](0)
        return web.json_response({"pack_id": pack_id, "assoc_mode": assoc_mode})

    async def api_jtms_reset(request: web.Request) -> web.Response:
        jtms.reset()
        jtms.build_mmsi_spoof_demo()
        flipped = jtms.propagate()
        return web.json_response(_jtms_state(flipped))

    async def api_jtms_retract(request: web.Request) -> web.Response:
        body = await _body(request)
        fact_id = body.get("fact_id", "")
        if not jtms.fact_ids():
            jtms.build_mmsi_spoof_demo()
            jtms.propagate()
        try:
            flipped = jtms.retract(fact_id)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(_jtms_state(flipped))

    async def api_jtms_reinstate(request: web.Request) -> web.Response:
        body = await _body(request)
        fact_id = body.get("fact_id", "")
        try:
            flipped = jtms.reinstate(fact_id)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(_jtms_state(flipped))

    async def api_jtms_state(request: web.Request) -> web.Response:
        if not jtms.fact_ids():
            jtms.build_mmsi_spoof_demo()
            jtms.propagate()
        return web.json_response(_jtms_state())

    async def api_brief(request: web.Request) -> web.Response:
        if not jtms.conclusion_ids():
            jtms.build_mmsi_spoof_demo()
            jtms.propagate()
        ids = request.query.get("concl_ids", "")
        concl_ids = [c for c in ids.split(",") if c] or jtms.conclusion_ids()
        use_llm = request.query.get("use_llm", "1") not in ("0", "false", "False")
        payload = panel_payload(concl_ids, use_llm=use_llm)
        return web.json_response({"briefs": payload})

    async def api_eval(request: web.Request) -> web.Response:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, tracker_eval.evaluate_all)
        packs = [{
            "pack_id": r.pack_id,
            "name": r.name,
            "passed": r.passed,
            "failed": r.failed,
            "skipped": r.skipped,
            "marginal": r.marginal,
            "ok": r.ok,
            "checks": [
                {"key": c.key, "outcome": c.outcome, "detail": c.detail, "marginal": c.marginal}
                for c in r.checks
            ],
        } for r in results]
        return web.json_response({
            "packs": packs,
            "totals": {
                "passed": sum(p["passed"] for p in packs),
                "failed": sum(p["failed"] for p in packs),
                "skipped": sum(p["skipped"] for p in packs),
            },
        })

    async def api_global(request: web.Request) -> web.Response:
        try:
            since = max(0, int(request.query.get("since", "0")))
        except ValueError:
            return web.json_response({"error": "since must be an integer"}, status=400)
        return web.json_response(global_snapshot(app["global_feed"], since=since))

    async def api_global_pin(request: web.Request) -> web.Response:
        feed = app["global_feed"]
        if feed is None:
            return web.json_response({"pinned_mmsi": None, "live": False})
        body = await _body(request)
        raw_mmsi = body.get("mmsi")
        if raw_mmsi is None:
            feed.pin(None)
        else:
            mmsi = int(raw_mmsi)
            if not 100_000_000 <= mmsi <= 999_999_999:
                return web.json_response({"error": "mmsi must be 9 digits"}, status=400)
            feed.pin(mmsi)
        return web.json_response({"pinned_mmsi": feed.pinned_mmsi, "live": feed.live})

    async def api_dark_prediction(request: web.Request) -> web.Response:
        feed = app["global_feed"]
        mmsi = int(request.match_info["mmsi"])
        if feed is None or mmsi not in feed.vessels:
            return web.json_response({"error": "vessel not found"}, status=404)
        vessel = feed.vessel_snapshot(mmsi)
        if vessel is None:
            return web.json_response({"error": "vessel not found"}, status=404)
        if not vessel.get("dark"):
            return web.json_response(
                {"error": "trajectory prediction is only available after AIS silence"},
                status=409,
            )
        loop = asyncio.get_running_loop()
        lat = float(vessel["lat"])
        lon = float(vessel["lon"])
        client = app["ocean_client"]
        ocean = client.cached_current_grid(lat, lon)
        if ocean is None:
            ocean = {
                "configured": client.configured,
                "available": False,
                "pending": client.configured,
                "source": "Copernicus Marine Service",
            }
            cache_region = (round(lat * 4), round(lon * 4))
            if client.configured and cache_region not in app["ocean_pending"]:
                app["ocean_pending"].add(cache_region)

                async def warm_ocean() -> None:
                    try:
                        await asyncio.to_thread(client.current_grid, lat, lon)
                    finally:
                        app["ocean_pending"].discard(cache_region)

                task = asyncio.create_task(warm_ocean())
                app["ocean_tasks"].add(task)
                task.add_done_callback(app["ocean_tasks"].discard)

        weather_client = app["weather_client"]
        weather = weather_client.cached_conditions(lat, lon)
        if weather is None:
            weather = {
                "configured": True,
                "available": False,
                "pending": True,
                "source": "NOAA Global Forecast System",
            }
            weather_region = (round(lat * 2), round(lon * 2))
            if weather_region not in app["weather_pending"]:
                app["weather_pending"].add(weather_region)

                async def warm_weather() -> None:
                    try:
                        await asyncio.to_thread(
                            weather_client.current_conditions,
                            lat,
                            lon,
                        )
                    finally:
                        app["weather_pending"].discard(weather_region)

                task = asyncio.create_task(warm_weather())
                app["ocean_tasks"].add(task)
                task.add_done_callback(app["ocean_tasks"].discard)

        include_samples = request.query.get("include_samples") == "1"
        prediction = await loop.run_in_executor(
            None,
            lambda: predict_dark_vessel(
                vessel,
                ocean,
                weather,
                include_samples=include_samples,
            ),
        )
        return web.json_response(prediction)

    async def api_context_layers(request: web.Request) -> web.Response:
        return web.json_response({"layers": maritime_context().layer_payloads()})

    async def api_nearby_context(request: web.Request) -> web.Response:
        try:
            if "lat" in request.query and "lon" in request.query:
                lat = max(-90.0, min(90.0, float(request.query["lat"])))
                lon = max(-180.0, min(180.0, float(request.query["lon"])))
                radius_km = max(
                    10.0,
                    min(500.0, float(request.query.get("radius_km", "150"))),
                )
                lat_delta = radius_km / 111.32
                lon_delta = lat_delta / max(
                    0.15,
                    math.cos(math.radians(lat)),
                )
                west, east = lon - lon_delta, lon + lon_delta
                south, north = lat - lat_delta, lat + lat_delta
                center = (lat, lon)
                scope = "selected_vessel"
            else:
                west = float(request.query["west"])
                south = float(request.query["south"])
                east = float(request.query["east"])
                north = float(request.query["north"])
                center = (
                    (south + north) / 2,
                    (west + east) / 2,
                )
                radius_km = None
                scope = "map_view"
        except (KeyError, TypeError, ValueError):
            raise web.HTTPBadRequest(text="Invalid geographic bounds")

        west = max(-180.0, min(180.0, west))
        east = max(-180.0, min(180.0, east))
        south = max(-90.0, min(90.0, south))
        north = max(-90.0, min(90.0, north))
        if west >= east or south >= north:
            return web.json_response({
                "scope": scope,
                "ports": [],
                "source": "NGA World Port Index",
            })
        ports = await app["ports_client"].ports(
            west,
            south,
            east,
            north,
            center=center,
            limit=30,
        )
        if radius_km is not None:
            ports = [
                port
                for port in ports
                if float(port.get("distance_km", float("inf"))) <= radius_km
            ]
        return web.json_response({
            "scope": scope,
            "radius_km": radius_km,
            "ports": ports[:12],
            "source": "NGA World Port Index",
        })

    async def api_geocode(request: web.Request) -> web.Response:
        query = " ".join(request.query.get("q", "").split())
        if len(query) < 2:
            return web.json_response({"query": query, "places": []})
        if len(query) > 160:
            raise web.HTTPBadRequest(text="Search query is too long")
        places = await app["geocoding_client"].search(query)
        return web.json_response({
            "query": query,
            "places": places,
            "source": "OpenStreetMap Nominatim",
        })

    async def api_gfw_identity(request: web.Request) -> web.Response:
        mmsi = int(request.match_info["mmsi"])
        client = app["gfw_client"]
        if client is None:
            return web.json_response({
                "configured": False,
                "matched": False,
                "mmsi": mmsi,
                "source": "Global Fishing Watch",
            })
        return web.json_response({
            "configured": True,
            **await client.vessel_identity(mmsi),
        })

    async def api_gfw_activity(request: web.Request) -> web.Response:
        mmsi = int(request.match_info["mmsi"])
        client = app["gfw_client"]
        if client is None:
            return web.json_response({
                "configured": False,
                "matched": False,
                "mmsi": mmsi,
                "events": [],
                "source": "Global Fishing Watch",
            })
        return web.json_response({
            "configured": True,
            **await client.vessel_activity(mmsi),
        })

    async def api_gfw_layers(request: web.Request) -> web.Response:
        client = app["gfw_client"]
        if client is None:
            return web.json_response({"configured": False, "layers": []})
        return web.json_response(await client.map_layers())

    async def api_gfw_tile(request: web.Request) -> web.Response:
        client = app["gfw_client"]
        if client is None:
            raise web.HTTPServiceUnavailable()
        status, body, content_type = await client.map_tile(
            request.match_info["kind"],
            int(request.match_info["z"]),
            int(request.match_info["x"]),
            int(request.match_info["y"]),
        )
        return web.Response(
            status=status,
            body=body,
            content_type=content_type.split(";", 1)[0],
            headers={"Cache-Control": "public, max-age=21600"},
        )

    app.router.add_get("/api/global", api_global)
    app.router.add_post("/api/global/pin", api_global_pin)
    app.router.add_get(r"/api/global/{mmsi:\d{9}}/prediction", api_dark_prediction)
    app.router.add_get("/api/context/layers", api_context_layers)
    app.router.add_get("/api/context/nearby", api_nearby_context)
    app.router.add_get("/api/geocode", api_geocode)
    app.router.add_get(r"/api/global/{mmsi:\d{9}}/gfw", api_gfw_identity)
    app.router.add_get(
        r"/api/global/{mmsi:\d{9}}/gfw/activity",
        api_gfw_activity,
    )
    app.router.add_get("/api/gfw/layers", api_gfw_layers)
    app.router.add_get(
        r"/api/gfw/tiles/{kind:fishing|sar}/{z:\d+}/{x:\d+}/{y:\d+}.png",
        api_gfw_tile,
    )
    app.router.add_get("/api/scenario", api_scenario)
    app.router.add_get("/api/state", api_state)
    app.router.add_post("/api/play", api_play)
    app.router.add_post("/api/pause", api_pause)
    app.router.add_post("/api/speed", api_speed)
    app.router.add_post("/api/seek", api_seek)
    app.router.add_post("/api/reset", api_reset)
    app.router.add_post("/api/switch", api_switch)
    app.router.add_post("/api/jtms/reset", api_jtms_reset)
    app.router.add_post("/api/jtms/retract", api_jtms_retract)
    app.router.add_post("/api/jtms/reinstate", api_jtms_reinstate)
    app.router.add_get("/api/jtms/state", api_jtms_state)
    app.router.add_get("/api/brief", api_brief)
    app.router.add_get("/api/eval", api_eval)

    # -------------------------------------------------------------------- WS

    async def ws_handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        app["clients"].add(ws)
        cache = active()

        await ws.send_str(json.dumps({
            "type": "init",
            "pack_id": cache["pack_id"],
            "assoc_mode": cache["assoc_mode"],
            "assoc_source": main_assoc_source(cache["assoc_mode"]),
            "frame_interval_s": cache["frame_interval_s"],
            "total_frames": len(cache["frames"]),
            "zones": cache["zones"],
            "known_packs": KNOWN_PACKS,
        }))
        frame_idx = app["state"]["frame_idx"]
        log, alerts = history_for(cache, frame_idx)
        await ws.send_str(json.dumps({
            "type": "sync",
            **cache["frames"][frame_idx],
            "history_log": log,
            "history_alerts": alerts,
            "pack_id": cache["pack_id"],
            "assoc_mode": cache["assoc_mode"],
            **app["state"],
        }))

        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            app["clients"].discard(ws)
        return ws

    app.router.add_get("/ws", ws_handler)

    # ---------------------------------------------------------------- static

    web_dir = Path(__file__).resolve().parent
    app.router.add_get("/", lambda r: web.FileResponse(web_dir / "dashboard.html"))
    app.router.add_get("/dashboard", lambda r: web.FileResponse(web_dir / "dashboard.html"))
    app.router.add_static("/", web_dir, show_index=False)

    return app


async def _body(request: web.Request) -> dict:
    if request.can_read_body:
        try:
            return await request.json()
        except json.JSONDecodeError:
            return {}
    return {}


def main() -> None:
    # Synthetic pack is the default for a sub-second boot.
    # AEGIS_PACK=s01_dark_in_sanctuary exercises the acceptance criteria
    # (40+ live tracks) but costs ~2-3 minutes of precompute at startup --
    # that's the whole tracker pipeline running once, not a demo-time cost.
    pack_id = os.environ.get("AEGIS_PACK", "s02_synthetic_demo")
    max_frames = int(os.environ.get("AEGIS_FRAMES", "-1"))
    port = int(os.environ.get("PORT", "8765"))

    print(f"[server] precomputing {pack_id} ...", flush=True)
    t0 = time.perf_counter()
    cache = precompute(pack_id, max_frames)
    n = len(cache["frames"])
    print(
        f"[server] cached {n} frames in {time.perf_counter() - t0:.1f}s "
        f"(id_switches so far: {cache['frames'][-1]['id_switches'] if n else 0})",
        flush=True,
    )

    app = build_app(cache)
    web.run_app(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
