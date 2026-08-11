import json
import asyncio

from data import global_ais


def test_unconfigured_feed_returns_no_synthetic_contacts():
    result = global_ais.global_snapshot(None)

    assert result["live"] is False
    assert result["vessels"] == []
    assert result["status"]["configured"] is False
    assert result["status"]["dark_after_s"] == 45.0


def test_real_contact_becomes_dark_after_report_timeout(monkeypatch):
    now = [1_700_000_000.0]
    monkeypatch.setattr(global_ais.time, "time", lambda: now[0])
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(
        123456789,
        {
            "mmsi": 123456789,
            "name": "Test Vessel",
            "lat": 37.8,
            "lon": -122.4,
            "course": 90.0,
            "speed_kn": 12.0,
        },
    )

    assert feed.snapshot()[0]["dark"] is False
    now[0] += global_ais.DARK_AFTER_S + 1
    assert feed.snapshot()[0]["dark"] is True
    assert feed.status()["dark_after_s"] == global_ais.DARK_AFTER_S


def test_persisted_last_report_preserves_total_silence_age(monkeypatch, tmp_path):
    now = [1_700_000_000.0]
    monkeypatch.setattr(global_ais.time, "time", lambda: now[0])
    state_path = tmp_path / "ais-state.json.gz"
    feed = global_ais.GlobalAisFeed("not-a-real-key", state_path=state_path)
    feed._record(
        123456789,
        {
            "name": "Persistent Vessel",
            "lat": 37.8,
            "lon": -122.4,
            "course": 90.0,
            "speed_kn": 12.0,
        },
    )
    now[0] += 95 * 60
    feed.save_state()

    restored = global_ais.GlobalAisFeed(
        "not-a-real-key",
        state_path=state_path,
    )
    vessel = restored.snapshot()[0]

    assert vessel["mmsi"] == 123456789
    assert vessel["age_s"] == 95 * 60
    assert vessel["dark"] is True


def test_static_voyage_data_merges_without_refreshing_position_age(monkeypatch):
    now = [1_700_000_000.0]
    monkeypatch.setattr(global_ais.time, "time", lambda: now[0])
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(123456789, {"lat": 1.0, "lon": 2.0, "name": "Old"})
    now[0] += 10

    feed._handle_message(json.dumps({
        "MessageType": "ShipStaticData",
        "MetaData": {"MMSI": 123456789},
        "Message": {
            "ShipStaticData": {
                "Name": "AEGIS TEST",
                "ImoNumber": 7654321,
                "CallSign": "WXYZ",
                "Type": 70,
                "Destination": "SFO",
                "MaximumStaticDraught": 8.5,
            }
        },
    }))

    vessel = feed.snapshot()[0]
    assert vessel["name"] == "AEGIS TEST"
    assert vessel["imo"] == 7654321
    assert vessel["destination"] == "SFO"
    assert vessel["age_s"] == 10


def test_static_only_reports_do_not_consume_position_contact_capacity():
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record_static(123456789, {"name": "STATIC FIRST", "imo": 7654321})

    assert feed.vessels == {}
    assert feed.static_data[123456789]["name"] == "STATIC FIRST"

    feed._record(123456789, {"lat": 1.0, "lon": 2.0})

    assert feed.vessels[123456789]["name"] == "STATIC FIRST"
    assert feed.vessels[123456789]["imo"] == 7654321
    assert feed.static_data == {}


def test_live_identity_switch_count_uses_observed_static_changes():
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(
        123456789,
        {
            "lat": 1.0,
            "lon": 2.0,
            "name": "VESSEL ONE",
            "imo": 7654321,
            "call_sign": "CALL1",
        },
    )

    feed._record_static(
        123456789,
        {"name": "VESSEL TWO", "imo": 7654321, "call_sign": "CALL1"},
    )
    feed._record_static(
        123456789,
        {"name": "VESSEL TWO", "imo": 7654321, "call_sign": "CALL1"},
    )
    feed._record(
        123456789,
        {
            "lat": 1.1,
            "lon": 2.1,
            "name": "VESSEL THREE",
            "imo": 7654321,
            "call_sign": "CALL1",
        },
    )

    assert feed.identity_switches == 2
    assert feed.status()["identity_switches"] == 2


def test_selected_contact_is_not_evicted_at_capacity(monkeypatch):
    monkeypatch.setattr(global_ais, "MAX_TRACKED", 2)
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(111111111, {"lat": 1.0, "lon": 1.0})
    feed._record(222222222, {"lat": 2.0, "lon": 2.0})
    feed.pin(111111111)

    feed._record(333333333, {"lat": 3.0, "lon": 3.0})

    assert set(feed.vessels) == {111111111, 333333333}
    assert feed.status()["pinned_mmsi"] == 111111111
    feed.pin(None)
    assert feed.status()["pinned_mmsi"] is None


def test_dark_contacts_survive_active_capacity_until_ais_resumes(monkeypatch):
    now = [1_700_000_000.0]
    monkeypatch.setattr(global_ais, "MAX_TRACKED", 2)
    monkeypatch.setattr(global_ais.time, "time", lambda: now[0])
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(111111111, {"lat": 1.0, "lon": 1.0})
    feed._record(222222222, {"lat": 2.0, "lon": 2.0})
    now[0] += global_ais.DARK_AFTER_S + 1
    feed.snapshot()

    feed._record(333333333, {"lat": 3.0, "lon": 3.0})
    feed._record(444444444, {"lat": 4.0, "lon": 4.0})

    assert set(feed.vessels) == {111111111, 222222222, 333333333, 444444444}
    assert feed.status()["dark_contacts"] == 2

    feed._record(111111111, {"lat": 1.1, "lon": 1.1})

    assert 111111111 in feed.vessels
    assert 111111111 not in feed._dark_mmsi
    assert len(feed._active_order) == 2


def test_incremental_snapshot_returns_only_changed_contacts():
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    feed._record(111111111, {"lat": 1.0, "lon": 1.0})
    feed._record(222222222, {"lat": 2.0, "lon": 2.0})
    initial = feed.snapshot_since()

    feed._record(111111111, {"lat": 1.1, "lon": 1.1})
    delta = feed.snapshot_since(initial["revision"])

    assert initial["full"] is True
    assert delta["full"] is False
    assert [row["mmsi"] for row in delta["vessels"]] == [111111111]
    assert delta["removed"] == []


def test_position_reports_build_bounded_history():
    feed = global_ais.GlobalAisFeed("not-a-real-key")
    for index in range(global_ais.MAX_HISTORY + 5):
        feed._record(123456789, {"lat": float(index), "lon": float(index)})

    assert len(feed.vessels[123456789]["history"]) == global_ais.MAX_HISTORY


def test_provider_factory_keeps_aisstream_and_digitraffic_selectable(tmp_path):
    aisstream = global_ais.create_global_feed(
        "aisstream",
        aisstream_api_key="test-key",
        state_path=tmp_path / "aisstream.json.gz",
    )
    digitraffic = global_ais.create_global_feed(
        "digitraffic",
        state_path=tmp_path / "digitraffic.json.gz",
    )

    assert isinstance(aisstream, global_ais.GlobalAisFeed)
    assert not isinstance(aisstream, global_ais.DigitrafficAisFeed)
    assert isinstance(digitraffic, global_ais.DigitrafficAisFeed)
    assert aisstream.status()["provider"] == "aisstream"
    assert digitraffic.status()["provider"] == "digitraffic"


def test_digitraffic_schema_maps_metadata_and_live_positions(monkeypatch):
    now = 1_700_000_100.0
    monkeypatch.setattr(global_ais.time, "time", lambda: now)
    feed = global_ais.DigitrafficAisFeed(poll_seconds=5)
    metadata = [{
        "mmsi": 230123456,
        "name": "BALTIC TEST",
        "imo": 7654321,
        "callSign": "OHTEST",
        "shipType": 70,
        "destination": "HELSINKI",
        "draught": 68,
    }]
    locations = {
        "type": "FeatureCollection",
        "features": [{
            "mmsi": 230123456,
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [24.962472, 60.1763],
            },
            "properties": {
                "mmsi": 230123456,
                "sog": 10.7,
                "cog": 326.6,
                "navStat": 0,
                "rot": 0,
                "heading": 325,
                "timestampExternal": 1_700_000_000_000,
            },
        }],
    }

    async def fake_get_json(_session, url, _etag):
        if url == global_ais.DIGITRAFFIC_VESSELS_URL:
            return metadata, "metadata-etag"
        return locations, "locations-etag"

    feed._get_json = fake_get_json
    asyncio.run(feed._refresh_metadata(None))
    asyncio.run(feed._refresh_locations(None))

    vessel = feed.snapshot()[0]
    assert vessel["mmsi"] == 230123456
    assert vessel["name"] == "BALTIC TEST"
    assert vessel["draught_m"] == 6.8
    assert vessel["lat"] == 60.1763
    assert vessel["lon"] == 24.96247
    assert vessel["course"] == 326.6
    assert vessel["speed_kn"] == 10.7
    assert vessel["age_s"] == 100
    assert vessel["dark"] is False
    assert feed.status()["coverage"] == "Finnish and Baltic waters"
    assert feed.status()["dark_after_s"] == 180


def test_digitraffic_repeated_snapshot_does_not_refresh_last_seen(monkeypatch):
    now = [1_700_000_010.0]
    monkeypatch.setattr(global_ais.time, "time", lambda: now[0])
    feed = global_ais.DigitrafficAisFeed(poll_seconds=5)
    payload = {
        "type": "FeatureCollection",
        "features": [{
            "mmsi": 230123456,
            "geometry": {"type": "Point", "coordinates": [24.9, 60.1]},
            "properties": {
                "sog": 8.0,
                "cog": 90.0,
                "timestampExternal": 1_700_000_000_000,
            },
        }],
    }

    async def fake_get_json(_session, _url, _etag):
        return payload, "locations-etag"

    feed._get_json = fake_get_json
    asyncio.run(feed._refresh_locations(None))
    first_revision = feed.revision
    now[0] += 190
    asyncio.run(feed._refresh_locations(None))

    vessel = feed.snapshot()[0]
    assert feed.revision == first_revision + 1
    assert feed.position_reports == 1
    assert vessel["age_s"] == 200
    assert vessel["dark"] is True
