from data.global_fishing_watch import (
    load_protected_area_index,
    parse_events,
    parse_vessel_search,
)


def test_parse_vessel_search_uses_matching_latest_identity():
    payload = {
        "entries": [{
            "dataset": "public-global-vessel-identity:v1",
            "registryInfo": [{
                "ssvid": "123456789",
                "shipname": "REGISTRY NAME",
                "imo": "7654321",
                "sourceCode": ["IMO"],
            }],
            "selfReportedInfo": [
                {
                    "id": "old",
                    "ssvid": "123456789",
                    "shipname": "OLD NAME",
                    "transmissionDateTo": "2020-01-01T00:00:00Z",
                },
                {
                    "id": "latest",
                    "ssvid": "123456789",
                    "shipname": "LATEST NAME",
                    "flag": "USA",
                    "positionsCounter": 42,
                    "transmissionDateTo": "2026-01-01T00:00:00Z",
                },
            ],
            "combinedSourcesInfo": [{
                "vesselId": "latest",
                "shiptypes": [{"name": "FISHING"}],
                "geartypes": [{"name": "LONGLINES"}],
            }],
        }],
    }

    result = parse_vessel_search(payload, 123456789)

    assert result["matched"] is True
    assert result["vessel_id"] == "latest"
    assert result["name"] == "LATEST NAME"
    assert result["flag"] == "USA"
    assert result["positions_count"] == 42
    assert result["ship_types"] == ["FISHING"]
    assert result["gear_types"] == ["LONGLINES"]


def test_parse_vessel_search_handles_no_match():
    result = parse_vessel_search({"entries": []}, 123456789)

    assert result == {
        "matched": False,
        "mmsi": 123456789,
        "source": "Global Fishing Watch",
    }


def test_parse_events_normalizes_activity_and_preserves_region_evidence():
    events = parse_events({
        "entries": [{
            "id": "evt-1",
            "type": "fishing",
            "start": "2026-07-01T00:00:00Z",
            "end": "2026-07-01T03:30:00Z",
            "position": {"lat": 48.2, "lon": -5.4},
            "regions": {
                "mpa": ["123"],
                "mpaNoTake": ["123"],
                "eez": ["5678"],
                "rfmo": ["NEAFC"],
            },
            "fishing": {"totalDistanceKm": 11.2},
        }],
    })

    assert events == [{
        "id": "evt-1",
        "type": "FISHING",
        "start": "2026-07-01T00:00:00Z",
        "end": "2026-07-01T03:30:00Z",
        "lat": 48.2,
        "lon": -5.4,
        "duration_hours": 3.5,
        "regions": {
            "mpa": ["123"],
            "mpaNoTake": ["123"],
            "eez": ["5678"],
            "rfmo": ["NEAFC"],
        },
        "distances": {},
        "port": None,
        "encounter": None,
        "fishing": {"totalDistanceKm": 11.2},
        "loitering": None,
        "gap": None,
        "source": "Global Fishing Watch",
        "classification": "modelled",
    }]


def test_load_protected_area_index_uses_supplied_wdpa_fields(tmp_path):
    source = tmp_path / "wdpa.csv"
    source.write_text(
        "SITE_ID,NAME_ENG,DESIG_ENG,IUCN_CAT,NO_TAKE,STATUS,"
        "STATUS_YR,MANG_AUTH,ISO3,PRNT_ISO3\n"
        "123,Blue Bank,Marine Reserve,Ia,All,Designated,2024,"
        "Marine Agency,FRA,FRA\n",
        encoding="utf-8",
    )

    index = load_protected_area_index(source)

    assert index["123"]["name"] == "Blue Bank"
    assert index["123"]["no_take"] == "All"
    assert index["123"]["country"] == "FRA"
    assert index["123"]["source"] == (
        "Protected Planet WDPA/WDOECM August 2026"
    )
