from data.global_fishing_watch import parse_vessel_search


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
