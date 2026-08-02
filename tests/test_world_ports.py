from data.world_ports import haversine_km, parse_ports


def test_parse_ports_sorts_by_distance_and_preserves_facilities():
    payload = {
        "features": [
            {
                "geometry": {"coordinates": [2.35, 48.85]},
                "properties": {
                    "INDEX_NO": 2,
                    "PORT_NAME": "FAR PORT",
                    "COUNTRY": "FR",
                    "PILOT_REQD": "N",
                    "TUG_ASSIST": "Y",
                },
            },
            {
                "geometry": {"coordinates": [2.1, 48.1]},
                "properties": {
                    "INDEX_NO": 1,
                    "PORT_NAME": "NEAR PORT",
                    "COUNTRY": "FR",
                    "PORTOFENTR": "Y",
                    "PILOT_REQD": "Y",
                },
            },
        ],
    }

    ports = parse_ports(payload, center=(48.0, 2.0))

    assert [port["name"] for port in ports] == ["NEAR PORT", "FAR PORT"]
    assert ports[0]["port_of_entry"] is True
    assert ports[0]["pilot_required"] is True
    assert ports[0]["source"] == "NGA World Port Index"


def test_haversine_distance_matches_one_degree_latitude():
    assert 110 < haversine_km(0, 0, 1, 0) < 112
