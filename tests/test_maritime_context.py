from data.maritime_context import MaritimeContext


def test_every_bundled_reference_layer_is_exposed():
    context = MaritimeContext()

    assert {layer["layer_id"] for layer in context.layer_payloads()} == {
        "coastline",
        "cables",
        "sanctuary",
        "port",
    }


def test_live_fix_is_screened_against_ofac_subset():
    result = MaritimeContext().annotate(
        {
            "mmsi": 572469210,
            "name": "ARTAVIL",
            "lat": 20.0,
            "lon": 60.0,
            "dark": False,
        }
    )

    assert result["context"]["ofac"]["name"] == "ARTAVIL"
    assert result["context"]["ofac"]["match_basis"] == "mmsi"
    assert result["risk"]["high_usd"] == 589
    assert result["risk"]["items"][0]["tier"] == "desk_review"
    assert result["risk"]["source"]["rate_schedule"] == "FY26 Reimbursable Standard Rates"


def test_live_fix_is_checked_for_port_and_cable_proximity():
    context = MaritimeContext()
    port = context.annotate(
        {"mmsi": 1, "name": "Port contact", "lat": 37.79, "lon": -122.39}
    )
    cable = context.annotate(
        {"mmsi": 2, "name": "Cable contact", "lat": 35.0522, "lon": -122.8498}
    )

    assert port["context"]["in_port"] is True
    assert cable["context"]["near_cables"][0]["distance_km"] < 0.1
