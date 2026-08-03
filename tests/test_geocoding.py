from data.geocoding import parse_places


def test_parse_places_converts_nominatim_bounds():
    places = parse_places([{
        "display_name": "Marseille, France",
        "lat": "43.2965",
        "lon": "5.3698",
        "boundingbox": ["43.169", "43.391", "5.228", "5.533"],
        "type": "city",
        "category": "place",
    }])

    assert places == [{
        "name": "Marseille, France",
        "lat": 43.2965,
        "lon": 5.3698,
        "bounds": [[43.169, 5.228], [43.391, 5.533]],
        "type": "city",
        "category": "place",
        "source": "OpenStreetMap Nominatim",
    }]


def test_parse_places_skips_invalid_coordinates():
    assert parse_places([{"display_name": "Unknown"}]) == []
