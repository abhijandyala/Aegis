from unittest.mock import patch

from data.weather_conditions import WeatherConditionsClient


class _Response:
    def __init__(self, body: str) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_noaa_units_row_is_ignored_and_wind_vector_is_parsed():
    body = (
        "time,latitude,longitude,ugrd10m,vgrd10m\n"
        "UTC,degrees_north,degrees_east,m s-1,m s-1\n"
        "2026-08-02T21:00:00Z,37.5,237.0,5.5,-8.0\n"
    )
    with patch(
        "data.weather_conditions.urllib.request.urlopen",
        return_value=_Response(body),
    ):
        result = WeatherConditionsClient().current_conditions(37.7, -123.0)

    assert result["available"] is True
    assert result["valid_at"] == "2026-08-02T21:00:00Z"
    assert result["center"]["lon"] == -123.0
    assert result["center"]["east_mps"] == 5.5
    assert result["center"]["north_mps"] == -8.0


def test_noaa_failure_returns_unavailable_reading():
    with patch(
        "data.weather_conditions.urllib.request.urlopen",
        side_effect=TimeoutError,
    ):
        result = WeatherConditionsClient().current_conditions(37.7, -123.0)

    assert result["available"] is False
    assert result["source"] == "NOAA Global Forecast System"
