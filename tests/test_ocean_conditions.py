import threading
import time
from concurrent.futures import ThreadPoolExecutor

from data.ocean_conditions import OceanConditionsClient


def test_different_regions_fetch_concurrently():
    client = OceanConditionsClient("user", "password")
    counter_lock = threading.Lock()
    active = 0
    maximum_active = 0

    def fake_fetch(lat, lon, now):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.08)
        with counter_lock:
            active -= 1
        return {"available": True, "center": {"lat": lat, "lon": lon}}

    client._fetch = fake_fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.current_grid, 37.7, -123.0),
            executor.submit(client.current_grid, 48.5, -5.0),
        ]
        results = [future.result() for future in futures]

    assert all(result["available"] for result in results)
    assert maximum_active == 2


def test_same_region_shares_one_fetch():
    client = OceanConditionsClient("user", "password")
    calls = 0
    calls_lock = threading.Lock()

    def fake_fetch(lat, lon, now):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return {"available": True, "center": {"lat": lat, "lon": lon}}

    client._fetch = fake_fetch
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(client.current_grid, 37.7, -123.0),
            executor.submit(client.current_grid, 37.71, -123.01),
        ]
        results = [future.result() for future in futures]

    assert all(result["available"] for result in results)
    assert calls == 1
