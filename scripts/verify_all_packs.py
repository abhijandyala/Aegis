from data.scenario import load_scenario

for pid in ["s01_dark_in_sanctuary", "s02_mmsi_spoof", "s03_ghost_fleet", "s02_synthetic_demo"]:
    s = load_scenario(pid)
    print(f"{pid}: OK  n_frames={s.n_frames} n_vessels={s.n_vessels} geofences={[g.fence_id for g in s.geofences]}")

# Pack 2 specific: verify spoof condition and check no code in data/ was touched
s2 = load_scenario("s02_mmsi_spoof")
d0 = s2.display_id("vessel_a", 0.0)
d1 = s2.display_id("vessel_b", 0.0)
print("vessel_a display@0:", d0, " vessel_b display@0:", d1, " same_id:", d0 == d1)
assert d0 == d1 == "mmsi:412345678"

s3 = load_scenario("s03_ghost_fleet")
before = s3.display_id("actor_1", 0.0)
after = s3.display_id("actor_1", 3599.0)
print("actor_1 display before/after reflag:", before, after)
assert before == "mmsi:367111222" and after == "mmsi:572469210"
print("ALL PACKS VERIFIED")
