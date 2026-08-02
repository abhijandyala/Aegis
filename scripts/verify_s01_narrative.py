from data.scenario import load_scenario, _true_position_at

s = load_scenario("s01_dark_in_sanctuary")
print("n_frames", s.n_frames, "duration_s", s.duration_s)
print("geofences", [g.fence_id for g in s.geofences])

fence = s.geofences[0]
track = s.true_tracks["target_1"]

# find actual boundary crossing time on this track
crossed_at = None
for t, x, y in track:
    if fence.contains(x, y):
        crossed_at = t
        break
print("actual sanctuary entry (from track):", crossed_at)

ais = [m for f in s.frames for m in f.measurements if s.ground_truth[m.meas_id] == "target_1" and m.source == "ais"]
radar = [m for f in s.frames for m in f.measurements if s.ground_truth[m.meas_id] == "target_1" and m.source == "radar"]
print("last AIS t:", max(m.t for m in ais))
print("radar contact times:", sorted(m.t for m in radar))

# predicted-vs-true offset check
dark_t = 3135.0
radar_t = 5535.0
pos_dark = _true_position_at(track, dark_t)
pos_radar = _true_position_at(track, radar_t)
# naive constant-velocity prediction using velocity in the ~30s before going dark
pos_dark_minus = _true_position_at(track, dark_t - 30.0)
vx = (pos_dark[0] - pos_dark_minus[0]) / 30.0
vy = (pos_dark[1] - pos_dark_minus[1]) / 30.0
pred_x = pos_dark[0] + vx * (radar_t - dark_t)
pred_y = pos_dark[1] + vy * (radar_t - dark_t)
import math
offset = math.hypot(pos_radar[0] - pred_x, pos_radar[1] - pred_y)
print("predicted vs true offset at radar contact (m):", round(offset))
print("geofence contains true position at dark_t:", fence.contains(*pos_dark))
