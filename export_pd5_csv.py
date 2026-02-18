import csv
import time
import mujoco
import mujoco.viewer

from so101_mujoco_utils2 import (
    set_initial_pose,
    get_positions_dict,
    send_position_command,
)

MODEL_PATH = "model/scene_urdf.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

starting_position = {
    "shoulder_pan":  -4.4003158666,
    "shoulder_lift": -92.2462050161,
    "elbow_flex":     89.9543738355,
    "wrist_flex":     55.1185398916,
    "wrist_roll":      0.0,
    "gripper":         0.0,
}

desired_zero = {
    "shoulder_pan":  0.0,
    "shoulder_lift": 0.0,
    "elbow_flex":    0.0,
    "wrist_flex":    0.0,
    "wrist_roll":    0.0,
    "gripper":       0.0,
}

def lerp_pose(p0: dict, pf: dict, a: float) -> dict:
    out = {}
    for k in pf.keys():
        out[k] = (1 - a) * float(p0[k]) + a * float(pf[k])
    return out

def snapshot_actual_deg(m, d) -> list:
    pos = get_positions_dict(m, d)  # degrees + gripper 0..100
    return [float(pos[j]) for j in JOINTS]

def snapshot_cmd_deg(cmd: dict) -> list:
    return [float(cmd[j]) for j in JOINTS]

def run_segment(m, d, viewer, start_pose: dict, end_pose: dict, duration_s: float,
                log_actual, log_cmd) -> None:
    t0 = time.time()
    while viewer.is_running():
        t = time.time() - t0
        if t >= duration_s:
            break
        a = min(t / duration_s, 1.0)
        cmd = lerp_pose(start_pose, end_pose, a)

        log_cmd.append([float(d.time)] + snapshot_cmd_deg(cmd))
        log_actual.append([float(d.time)] + snapshot_actual_deg(m, d))

        send_position_command(m, d, cmd)
        mujoco.mj_step(m, d)
        viewer.sync()

def run_hold(m, d, viewer, duration_s: float, log_actual, log_cmd) -> None:
    hold_cmd = get_positions_dict(m, d)  # degrees + gripper
    t0 = time.time()
    while viewer.is_running():
        if (time.time() - t0) >= duration_s:
            break

        log_cmd.append([float(d.time)] + snapshot_cmd_deg(hold_cmd))
        log_actual.append([float(d.time)] + snapshot_actual_deg(m, d))

        send_position_command(m, d, hold_cmd)
        mujoco.mj_step(m, d)
        viewer.sync()

def main():
    m = mujoco.MjModel.from_xml_path(MODEL_PATH)
    d = mujoco.MjData(m)

    set_initial_pose(m, d, starting_position)

    log_actual = []
    log_cmd = []

    with mujoco.viewer.launch_passive(m, d) as viewer:
        pose0 = get_positions_dict(m, d)
        run_segment(m, d, viewer, pose0, desired_zero, 2.0, log_actual, log_cmd)
        run_hold(m, d, viewer, 2.0, log_actual, log_cmd)

        pose1 = get_positions_dict(m, d)
        run_segment(m, d, viewer, pose1, starting_position, 2.0, log_actual, log_cmd)
        run_hold(m, d, viewer, 2.0, log_actual, log_cmd)

    with open("best_PD5_actual_deg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"q_{j}_deg" for j in JOINTS])
        w.writerows(log_actual)

    with open("best_PD5_cmd_deg.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s"] + [f"cmd_{j}_deg" for j in JOINTS])
        w.writerows(log_cmd)

    print(f"OK: saved {len(log_actual)} samples to best_PD5_actual_deg.csv and best_PD5_cmd_deg.csv")

if __name__ == "__main__":
    main()
