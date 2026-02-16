import mujoco
import mujoco.viewer
import csv

from so101_mujoco_utils2 import (
    set_initial_pose,
    move_to_pose,
    hold_position,
    get_qpos_adr
)

MODEL_PATH = "model/scene_urdf.xml"

m = mujoco.MjModel.from_xml_path(MODEL_PATH)
d = mujoco.MjData(m)

joint_names = ["shoulder_pan","shoulder_lift","elbow_flex","wrist_flex","wrist_roll"]
qpos_adr = get_qpos_adr(m, joint_names)

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

print(type(starting_position), starting_position)

set_initial_pose(m, d, starting_position)

log = []  # rows: [time, q1, q2, q3, q4, q5]

with mujoco.viewer.launch_passive(m, d) as viewer:
    move_to_pose(m, d, viewer, desired_zero, duration=2.0, realtime=True,
                 log=log, qpos_adr=qpos_adr, joint_names=joint_names)

    hold_position(m, d, viewer, duration=2.0, realtime=True,
                  log=log, qpos_adr=qpos_adr, joint_names=joint_names)

    move_to_pose(m, d, viewer, starting_position, duration=2.0, realtime=True,
                 log=log, qpos_adr=qpos_adr, joint_names=joint_names)

    hold_position(m, d, viewer, duration=2.0, realtime=True,
                  log=log, qpos_adr=qpos_adr, joint_names=joint_names)

# guardar CSV
with open("trajectory_p2.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time"] + joint_names)
    w.writerows(log)

print("Saved trajectory_p2.csv with", len(log), "samples")
