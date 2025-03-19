import mujoco
import mujoco.viewer
import numpy as np

xml_path = r"C:\Users\sai42\OneDrive\Documents\robotics\Mujoco\swarm.xml"  # Full path

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# PID Control Gains
Kp = 5.0
Kd = 0.1
prev_errors = np.zeros((7, 2))  # Changed from 3 to 7
reached_goal = np.zeros(7, dtype=bool)  # Track which robots reached the goal

# Get goal ID
goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")

# Robot body IDs (added 4 more robots)
robot_body_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot1"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot2"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot3"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot4"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot5"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot6"),
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot7"),
]

# Get geom IDs for robots and walls
robot_geom_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot{i+1}") for i in range(7)
]
wall_geom_ids = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    for name in ["wall_north", "wall_south", "wall_west", "wall_east", "dividing_wall_right", "wall_inner1", "wall_inner2"]
]

def get_goal_position():
    return data.xpos[goal_body_id][:2]

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        goal_x, goal_y = get_goal_position()
        threshold = 0.2  # Increased threshold to ensure robots stop near the goal

        robot_positions = np.array([data.xpos[id][:2] for id in robot_body_ids])

        # Check for contacts (walls)
        has_contact = np.zeros(6, dtype=bool)

        # Iterate through all contacts in data.contact
        for i in range(data.ncon):
            contact = data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if one of the geoms is a robot and the other is a wall
            for j in range(7):
                robot_geom = robot_geom_ids[j]
                if (geom1 == robot_geom and geom2 in wall_geom_ids) or (geom2 == robot_geom and geom1 in wall_geom_ids):
                    has_contact[j] = True
                    break

        # Control loop for each robot
        for i in range(7):
            robot_x, robot_y = robot_positions[i]
            error_x = goal_x - robot_x
            error_y = goal_y - robot_y

            d_error_x = error_x - prev_errors[i][0]
            d_error_y = error_y - prev_errors[i][1]
            prev_errors[i] = [error_x, error_y]

            # Removed speed reduction on wall contact
            move_x = (Kp * error_x) + (Kd * d_error_x)
            move_y = (Kp * error_y) + (Kd * d_error_y)

            # Stop when close to goal
            if not reached_goal[i] and (abs(error_x) < threshold and abs(error_y) < threshold):
                move_x = 0
                move_y = 0
                reached_goal[i] = True
            elif reached_goal[i]:
                move_x = 0
                move_y = 0

            data.ctrl[i * 2] = move_x
            data.ctrl[i * 2 + 1] = move_y

        viewer.sync() 