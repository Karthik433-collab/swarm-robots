import mujoco
import mujoco.viewer
import numpy as np

# Load the MuJoCo model
xml_path = r"C:\Users\sai42\OneDrive\Documents\robotics\Mujoco\swarm.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Initial positions
east_wall_pos = [2.7, 0.0]
for i in range(10, 13):  # Robots 11-13 (indices 10-12)
    robot_name = f"robot{i+1}"
    x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_x")
    y_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_y")
    data.qpos[x_joint] = east_wall_pos[0]
    data.qpos[y_joint] = east_wall_pos[1]
wall_north_pos = [1.5, 3.0]
for i in range(3):  # Robots 1-3
    robot_name = f"robot{i+1}"
    x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_x")
    y_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_y")
    data.qpos[x_joint] = wall_north_pos[0]
    data.qpos[y_joint] = wall_north_pos[1] - 0.3
robot8_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot8_x")
robot8_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot8_y")
data.qpos[robot8_x] = east_wall_pos[0]
data.qpos[robot8_y] = east_wall_pos[1]
west_wall_pos = [1.5, 3.0]
robot9_x = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot9_x")
robot9_y = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "robot9_y")
data.qpos[robot9_x] = west_wall_pos[0]
data.qpos[robot9_y] = west_wall_pos[1]
mujoco.mj_forward(model, data)

# PID and swarm parameters
Kp = 10.0  # Proportional gain
Kd = 0.1   # Derivative gain
prev_errors = np.zeros((12, 2))  # 12 robots
reached_goal = np.zeros(12, dtype=bool)  # 12 robots

# Swarm behavior parameters
cohesion_gain = 0.5
alignment_gain = 0.3
separation_gain = 1.0
goal_gain = 1.5
neighbor_radius = 1.0
min_separation = 0.3

# Get body and geom IDs
goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
robot_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot{i+1}") for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]  # 12 robots: 1-6, 8-13
robot_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot{i+1}") for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]]  # 12 robots
wall_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in 
                 ["wall_north", "wall_south", "wall_west", "wall_east", 
                  "small_room_wall2", "small_room_wall3"]]
central_obstacle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "central_obstacle")
obstacle_geom_id = model.body_geomadr[central_obstacle_body_id]

arena_x_min, arena_x_max = -4.5, 4.5
arena_y_min, arena_y_max = -2.8, 2.8

def choose_random_target():
    return np.array([
        np.random.uniform(arena_x_min, arena_x_max),
        np.random.uniform(arena_y_min, arena_y_max)
    ])

# Compute swarm forces for a robot
def compute_swarm_forces(robot_idx, positions, velocities):
    robot_pos = positions[robot_idx]
    cohesion = np.zeros(2)
    alignment = np.zeros(2)
    separation = np.zeros(2)
    neighbor_count = 0
    
    for j in range(12):
        if j == robot_idx or (6 <= j <= 8):  # Adjusted: wandering robots are now 8-10 (indices 6-8)
            continue
        diff = positions[j] - robot_pos
        dist = np.linalg.norm(diff)
        
        if dist < neighbor_radius:
            cohesion += positions[j]
            alignment += velocities[j]
            neighbor_count += 1
        
        if dist < min_separation and dist > 0:
            separation -= (diff / (dist + 1e-6)) * (min_separation - dist)
    
    if neighbor_count > 0:
        cohesion = (cohesion / neighbor_count) - robot_pos
        alignment = alignment / neighbor_count
    
    return cohesion, alignment, separation

with mujoco.viewer.launch_passive(model, data) as viewer:
    goal_pos = data.xpos[goal_body_id][:2].copy()
    
    # Initialize targets
    current_targets = []
    for i in range(12):
        if i < 6 or 9 <= i < 12:  # Goal-seekers: 1-6, 11-13 (indices 0-5, 9-11)
            current_targets.append(goal_pos)
        else:  # Wanderers: 8-10 (indices 6-8)
            current_targets.append(choose_random_target())
    
    prev_positions = np.zeros((12, 2))
    
    while viewer.is_running():
        robot_positions = np.array([data.xpos[id][:2] for id in robot_body_ids])
        velocities = (robot_positions - prev_positions) / model.opt.timestep
        
        # Update targets for wandering robots (8-10, indices 6-8)
        for i in range(6, 9):
            if np.linalg.norm(robot_positions[i] - current_targets[i]) < 0.2:
                current_targets[i] = choose_random_target()
        
        # Collision detection
        has_collision = np.zeros(12, dtype=bool)
        for contact in data.contact[:data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            for j in range(12):  # Fixed to 12
                if (geom1 == robot_geom_ids[j] and geom2 in wall_geom_ids + [obstacle_geom_id]) or \
                   (geom2 == robot_geom_ids[j] and geom1 in wall_geom_ids + [obstacle_geom_id]):
                    has_collision[j] = True
        
        # Control logic
        for i in range(12):
            if has_collision[i]:
                data.ctrl[i*2:i*2+2] = [0, 0]
                continue
                
            robot_pos = robot_positions[i]
            target = current_targets[i]
            error = target - robot_pos
            d_error = error - prev_errors[i]
            
            if i < 6 or 9 <= i < 12:  # Goal-seekers: 1-6, 11-13 (indices 0-5, 9-11)
                cohesion, alignment, separation = compute_swarm_forces(i, robot_positions, velocities)
                control = (Kp * error + Kd * d_error + 
                          cohesion_gain * cohesion + 
                          alignment_gain * alignment + 
                          separation_gain * separation + 
                          goal_gain * (goal_pos - robot_pos))
                
                if np.linalg.norm(robot_pos - goal_pos) < 0.2:
                    control = [0, 0]
                    reached_goal[i] = True
            else:  # Wanderers: 8-10 (indices 6-8)
                control = Kp * error + Kd * d_error
                goal_zone_center = data.xpos[goal_body_id][:2]
                goal_radius = 1.0
                to_goal = robot_pos - goal_zone_center
                distance = np.linalg.norm(to_goal)
                if distance < goal_radius:
                    repulsion_gain = 8.0
                    safe_direction = to_goal / (distance + 1e-6)
                    control += repulsion_gain * (goal_radius - distance) * safe_direction
            
            data.ctrl[i*2:i*2+2] = control
            prev_errors[i] = error
        
        mujoco.mj_step(model, data)
        prev_positions = robot_positions.copy()
        viewer.sync()
