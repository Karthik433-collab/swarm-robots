import mujoco
import mujoco.viewer
import numpy as np

# Load the MuJoCo model
xml_path = r"C:\Users\sai42\OneDrive\Documents\robotics\Mujoco\swarm.xml"
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# ====== ONLY ADDED SECTION ======
# Set robots 1-3 to start together at the north wall position
# Set robots 11-13 at east wall together
east_wall_pos = [2.7, 0.0]  # 0.3m inside wall_east (3.0 - 0.3)
for i in range(11, 14):
    robot_name = f"robot{i}"
    x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_x")
    y_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_y")
    data.qpos[x_joint] = east_wall_pos[0]
    data.qpos[y_joint] = east_wall_pos[1]

wall_north_pos = [1.5, 3.0]
for i in range(3):
    robot_name = f"robot{i+1}"
    x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_x")
    y_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{robot_name}_y")
    data.qpos[x_joint] = wall_north_pos[0]
    data.qpos[y_joint] = wall_north_pos[1] - 0.3
mujoco.mj_forward(model, data)  # Update physics
# ====== END OF ADDED SECTION ======

# Original code continues unchanged below
# PID controller parameters
Kp = 5.0  # Proportional gain
Kd = 0.1  # Derivative gain
prev_errors = np.zeros((13, 2))
reached_goal = np.zeros(13, dtype=bool)

# Get body and geom IDs from the model
goal_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "goal")
robot_body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot{i+1}") for i in range(13)]
robot_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot{i+1}") for i in range(13)]
wall_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in 
                 ["wall_north", "wall_south", "wall_west", "wall_east", 
                  "small_room_wall2", "small_room_wall3"]]
central_obstacle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "central_obstacle")
obstacle_geom_id = model.body_geomadr[central_obstacle_body_id]

# Define central area bounds for robots 8 and 9
central_x_min = -0.5
central_x_max = 0.5
central_y_min = -0.5
central_y_max = 0.5


arena_x_min, arena_x_max = -4.5, 4.5
arena_y_min, arena_y_max = -2.8, 2.8

def choose_random_target():
    return np.array([
        np.random.uniform(arena_x_min, arena_x_max),
        np.random.uniform(arena_y_min, arena_y_max)
    ])

with mujoco.viewer.launch_passive(model, data) as viewer:
    goal_pos = data.xpos[goal_body_id][:2].copy()
    
    # Initialize targets: 
    # 1-6 and 11-13 -> goal, 7-10 -> random
    current_targets = []
    for i in range(13):
        if i < 6 or 10 <= i < 13:  # Robots 1-6 and 11-13
            current_targets.append(goal_pos)
        else:  # Robots 7-10
            current_targets.append(choose_random_target())
    
    while viewer.is_running():
        robot_positions = np.array([data.xpos[id][:2] for id in robot_body_ids])
        
        # Update targets only for wandering robots (7-10)
        for i in range(6, 10):
            if np.linalg.norm(robot_positions[i] - current_targets[i]) < 0.2:
                current_targets[i] = choose_random_target()
        
        # Collision detection (unchanged)
        has_collision = np.zeros(13, dtype=bool)
        for contact in data.contact[:data.ncon]:
            geom1, geom2 = contact.geom1, contact.geom2
            for j in range(13):
                if (geom1 == robot_geom_ids[j] and geom2 in wall_geom_ids + [obstacle_geom_id]) or \
                   (geom2 == robot_geom_ids[j] and geom1 in wall_geom_ids + [obstacle_geom_id]):
                    has_collision[j] = True
        
        # Control logic
        for i in range(13):
            if has_collision[i]:
                data.ctrl[i*2:i*2+2] = [0, 0]
                continue
                
            target = current_targets[i]
            robot_pos = robot_positions[i]
            error = target - robot_pos
            d_error = error - prev_errors[i]
            
            # Goal-seeking robots (1-6 and 11-13)
            if i < 6 or 10 <= i < 13:
                control = Kp * error + Kd * d_error
                if np.linalg.norm(error) < 0.2:
                    control = [0, 0]
                    reached_goal[i] = True
            # Wandering robots (7-10)
            else:  
                control = Kp * error + Kd * d_error  # Same PID but keeps moving
                
            data.ctrl[i*2:i*2+2] = control
            prev_errors[i] = error
        
        mujoco.mj_step(model, data)
        viewer.sync()
