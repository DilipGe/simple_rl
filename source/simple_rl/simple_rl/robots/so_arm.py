import math
import os
from pathlib import Path


import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg


# Dynamically locate the robot.usd file relative to this script
ROBOT_DIR = Path(__file__).parent
USD_PATH = os.path.join(ROBOT_DIR, "robot.usd")


SO_ARM101_CFG = ArticulationCfg(
    # 1. Define where the robot will live in the simulation stage
    # The regex {ENV_REGEX_NS} handles multi-environment parallel training seamlessly
    prim_path="{ENV_REGEX_NS}/Robot",
    
    # 2. Spawn the USD asset into the simulator
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        # Rigid body and articulation properties
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        rot=(1.0, 0.0, 0.0, 0.0),
        # NOTE: joint_pos is in radians, not degrees.
        joint_pos={
            "joint_1": 0.0,
            "joint2": math.radians(90.0),
            "joint3": math.radians(-90.0),
            "joint4": 0.0,
            "joint5": -0.0,
            "joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_1", "joint2", "joint3", "joint4", "joint5"],
            effort_limit_sim=15.9,
            velocity_limit_sim=13.8,
            stiffness={
                "joint_1": 200.0,
                "joint2": 170.0,
                "joint3": 120.0,
                "joint4": 80.0,
                "joint5": 50.0,
            },
            damping={
                "joint_1": 80.0,
                "joint2": 65.0,
                "joint3": 45.0,
                "joint4": 30.0,
                "joint5": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit_sim=2.5,
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)