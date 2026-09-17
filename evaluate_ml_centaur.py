import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import time

# 1. The Physics Environment (Must match exactly)
XML = """
<mujoco model="centaur_wbc">
    <compiler angle="radian" coordinate="local"/>
    <option timestep="0.005" gravity="0 0 -9.81"/>

    <default>
        <joint damping="2.0" armature="0.02"/>
    </default>

    <asset>
        <texture type="2d" name="grid" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="512" height="512"/>
        <material name="mat_floor" texture="grid" texrepeat="5 5" texuniform="true" reflectance="0.2"/>
        <material name="mat_body" rgba="0.2 0.2 0.2 1"/>
        <material name="mat_leg" rgba="0.6 0.6 0.6 1"/>
        <material name="mat_arm" rgba="0.9 0.4 0.1 1"/>
        <material name="mat_target" rgba="0.9 0.1 0.1 1"/>
    </asset>

    <worldbody>
        <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="true"/>
        <geom name="floor" type="plane" size="5 5 0.1" material="mat_floor"/>

        <geom name="table" type="box" size="0.15 0.15 0.1" pos="0.45 0 0.1" rgba="0.3 0.3 0.3 1"/>
        <body name="target_block" pos="0.4 0 0.22">
            <freejoint/>
            <geom type="box" size="0.03 0.03 0.03" material="mat_target" mass="0.1" friction="1 1 1"/>
        </body>

        <body name="torso" pos="0 0 0.45">
            <freejoint/> 
            <geom type="box" size="0.2 0.1 0.05" material="mat_body" mass="5.0"/>

            <!-- Front Left Leg -->
            <body name="fl_hip" pos="0.15 0.12 0">
                <joint name="fl_hip_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                <body name="fl_calf" pos="0 0 -0.15">
                    <joint name="fl_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                    <geom type="sphere" size="0.02" pos="0 0 -0.15" friction="1.5 1.5 1.5"/>
                </body>
            </body>

            <!-- Front Right Leg -->
            <body name="fr_hip" pos="0.15 -0.12 0">
                <joint name="fr_hip_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                <body name="fr_calf" pos="0 0 -0.15">
                    <joint name="fr_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                    <geom type="sphere" size="0.02" pos="0 0 -0.15" friction="1.5 1.5 1.5"/>
                </body>
            </body>

            <!-- Back Left Leg -->
            <body name="bl_hip" pos="-0.15 0.12 0">
                <joint name="bl_hip_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                <body name="bl_calf" pos="0 0 -0.15">
                    <joint name="bl_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                    <geom type="sphere" size="0.02" pos="0 0 -0.15" friction="1.5 1.5 1.5"/>
                </body>
            </body>

            <!-- Back Right Leg -->
            <body name="br_hip" pos="-0.15 -0.12 0">
                <joint name="br_hip_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                <body name="br_calf" pos="0 0 -0.15">
                    <joint name="br_knee_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                    <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.15" material="mat_leg"/>
                    <geom type="sphere" size="0.02" pos="0 0 -0.15" friction="1.5 1.5 1.5"/>
                </body>
            </body>

            <!-- The Arm -->
            <body name="arm_base" pos="0.1 0 0.05">
                <geom type="cylinder" size="0.04 0.02" material="mat_body"/>
                <body name="arm_shoulder" pos="0 0 0.02">
                    <joint name="arm_j1_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                    <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" material="mat_arm"/>
                    <body name="arm_elbow" pos="0 0 0.15">
                        <joint name="arm_j2_lift" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
                        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.15" material="mat_arm"/>
                        <body name="arm_wrist" pos="0 0 0.15">
                            <joint name="arm_j3_flex" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
                            <geom type="capsule" size="0.015" fromto="0 0 0 0 0 0.1" material="mat_arm"/>
                            <body name="end_effector" pos="0 0 0.1">
                                <geom type="sphere" size="0.02" material="mat_body"/>
                            </body>
                        </body>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>

    <actuator>
        <position name="m_fl_hip" joint="fl_hip_joint" kp="250" ctrllimited="true" ctrlrange="-1.5 1.5"/>
        <position name="m_fl_knee" joint="fl_knee_joint" kp="250" ctrllimited="true" ctrlrange="-2.5 0"/>
        <position name="m_fr_hip" joint="fr_hip_joint" kp="250" ctrllimited="true" ctrlrange="-1.5 1.5"/>
        <position name="m_fr_knee" joint="fr_knee_joint" kp="250" ctrllimited="true" ctrlrange="-2.5 0"/>
        <position name="m_bl_hip" joint="bl_hip_joint" kp="250" ctrllimited="true" ctrlrange="-1.5 1.5"/>
        <position name="m_bl_knee" joint="bl_knee_joint" kp="250" ctrllimited="true" ctrlrange="-2.5 0"/>
        <position name="m_br_hip" joint="br_hip_joint" kp="250" ctrllimited="true" ctrlrange="-1.5 1.5"/>
        <position name="m_br_knee" joint="br_knee_joint" kp="250" ctrllimited="true" ctrlrange="-2.5 0"/>
        <position name="m_arm_j1" joint="arm_j1_pan" kp="80" ctrllimited="true" ctrlrange="-3.14 3.14"/>
        <position name="m_arm_j2" joint="arm_j2_lift" kp="80" ctrllimited="true" ctrlrange="-2.5 2.5"/>
        <position name="m_arm_j3" joint="arm_j3_flex" kp="80" ctrllimited="true" ctrlrange="-2.5 2.5"/>
    </actuator>
</mujoco>
"""


# 2. The PyTorch Architecture (Must match exactly)
class CentaurWBCPolicy(nn.Module):
    def __init__(self, in_dim=21, out_dim=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


def main():
    print("🧠 Loading Trained PyTorch Loco-Manipulation Policy...")
    policy = CentaurWBCPolicy()
    policy.load_state_dict(torch.load("models/wbc_policy.pth"))
    policy.eval()

    print("🤖 Initializing MuJoCo Simulation...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        episode = 1

        while viewer.is_running():
            print(f"\n🎬 Starting Evaluation Episode {episode}")
            mujoco.mj_resetData(model, data)

            # 1. Randomize Target Position (The AI must track this)
            block_y = np.random.uniform(-0.10, 0.10)
            data.qpos[3:6] = [0.4, block_y, 0.22]
            print(f"🎯 Target Coordinates: Y = {block_y:.3f}")

            # 2. Visual Domain Randomization (Sim-to-Real)
            new_friction = np.random.uniform(0.1, 1.5)
            new_mass = np.random.uniform(4.0, 7.0)
            model.geom_friction[floor_id][0] = new_friction
            model.body_mass[torso_id] = new_mass

            friction_ratio = (new_friction - 0.1) / 1.4
            color_ice = np.array([0.6, 0.8, 1.0, 1.0])
            color_asphalt = np.array([0.2, 0.2, 0.2, 1.0])
            model.geom_rgba[floor_id] = (1 - friction_ratio) * color_ice + friction_ratio * color_asphalt

            print(f"🌍 Floor Friction: {new_friction:.2f} | ⚖️ Payload Mass: {new_mass:.2f} kg")

            # 3. Closed-Loop PyTorch Inference
            for step in range(250):  # Run for a few seconds per episode
                # Extract state exactly as generated
                block_pos = data.qpos[3:6].copy()
                joint_pos = data.qpos[7:].copy()
                state = np.concatenate([block_pos, joint_pos])

                # Convert to PyTorch tensor and infer
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    action = policy(state_tensor).squeeze(0).numpy()

                # Apply predicted whole-body control
                data.ctrl[:] = action
                mujoco.mj_step(model, data)

                viewer.sync()
                time.sleep(0.01)

            episode += 1


if __name__ == "__main__":
    main()