import mujoco
import numpy as np
import os

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


def main():
    print("🤖 Initializing Centaur Headless Data Engine...")
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)

    expert_observations = []
    expert_actions = []

    total_episodes = 300
    print(f"🚀 Generating {total_episodes} expert loco-manipulation trajectories...")

    pose_stand = np.array([0.5, -1.0, 0.5, -1.0, 0.5, -1.0, 0.5, -1.0, 0.0, 0.5, -0.5])

    for episode in range(total_episodes):
        mujoco.mj_resetData(model, data)

        # Randomize block position on the table (Y-axis variation)
        block_y = np.random.uniform(-0.10, 0.10)
        data.qpos[3:6] = [0.4, block_y, 0.22]  # Set freejoint of target block

        # Calculate base pan angle to face the block dynamically
        base_pan = np.arctan2(block_y, 0.4)

        pose_reach = np.array([
            0.7, -1.3,
            0.7, -1.3,
            0.4, -0.5,
            0.4, -0.5,
            base_pan, 2.3, 0.05
        ])

        # Simulate 100 steps per episode
        for step in range(100):
            # Smooth transition from standing to reaching
            if step < 40:
                alpha = step / 40.0
                action = (1 - alpha) * pose_stand + alpha * pose_reach
            else:
                action = pose_reach

            data.ctrl[:] = action
            mujoco.mj_step(model, data)

            # State vector: [Block X, Block Y, Block Z, 11 Joint Positions]
            block_pos = data.qpos[3:6].copy()
            joint_pos = data.qpos[7:].copy()  # Includes floating base orientation + all 11 joints
            state = np.concatenate([block_pos, joint_pos])

            expert_observations.append(state)
            expert_actions.append(action)

    os.makedirs("dataset", exist_ok=True)
    np.savez(
        "dataset/centaur_expert_data.npz",
        states=np.array(expert_observations),
        actions=np.array(expert_actions)
    )
    print(f"✅ Centaur Dataset Saved Successfully! ({len(expert_observations)} total steps)")


if __name__ == "__main__":
    main()