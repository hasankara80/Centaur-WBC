# Centaur-WBC: Whole-Body Loco-Manipulation & Domain Randomization
An end-to-end Imitation Learning pipeline demonstrating Whole-Body Control (WBC) for an arm-equipped quadruped. Built in MuJoCo and PyTorch, this project coordinates base, torso, and manipulator motion as a single dynamic system to execute stable reaching tasks across unstructured physical parameters.

## 🎬 Sim-to-Real Deployment Demo


<div align="center">
  <video src="https://github.com/user-attachments/assets/459d3ac7-b3bb-4ba7-a425-b3ceda3856c5" width="80%" autoplay loop muted playsinline></video>
</div>


## 🧠 System Architecture
This repository is structured as a 3-stage Imitation Learning pipeline, transitioning from hardcoded kinematic stabilization to closed-loop neural policy inference.

**1. Custom Kinematic Chain & Physics (view_centaur.py)**
Embodiment: Engineered a 15-DOF robot (12-DOF quadruped base + 3-DOF manipulator) using MuJoCo XML.

Physics Stabilization: Tuned joint damping, armature inertia, and high-stiffness position actuators to prevent NaN simulation collapses during edge-case contact dynamics and high-extension payload shifts.

Whole-Body Coordination: Implemented a dynamic reaching controller that explicitly drops the front suspension and extends the rear legs to shift the center of mass, preventing tip-over during grasping.

**2. Expert Data Engine (generate_loco_data.py)**
Programmatic Generation: Programmatically spawns spatial targets and computes the synchronized 11-motor action sequences required to reach the target while maintaining base stability.

Dataset: Generated 30,000 steps of expert state-action trajectories for training.

**3. Imitation Learning Policy (train_wbc_policy.py)**
Architecture: Trained a PyTorch Multi-Layer Perceptron (MLP) mapping the spatial target coordinates and 11-DOF joint states directly to continuous motor action setpoints.

Performance: Converged to an MSE loss of 0.000010, mathematically solving the relationship between target coordinates and the whole-body balancing act.

**4. Sim-to-Real Domain Randomization (evaluate_ml_centaur.py)**
Covariate Shifts: Evaluates the PyTorch policy in a continuous, closed-loop MuJoCo simulation.

Dynamic Perturbations: Randomizes floor friction (0.1 to 1.5) and payload mass (4.0 kg to 7.0 kg) every episode.

Boundary Testing: By pushing the payload up to 7.0 kg, the simulation intentionally tests the physical boundaries of the learned policy, demonstrating authentic sim-to-real kinematic sag and dynamic recovery.

## 🚀 Quickstart Guide
**1. Generate the Expert Dataset:**

```Bash
python generate_loco_data.py
```
**2. Train the PyTorch Policy:**

```Bash
python train_wbc_policy.py
```

**3. Run the Randomized Deployment Simulation:**

```Bash
mjpython evaluate_ml_centaur.py
```

## 🛠️ Technical Stack
Simulation Physics: MuJoCo (mujoco)

Machine Learning: PyTorch (torch.nn, torch.optim)

Data Processing: NumPy
