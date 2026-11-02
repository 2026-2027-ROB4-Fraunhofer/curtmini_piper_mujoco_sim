# Curtmini Piper MuJoCo sim

MuJoCo simulation of the Curt Mini base with the Piper arm, with the same ROS 2 interfaces as
[curtmini_piper_gz_sim](https://github.com/ipa-may/curtmini_piper_gz_sim): topics, frames,
controller names and launch arguments. SLAM, Nav2, MoveIt and teleoperation run unchanged
against either simulator. MuJoCo needs no GPU.

The model is built at every launch from the existing sources, so changes there are picked up
after a restart:

- robot: the `curtmini_piper_gz_sim` xacro (with `config/geometry.yaml` and `arm_model`)
  → URDF → MuJoCo;
- worlds: the `curtmini_piper_gz_sim` SDF worlds and `neo_gz_worlds` models → MuJoCo;
- controllers: `curtmini_piper_gz_sim/config/simulation_controllers.yaml`, unchanged.

ROS control runs through [mujoco_ros2_control](https://github.com/ros-controls/mujoco_ros2_control).

## Install

ROS 2 Jazzy on Ubuntu 24.04. From the workspace root:

```bash
git clone https://github.com/2026-2027-ROB4-Fraunhofer/curtmini_piper_mujoco_sim.git \
  src/curtmini_piper_mujoco_sim
vcs import src < src/curtmini_piper_mujoco_sim/dependencies.repos
```

The Python `mujoco` package must match the MuJoCo library shipped with ROS (3.12.0). Install it
before `rosdep`, which would otherwise install the latest version:

```bash
pip install --user --break-system-packages mujoco==3.12.0 trimesh
rosdep install --from-paths src --ignore-src -y --skip-keys "candle_ros2 agx_arm_moveit agx_arm_ctrl"
colcon build --symlink-install --packages-up-to curtmini_piper_mujoco_sim curtmini_piper_navigation
source install/setup.bash
```

The skipped keys are real-robot drivers that the simulation does not need.
`curtmini_piper_navigation` (SLAM and Nav2) is not a dependency of the simulation, so it is
named explicitly.

## Run

```bash
ros2 launch curtmini_piper_mujoco_sim simulation.launch.py
ros2 launch curtmini_piper_mujoco_sim simulation.launch.py world:=curtmini_piper_map use_apriltags:=true
```

| Argument | Default | Description |
|---|---|---|
| `arm_model` | `piper_x` | `piper_x`, `piper_l` or `piper_h` |
| `world` | `curtmini_piper` | World name from `curtmini_piper_gz_sim` or `neo_gz_worlds`, or a path to an SDF file |
| `gui` | `true` | Open the MuJoCo viewer |
| `paused` | `false` | Start paused |
| `use_rviz` | `true` | RViz with the MoveIt panel |
| `start_joystick` | `false` | Joystick teleoperation |
| `use_hokuyo` | `true` | Hokuyo lidar on `/scan` |
| `use_cameras` | `false` | Camera frames only |
| `use_apriltags` | `false` | AprilTag panels from `apriltag_layout_file` |
| `apriltag_layout_file` | `curtmini_piper_gz_sim/config/apriltags.yaml` | Tag IDs, sizes and poses |
| `initial_state_file` | `curtmini_piper_gz_sim/config/initial_state.yaml` | Base pose and arm joint positions |

Drive with the keyboard:

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args \
  -r cmd_vel:=/base_controller/cmd_vel -p stamped:=true
```

### SLAM and Nav2

Use the launch files of `curtmini_piper_navigation` as with Gazebo:

```bash
ros2 launch curtmini_piper_navigation mapping.launch.py
ros2 run nav2_map_server map_saver_cli -f ~/maps/office --ros-args -p use_sim_time:=true
ros2 launch curtmini_piper_navigation navigation.launch.py map:=$HOME/maps/office.yaml
ros2 launch curtmini_piper_navigation rviz.launch.py
```

Stop mapping before starting Nav2. Set the initial pose (RViz **2D Pose Estimate**) within about
60 s of starting Nav2; otherwise the navigation servers stay inactive and goals are rejected.


## Tools

```bash
ros2 run curtmini_piper_mujoco_sim build_model --world curtmini_piper_map --view
colcon test --packages-select curtmini_piper_mujoco_sim && colcon test-result --verbose
```

`build_model` builds the model without ROS and opens it in the MuJoCo viewer. Generated models
and converted meshes are cached in `~/.cache/curtmini_piper_mujoco_sim`; delete it to rebuild.

## Package layout

| Path | Content |
|---|---|
| `launch/simulation.launch.py` | Builds the model and starts the simulation, controllers, MoveIt and RViz |
| `curtmini_piper_mujoco_sim/build.py` | Build pipeline: xacro → URDF → MuJoCo scene and robot description |
| `curtmini_piper_mujoco_sim/robot_mjcf.py` | Robot: visuals, actuators, gripper mimic joints, IMU, roller wheels |
| `curtmini_piper_mujoco_sim/meshes.py` | DAE and large STL meshes → OBJ, textures → PNG |
| `curtmini_piper_mujoco_sim/ros2_control.py` | MuJoCo hardware block for `ros2_control` |
| `curtmini_piper_mujoco_sim/lidar.py` | Lidar from the Gazebo sensor block |
| `curtmini_piper_mujoco_sim/world_mjcf.py` | SDF world → MuJoCo |
| `curtmini_piper_mujoco_sim/apriltags.py` | AprilTag panels |
| `scripts/scan_relay` | Raw scan → `/scan` (+inf for no return, range noise) |
| `scripts/build_model` | Build and view the model without ROS |
| `config/mujoco.yaml` | MuJoCo settings: physics, friction, actuator gains, wheel rollers, IMU noise |
| `config/extra_controllers.yaml` | IMU broadcaster and the base odometry correction |
