"""Launch the Curt Mini + Piper MuJoCo simulation.

Provides the same launch arguments and ROS interfaces as curtmini_piper_gz_sim.
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap

from curtmini_piper_gz_sim.cameras import camera_mount_actions
from curtmini_piper_gz_sim.gazebo import resolve_world
from curtmini_piper_gz_sim.moveit import build_moveit_config, rviz_action
from curtmini_piper_gz_sim.robot import (
    controller_spawner_action,
    robot_mappings,
    robot_state_publisher_action,
)
from curtmini_piper_mujoco_sim.build import build
from curtmini_piper_mujoco_sim.lidar import raw_topic

# Node created by mujoco_ros2_control; provides the set_pause and reset_world services.
MUJOCO_NODE = 'mujoco_ros2_control_node'


def _as_bool(context, name):
    return LaunchConfiguration(name).perform(context).lower() == 'true'


def _launch_setup(context):
    sim_share = Path(get_package_share_directory('curtmini_piper_gz_sim'))
    own_share = Path(get_package_share_directory('curtmini_piper_mujoco_sim'))
    arm_model = LaunchConfiguration('arm_model').perform(context)
    world = resolve_world(LaunchConfiguration('world').perform(context), sim_share,
                          Path(get_package_share_directory('neo_gz_worlds')))

    model = build(
        arm_model=arm_model,
        use_hokuyo=_as_bool(context, 'use_hokuyo'),
        use_cameras=_as_bool(context, 'use_cameras'),
        initial_state_file=LaunchConfiguration('initial_state_file').perform(context),
        headless=not _as_bool(context, 'gui'),
        world_file=world,
        apriltag_layout_file=(LaunchConfiguration('apriltag_layout_file').perform(context)
                              if _as_bool(context, 'use_apriltags') else None),
    )
    mappings = robot_mappings(sim_share, arm_model, _as_bool(context, 'use_hokuyo'),
                              _as_bool(context, 'use_cameras'))
    moveit_config = build_moveit_config(mappings)

    actions = [
        robot_state_publisher_action(model['robot_description']),
        Node(
            package='mujoco_ros2_control',
            executable='ros2_control_node',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                str(sim_share / 'config' / 'simulation_controllers.yaml'),
                str(own_share / 'config' / 'extra_controllers.yaml'),
                model['plugin_parameters'],
            ],
            remappings=[('~/robot_description', '/robot_description'),
                        ('/imu_sensor_broadcaster/imu', '/imu/data')],
            on_exit=Shutdown(),
        ),
        controller_spawner_action(
            'joint_state_broadcaster',
            'base_controller',
            'arm_controller',
            'gripper_controller',
            'imu_sensor_broadcaster',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(Path(get_package_share_directory('curtmini_piper_moveit_config'))
                    / 'launch' / 'move_group.launch.py')
            ),
            launch_arguments={
                'controller_mode': 'sim',
                'arm_model': arm_model,
                'use_sim_time': 'true',
                'use_hokuyo': str(_as_bool(context, 'use_hokuyo')).lower(),
                'use_sensors_3d': 'false',
            }.items(),
        ),
    ]

    for lidar in model['lidars']:
        actions.append(Node(
            package='curtmini_piper_mujoco_sim',
            executable='scan_relay',
            name=f'{lidar["name"]}_relay',
            output='screen',
            parameters=[{'use_sim_time': True,
                         'input_topic': raw_topic(lidar),
                         'output_topic': lidar['topic'],
                         'noise_stddev': lidar['noise']}],
        ))

    if _as_bool(context, 'paused'):
        actions.append(ExecuteProcess(
            cmd=['ros2', 'service', 'call', f'/{MUJOCO_NODE}/set_pause',
                 'mujoco_ros2_control_msgs/srv/SetPause', '{paused: true}'],
            output='screen',
        ))

    if _as_bool(context, 'use_cameras'):
        description_share = Path(get_package_share_directory('curtmini_piper_description'))
        actions.extend(camera_mount_actions(description_share))

    if _as_bool(context, 'start_joystick'):
        actions.append(GroupAction([
            SetRemap(src='/joy_teleop/cmd_vel', dst='/base_controller/cmd_vel'),
            IncludeLaunchDescription(PythonLaunchDescriptionSource(
                str(Path(get_package_share_directory('curt_mini_teleop'))
                    / 'launch' / 'joystick.launch.py'))),
        ]))

    if _as_bool(context, 'use_rviz'):
        actions.append(rviz_action(moveit_config))

    return actions


def generate_launch_description():
    sim_share = Path(get_package_share_directory('curtmini_piper_gz_sim'))
    return LaunchDescription([
        DeclareLaunchArgument(
            'arm_model', default_value='piper_x', choices=['piper_x', 'piper_l', 'piper_h'],
            description='Piper arm and AGX gripper model to simulate.'),
        DeclareLaunchArgument(
            'world', default_value='curtmini_piper',
            description=(
                'World name from curtmini_piper_gz_sim or neo_gz_worlds, with optional .sdf '
                'extension, or a path to a Gazebo world file.')),
        DeclareLaunchArgument(
            'gui', default_value='true', choices=['true', 'false'],
            description='Open the MuJoCo viewer window.'),
        DeclareLaunchArgument(
            'paused', default_value='false', choices=['true', 'false'],
            description='Start the simulation paused.'),
        DeclareLaunchArgument(
            'use_rviz', default_value='true', choices=['true', 'false'],
            description='Start RViz with the MoveIt panel.'),
        DeclareLaunchArgument(
            'start_joystick', default_value='false', choices=['true', 'false'],
            description='Start Curt Mini joystick teleoperation.'),
        DeclareLaunchArgument(
            'use_hokuyo', default_value='true', choices=['true', 'false'],
            description='Add the Hokuyo UTM-30LX-EW simulation.'),
        DeclareLaunchArgument(
            'use_cameras', default_value='false', choices=['true', 'false'],
            description='Add the two D435i cameras.'),
        DeclareLaunchArgument(
            'use_apriltags', default_value='false', choices=['true', 'false'],
            description='Spawn the AprilTag exercise panels.'),
        DeclareLaunchArgument(
            'apriltag_layout_file', default_value=str(sim_share / 'config' / 'apriltags.yaml'),
            description='YAML file with AprilTag IDs, sizes, and world poses.'),
        DeclareLaunchArgument(
            'initial_state_file',
            default_value=str(sim_share / 'config' / 'initial_state.yaml'),
            description='YAML file with the base pose and six Piper joint positions at startup.'),
        OpaqueFunction(function=_launch_setup),
    ])
