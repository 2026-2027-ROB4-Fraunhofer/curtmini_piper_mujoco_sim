"""Tests for the generated robot model."""

from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import mujoco
import numpy as np
import pytest
import yaml

from curtmini_piper_mujoco_sim.build import build, gz_sim_share

ARM_JOINTS = [f'piper_joint{i}' for i in range(1, 7)]
WHEEL_JOINTS = ['front_left_motor', 'back_left_motor', 'front_right_motor', 'back_right_motor']


@pytest.fixture(scope='module')
def result(tmp_path_factory):
    return build(arm_model='piper_x', output_dir=tmp_path_factory.mktemp('model'))


def test_actuators_match_ros2_control_joints(result):
    model = result['model']
    description = ET.fromstring(result['robot_description'])
    blocks = description.findall('ros2_control')
    assert len(blocks) == 1
    controlled = {joint.get('name') for joint in blocks[0].findall('joint')}
    actuators = {model.actuator(i).name for i in range(model.nu)}
    assert controlled == actuators == set(ARM_JOINTS + WHEEL_JOINTS + ['piper_gripper'])


def test_imu_sensors_present(result):
    model = result['model']
    for suffix in ('_quat', '_gyro', '_accel'):
        assert model.sensor('imu_sensor' + suffix) is not None


def _hold_arm(model, data):
    """Command position actuators to hold their current joint positions."""
    for i in range(model.nu):
        if model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE \
                and model.actuator_biasprm[i, 1] != 0:
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]


def _base_controller_parameters():
    """base_controller parameters in the order the launch file loads them."""
    parameters = {}
    for path in (gz_sim_share() / 'config' / 'simulation_controllers.yaml',
                 Path(get_package_share_directory('curtmini_piper_mujoco_sim'))
                 / 'config' / 'extra_controllers.yaml'):
        with open(path) as stream:
            parameters.update(yaml.safe_load(stream)['base_controller']['ros__parameters'])
    return parameters


def _drive(model, linear, angular, seconds=4.0):
    """Drive like diff_drive_controller; return the true and the odometry (speed, yaw rate)."""
    base = _base_controller_parameters()
    separation = base['wheel_separation'] * base['wheel_separation_multiplier']
    radius = base['wheel_radius']
    wheels = {side: [model.joint(name).dofadr[0] for name in base[f'{side}_wheel_names']]
              for side in ('left', 'right')}
    data = mujoco.MjData(model)
    _hold_arm(model, data)
    for side, sign in (('left', -1), ('right', 1)):
        for wheel in base[f'{side}_wheel_names']:
            data.ctrl[model.actuator(wheel).id] = \
                (linear + sign * angular * separation / 2) / radius
    mujoco.mj_step(model, data, nstep=int(1.0 / model.opt.timestep))  # accelerate
    start_xy, start_time, yaws, odometry = data.qpos[:2].copy(), data.time, [], []
    while data.time - start_time < seconds - 1.0:
        mujoco.mj_step(model, data)
        w, x, y, z = data.qpos[3:7]
        yaws.append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        left, right = (np.mean(data.qvel[wheels[side]]) * radius for side in ('left', 'right'))
        odometry.append(((left + right) / 2, (right - left) / separation))
    elapsed = data.time - start_time
    true = (np.linalg.norm(data.qpos[:2] - start_xy) / elapsed,
            np.sum(np.diff(np.unwrap(yaws))) / elapsed)
    return true, tuple(np.mean(odometry, axis=0))


@pytest.mark.parametrize('linear, angular', [
    (0.5, 0.0), (0.0, 0.25), (0.0, 1.0), (0.0, 2.0), (0.3, 0.5),
])
def test_odometry_matches_motion(result, linear, angular):
    """Wheel odometry follows the true motion, and the base turns at the commanded rate."""
    (speed, yaw_rate), (odom_speed, odom_yaw_rate) = _drive(result['model'], linear, angular)
    if linear:
        assert odom_speed == pytest.approx(speed, rel=0.05)
    assert odom_yaw_rate == pytest.approx(yaw_rate, rel=0.05, abs=0.01)
    assert yaw_rate == pytest.approx(angular, rel=0.1, abs=0.02)


def test_robot_settles_on_ground(result):
    model = result['model']
    data = mujoco.MjData(model)
    _hold_arm(model, data)
    start = data.qpos[:2].copy()
    mujoco.mj_step(model, data, nstep=int(2.0 / model.opt.timestep))
    assert np.all(np.isfinite(data.qpos))
    assert np.linalg.norm(data.qpos[:2] - start) < 0.02, 'base drifted while standing still'
    assert 0.0 < data.qpos[2] < 0.3, f'base height {data.qpos[2]:.3f} m is not on the ground'


def test_lidar_matches_gazebo_block_and_sees_a_wall(result):
    """The Hokuyo has 1081 rays and detects a wall 2 m ahead."""
    lidar, = result['lidars']
    assert (lidar['link'], lidar['topic'], lidar['samples']) == ('hokuyo_link', '/scan', 1081)

    spec = mujoco.MjSpec.from_file(result['mujoco_model'])
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scan_origin = data.site_xpos[model.site(lidar['name'] + '_origin').id]
    wall_x = scan_origin[0] + 2.0
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, pos=[wall_x + 0.05, scan_origin[1], 1],
                            size=[0.05, 0.5, 1], contype=0, conaffinity=0)
    model = spec.compile()
    data = mujoco.MjData(model)
    _hold_arm(model, data)
    mujoco.mj_step(model, data, nstep=int(0.1 / model.opt.timestep))
    sensor = model.sensor(lidar['name'])
    ranges = data.sensordata[sensor.adr[0]:sensor.adr[0] + sensor.dim[0]]
    assert len(ranges) == 1081
    centre = ranges[540]  # ray at angle 0, straight ahead
    assert centre == pytest.approx(2.0, abs=0.03), 'housing or robot blocks the scan'
    assert ranges[0] == -1.0, 'ray at -135 deg should see nothing in the empty world'
