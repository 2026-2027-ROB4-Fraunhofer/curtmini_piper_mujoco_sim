"""Tests for the SDF world and AprilTag conversion."""

import mujoco
import numpy as np
import pytest
import yaml

from curtmini_piper_mujoco_sim.build import build, gz_sim_share


def _hold_arm(model, data):
    for i in range(model.nu):
        if model.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE \
                and model.actuator_biasprm[i, 1] != 0:
            data.ctrl[i] = data.qpos[model.jnt_qposadr[model.actuator_trnid[i, 0]]]


@pytest.fixture(scope='module')
def office(tmp_path_factory):
    return build(world_file=gz_sim_share() / 'worlds' / 'curtmini_piper_map.sdf',
                 apriltag_layout_file=gz_sim_share() / 'config' / 'apriltags.yaml',
                 output_dir=tmp_path_factory.mktemp('office'))


def test_office_contains_world_models(office):
    model = office['model']
    names = {model.body(i).name.split('::')[0] for i in range(model.nbody)}
    for expected in ('office_env', 'office_env_w', 'office_chair_red', 'office_chair_green',
                     'couch', 'office_desk', 'yellow_bin', 'apriltag_36h11_0'):
        assert expected in names


def test_robot_stands_still_at_spawn_in_office(office):
    model = office['model']
    data = mujoco.MjData(model)
    _hold_arm(model, data)
    start = data.qpos[:3].copy()
    mujoco.mj_step(model, data, nstep=int(2.0 / model.opt.timestep))
    assert np.linalg.norm(data.qpos[:2] - start[:2]) < 0.02
    assert abs(data.qpos[2] - start[2]) < 0.02


def test_lidar_sees_office_walls_all_around(office):
    model = office['model']
    data = mujoco.MjData(model)
    _hold_arm(model, data)
    mujoco.mj_step(model, data, nstep=int(0.1 / model.opt.timestep))
    sensor = model.sensor(office['lidars'][0]['name'])
    ranges = data.sensordata[sensor.adr[0]:sensor.adr[0] + sensor.dim[0]]
    assert np.all((ranges > 0.1) & (ranges < 30.0)), 'closed office: every ray hits a wall'


def test_apriltags_at_layout_poses(office):
    model = office['model']
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with open(gz_sim_share() / 'config' / 'apriltags.yaml') as stream:
        tags = yaml.safe_load(stream)['tags']
    for tag in tags:
        body = model.body(f"apriltag_36h11_{tag['id']}::panel").id
        pose = tag['pose']
        assert data.xpos[body] == pytest.approx([pose['x'], pose['y'], pose['z']], abs=1e-6)
        assert model.ntex >= len(tags)
