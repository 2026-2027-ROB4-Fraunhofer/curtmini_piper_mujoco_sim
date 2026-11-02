"""Build the MuJoCo scene and robot description from the curtmini_piper_gz_sim sources.

    xacro -> URDF -> MuJoCo robot, SDF world -> MuJoCo world: scene.xml
    URDF -> robot_description with the MuJoCo ros2_control hardware
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from curtmini_piper_gz_sim.robot import load_initial_state, robot_mappings, simulation_description
import mujoco
import yaml

from .apriltags import add_apriltags
from .lidar import connector_plugin_parameters, gazebo_lidars
from .ros2_control import mujoco_description
from .robot_mjcf import robot_spec
from .world_mjcf import add_world

OUTPUT_DIR = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / \
    'curtmini_piper_mujoco_sim' / 'models'


def gz_sim_share():
    return Path(get_package_share_directory('curtmini_piper_gz_sim'))


def world_model_dirs():
    """Directories searched for model:// URIs."""
    return [Path(get_package_share_directory('neo_gz_worlds')) / 'models',
            gz_sim_share() / 'models']


def default_world_file():
    return gz_sim_share() / 'worlds' / 'curtmini_piper.sdf'


def default_settings_file():
    return Path(get_package_share_directory('curtmini_piper_mujoco_sim')) / 'config' / 'mujoco.yaml'


def default_initial_state_file():
    return gz_sim_share() / 'config' / 'initial_state.yaml'


def load_settings(path=None):
    with open(path or default_settings_file(), encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def configure_physics(spec, settings):
    physics = settings['physics']
    spec.option.timestep = physics['timestep']
    spec.option.integrator = getattr(mujoco.mjtIntegrator, f'mjINT_{physics["integrator"].upper()}')
    spec.option.cone = getattr(mujoco.mjtCone, f'mjCONE_{physics["cone"].upper()}')
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960


def build(arm_model='piper_x', use_hokuyo=True, use_cameras=False, initial_state_file=None,
          headless=False, settings_file=None, output_dir=None, world_file=None,
          apriltag_layout_file=None):
    """Build the scene and write it to the cache.

    Returns a dict with robot_description, mujoco_model (scene.xml path), model (MjModel),
    lidars and plugin_parameters (path). No AprilTags if apriltag_layout_file is None.
    """
    settings = load_settings(settings_file)
    initial_state = load_initial_state(initial_state_file or default_initial_state_file())
    mappings = robot_mappings(gz_sim_share(), arm_model, use_hokuyo, use_cameras)
    urdf = simulation_description(gz_sim_share(), mappings, initial_state)

    spec = robot_spec(urdf, settings, initial_state['base'])
    configure_physics(spec, settings)
    world_file = Path(world_file or default_world_file())
    skipped = add_world(spec, settings, world_file, world_model_dirs())
    if apriltag_layout_file:
        skipped += add_apriltags(spec, settings, apriltag_layout_file, world_model_dirs())
    if skipped:
        print(f'[curtmini_piper_mujoco_sim] {world_file.name}: not converted: '
              f'{sorted(set(skipped))}')
    model = spec.compile()

    tags = '_apriltags' if apriltag_layout_file else ''
    output = Path(output_dir or OUTPUT_DIR / f'{arm_model}_{world_file.stem}{tags}')
    output.mkdir(parents=True, exist_ok=True)
    scene = output / 'scene.xml'
    scene.write_text(spec.to_xml())
    description = mujoco_description(urdf, str(scene), headless)
    (output / 'robot_description.urdf').write_text(description)
    lidars = gazebo_lidars(urdf)
    plugin_parameters = output / 'mujoco_plugins.yaml'
    plugin_parameters.write_text(yaml.safe_dump(connector_plugin_parameters(lidars) or {}))
    return {'robot_description': description, 'mujoco_model': str(scene), 'model': model,
            'lidars': lidars, 'plugin_parameters': str(plugin_parameters)}
