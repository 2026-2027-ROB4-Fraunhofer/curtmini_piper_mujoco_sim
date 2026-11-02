"""2-D lidars configured from the Gazebo <sensor> blocks in the robot description.

Each lidar becomes a mujoco_3d_lidar sensor, published by the Mujoco3dLidarPlugin on a raw
topic and republished by scan_relay. The sensor housing is drawn as sites so that it does not
block the rays.
"""

import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix
import mujoco

LIDAR_PLUGIN = 'mujoco.plugin.lidar'
GAZEBO_LIDAR_TYPES = ('gpu_lidar', 'lidar', 'gpu_ray', 'ray')
SITE_TYPES = {  # geom types that a site can also draw
    mujoco.mjtGeom.mjGEOM_BOX,
    mujoco.mjtGeom.mjGEOM_CYLINDER,
    mujoco.mjtGeom.mjGEOM_SPHERE,
    mujoco.mjtGeom.mjGEOM_CAPSULE,
}

_plugin_loaded = False


def load_plugin_library():
    """Load the mujoco.plugin.lidar library into this process."""
    global _plugin_loaded
    if not _plugin_loaded:
        prefix = get_package_prefix('mujoco_3d_lidar')
        mujoco.mj_loadPluginLibrary(
            f'{prefix}/lib/mujoco_3d_lidar/mujoco_plugin/libmujoco_3d_lidar.so')
        _plugin_loaded = True


def _number(element, path, default=None):
    text = element.findtext(path)
    return float(text) if text is not None else default


def gazebo_lidars(urdf_xml):
    """Return the parameters of every 2-D lidar <sensor> in the URDF's <gazebo> blocks."""
    lidars = []
    for gazebo in ET.fromstring(urdf_xml).findall('gazebo'):
        for sensor in gazebo.findall('sensor'):
            if sensor.get('type') not in GAZEBO_LIDAR_TYPES:
                continue
            scan = sensor.find('lidar')
            if scan is None:
                scan = sensor.find('ray')
            if int(_number(scan, 'scan/vertical/samples', 1)) != 1:
                raise RuntimeError(f'Lidar {sensor.get("name")}: only 2-D (one row) is supported')
            link = gazebo.get('reference')
            lidars.append({
                'name': sensor.get('name'),
                'link': link,
                'frame': sensor.findtext('gz_frame_id') or link,
                'topic': sensor.findtext('topic') or '/scan',
                'rate': _number(sensor, 'update_rate', 10.0),
                'samples': int(_number(scan, 'scan/horizontal/samples')),
                'min_angle': _number(scan, 'scan/horizontal/min_angle'),
                'max_angle': _number(scan, 'scan/horizontal/max_angle'),
                'min_range': _number(scan, 'range/min', 0.0),
                'max_range': _number(scan, 'range/max'),
                'noise': _number(scan, 'noise/stddev', 0.0),
            })
    return lidars


def _make_ray_transparent(body, visual_group):
    """Draw the body's visual geoms as sites and make the geoms invisible to rays."""
    for geom in body.geoms:
        if geom.group == visual_group and geom.type in SITE_TYPES:
            body.add_site(name=f'{geom.name}_site', type=geom.type, size=geom.size,
                          pos=geom.pos, quat=geom.quat, rgba=geom.rgba, group=visual_group)
        geom.rgba = [geom.rgba[0], geom.rgba[1], geom.rgba[2], 0.0]  # mj_ray skips alpha 0


def add_lidars(spec, lidars, visual_group):
    if not lidars:
        return
    load_plugin_library()
    spec.activate_plugin(LIDAR_PLUGIN)
    for lidar in lidars:
        body = spec.body(lidar['link'])
        _make_ray_transparent(body, visual_group)
        site = body.add_site(name=f'{lidar["name"]}_origin', size=[0.005, 0, 0],
                             group=visual_group + 4)
        plugin = spec.add_plugin(name=f'{lidar["name"]}_plugin', plugin_name=LIDAR_PLUGIN,
                                 active=True)
        plugin.config = {
            'resolution': f'{lidar["samples"]} 1',
            'azimuth_range': f'{lidar["min_angle"]} {lidar["max_angle"]}',
            'elevation_range': '0',
            'min_range': str(lidar['min_range']),
            'max_range': str(lidar['max_range']),
            'update_rate': str(lidar['rate']),
        }
        sensor = spec.add_sensor(name=lidar['name'], type=mujoco.mjtSensor.mjSENS_PLUGIN,
                                 objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name)
        sensor.plugin = plugin


def raw_topic(lidar):
    return f'/mujoco/{lidar["name"]}/scan_raw'


def connector_plugin_parameters(lidars):
    """Return the mujoco_plugins parameters for the lidars."""
    if not lidars:
        return {}
    plugin = {'type': 'mujoco_ros2_control_plugins/Mujoco3dLidarPlugin'}
    for lidar in lidars:
        plugin[lidar['name']] = {'frame_name': lidar['frame'], 'topic': raw_topic(lidar)}
    return {'/**': {'ros__parameters': {'mujoco_plugins': {'lidar_plugin': plugin}}}}
