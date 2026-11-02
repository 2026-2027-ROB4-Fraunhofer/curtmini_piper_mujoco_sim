"""Replace the Gazebo <ros2_control> blocks with one MujocoSystemInterface block.

mujoco_ros2_control starts one simulation per hardware component, so the joints of all blocks
are merged into a single system. Joint interfaces and initial values are kept.
"""

import xml.etree.ElementTree as ET

from .robot_mjcf import IMU_SITE

SYSTEM_NAME = 'MujocoSystem'
PLUGIN = 'mujoco_ros2_control/MujocoSystemInterface'
IMU_SENSOR = 'imu_sensor'
IMU_INTERFACES = (
    [f'orientation.{a}' for a in 'xyzw']
    + [f'angular_velocity.{a}' for a in 'xyz']
    + [f'linear_acceleration.{a}' for a in 'xyz']
)


def _param(parent, name, value):
    ET.SubElement(parent, 'param', name=name).text = str(value)


def mujoco_description(urdf_xml, mujoco_model, headless):
    """Return the URDF with the MuJoCo hardware and the IMU sensor."""
    root = ET.fromstring(urdf_xml)
    joints = []
    for block in root.findall('ros2_control'):
        joints.extend(block.findall('joint'))
        root.remove(block)
    for element in root.findall('gazebo'):
        root.remove(element)

    system = ET.SubElement(root, 'ros2_control', name=SYSTEM_NAME, type='system')
    hardware = ET.SubElement(system, 'hardware')
    ET.SubElement(hardware, 'plugin').text = PLUGIN
    _param(hardware, 'mujoco_model', mujoco_model)
    _param(hardware, 'headless', 'true' if headless else 'false')
    _param(hardware, 'sim_speed_factor', 1.0)
    system.extend(joints)

    sensor = ET.SubElement(system, 'sensor', name=IMU_SENSOR)
    _param(sensor, 'mujoco_type', 'imu')
    _param(sensor, 'mujoco_sensor_name', IMU_SITE)
    for interface in IMU_INTERFACES:
        ET.SubElement(sensor, 'state_interface', name=interface)

    ET.indent(root)
    return ET.tostring(root, encoding='unicode')
