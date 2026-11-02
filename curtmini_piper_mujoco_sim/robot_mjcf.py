"""Convert the robot URDF to a MuJoCo model.

MuJoCo's URDF importer reads links, joints, inertias and collision shapes. MjSpec then adds
what URDF cannot express: visuals, a free base joint, actuators, mimic constraints, sensors and
roller wheels.
"""

import math
import re
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import mujoco

from .lidar import add_lidars, gazebo_lidars
from .meshes import mujoco_mesh_parts

FREE_JOINT = 'floating_base_joint'
IMU_SITE = 'imu_sensor'
VISUAL_GROUP = 1     # visible in the viewer by default
COLLISION_GROUP = 3  # hidden in the viewer by default

# Wheel rollers (contype 0, conaffinity ENVIRONMENT_BIT) collide only with world geoms, which
# must use contype ENVIRONMENT_CONTYPE.
ENVIRONMENT_BIT = 2
ENVIRONMENT_CONTYPE = 1 | ENVIRONMENT_BIT

URDF_TAGS_NOT_FOR_MUJOCO = ('gazebo', 'ros2_control', 'transmission')


def _floats(text, default):
    return [float(v) for v in text.split()] if text else list(default)


def rpy_to_quat(rpy):
    """Convert URDF roll-pitch-yaw to a MuJoCo quaternion (w, x, y, z)."""
    roll, pitch, yaw = (0.5 * a for a in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def _resolve_package_uri(filename):
    match = re.match(r'package://([^/]+)/(.*)', filename)
    if match is None:
        return filename[len('file://'):] if filename.startswith('file://') else filename
    return f'{get_package_share_directory(match.group(1))}/{match.group(2)}'


def _origin(element):
    origin = element.find('origin')
    if origin is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    return _floats(origin.get('xyz'), [0, 0, 0]), _floats(origin.get('rpy'), [0, 0, 0])


def _visual_rgba(visual, named_materials):
    material = visual.find('material')
    if material is None:
        return None
    color = material.find('color')
    if color is not None:
        return _floats(color.get('rgba'), [0.7, 0.7, 0.7, 1])
    return named_materials.get(material.get('name'))


def ros2_control_joints(urdf_root):
    """Return {joint name: [command interfaces]} from all <ros2_control> blocks."""
    joints = {}
    for block in urdf_root.findall('ros2_control'):
        for joint in block.findall('joint'):
            joints[joint.get('name')] = [c.get('name') for c in joint.findall('command_interface')]
    return joints


def prepare_urdf(urdf_xml):
    """Return (URDF prepared for MuJoCo's importer, visuals removed from it).

    Visuals are re-added by add_visuals. fusestatic is disabled so that fixed sensor links
    (imu_link, hokuyo_link) remain separate bodies.
    """
    root = ET.fromstring(urdf_xml)
    for tag in URDF_TAGS_NOT_FOR_MUJOCO:
        for element in root.findall(tag):
            root.remove(element)

    named_materials = {}
    for material in root.findall('material'):
        color = material.find('color')
        if color is not None:
            named_materials[material.get('name')] = _floats(color.get('rgba'), [0.7] * 3 + [1])

    visuals = []
    for link in root.findall('link'):
        for visual in link.findall('visual'):
            xyz, rpy = _origin(visual)
            visuals.append({
                'link': link.get('name'),
                'pos': xyz,
                'quat': rpy_to_quat(rpy),
                'geometry': visual.find('geometry')[0],
                'rgba': _visual_rgba(visual, named_materials),
            })
            link.remove(visual)
        for collision in link.findall('collision'):
            mesh = collision.find('geometry/mesh')
            if mesh is None:
                continue
            parts = mujoco_mesh_parts(_resolve_package_uri(mesh.get('filename')))
            if len(parts) != 1:
                raise RuntimeError(f'Collision mesh of {link.get("name")} has {len(parts)} parts')
            mesh.set('filename', parts[0]['file'])

    for visual in visuals:
        geometry = visual['geometry']
        if geometry.tag == 'mesh':
            geometry.set('filename', _resolve_package_uri(geometry.get('filename')))

    mujoco_tag = ET.Element('mujoco')
    ET.SubElement(mujoco_tag, 'compiler', fusestatic='false', discardvisual='false',
                  strippath='false', balanceinertia='true',
                  boundmass='1e-4', boundinertia='1e-8')
    root.insert(0, mujoco_tag)
    return ET.tostring(root, encoding='unicode'), visuals


def hide_collision_geoms(spec):
    """Move collision geoms to a hidden group; they enclose the visual meshes."""
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.group = COLLISION_GROUP


def add_visuals(spec, visuals):
    """Add visual-only geoms (no contacts, no mass)."""
    meshes = {}
    for index, visual in enumerate(visuals):
        body = spec.body(visual['link'])
        geometry = visual['geometry']
        geoms = []
        if geometry.tag == 'mesh':
            scale = _floats(geometry.get('scale'), [1, 1, 1])
            for part in mujoco_mesh_parts(geometry.get('filename')):
                key = (part['file'], tuple(scale))
                if key not in meshes:
                    meshes[key] = f'visual_mesh{len(meshes)}'
                    # Shell inertia accepts flat meshes; visuals carry no mass anyway.
                    spec.add_mesh(name=meshes[key], file=part['file'], scale=scale,
                                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
                geoms.append(dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=meshes[key],
                                  rgba=part['rgba'] or visual['rgba']))
        elif geometry.tag == 'box':
            size = _floats(geometry.get('size'), [0, 0, 0])
            geoms.append(dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in size]))
        elif geometry.tag == 'cylinder':
            geoms.append(dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                              size=[float(geometry.get('radius')),
                                    float(geometry.get('length')) / 2, 0]))
        elif geometry.tag == 'sphere':
            geoms.append(dict(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                              size=[float(geometry.get('radius')), 0, 0]))
        else:
            raise RuntimeError(f'Unsupported visual geometry <{geometry.tag}> on {visual["link"]}')
        for geom in geoms:
            rgba = geom.pop('rgba', None) or visual['rgba'] or [0.7, 0.7, 0.7, 1]
            body.add_geom(name=f'{visual["link"]}_visual{index}_{len(body.geoms)}',
                          pos=visual['pos'], quat=visual['quat'], rgba=rgba,
                          group=VISUAL_GROUP, contype=0, conaffinity=0, density=0, **geom)


def _urdf_joints(urdf_root):
    return {joint.get('name'): joint for joint in urdf_root.findall('joint')}


def _limit(joint, attribute):
    limit = joint.find('limit')
    if limit is None or limit.get(attribute) is None:
        return None
    return float(limit.get(attribute))


def add_free_joint(spec, root_link, base_pose):
    body = spec.body(root_link)
    body.pos = [base_pose['x'], base_pose['y'], base_pose['z']]
    body.quat = rpy_to_quat([0.0, 0.0, base_pose['orientation']])
    body.add_freejoint(name=FREE_JOINT)


def add_actuators(spec, urdf_root, settings):
    """Add one actuator per ros2_control joint, named after the joint."""
    joints = _urdf_joints(urdf_root)
    for name, interfaces in ros2_control_joints(urdf_root).items():
        joint = joints[name]
        effort = _limit(joint, 'effort')
        actuator = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        if 'velocity' in interfaces:
            base = settings['base']
            actuator.set_to_velocity(kv=base['wheel_kv'])
            torque = base['wheel_max_torque']
        elif 'position' in interfaces:
            gains = settings['gripper'] if joint.get('type') == 'prismatic' else settings['arm']
            actuator.set_to_position(kp=gains['kp'], dampratio=gains['dampratio'])
            spec.joint(name).armature = gains['armature']
            lower, upper = _limit(joint, 'lower'), _limit(joint, 'upper')
            if lower is not None and upper is not None:
                actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
                actuator.ctrlrange = [lower, upper]
            torque = effort
        else:
            raise RuntimeError(f'Joint {name}: no position or velocity command interface')
        if torque:
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            actuator.forcerange = [-torque, torque]


def add_mimic_constraints(spec, urdf_root):
    """Add a joint equality for each URDF <mimic> joint."""
    mimic_bodies = []
    for joint in urdf_root.findall('joint'):
        mimic = joint.find('mimic')
        if mimic is None:
            continue
        multiplier = float(mimic.get('multiplier', 1.0))
        offset = float(mimic.get('offset', 0.0))
        equality = spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT,
                                     objtype=mujoco.mjtObj.mjOBJ_JOINT,
                                     name1=joint.get('name'), name2=mimic.get('joint'))
        equality.data[:5] = [offset, multiplier, 0.0, 0.0, 0.0]
        mimic_bodies.append(joint.find('child').get('link'))
    # Mimic fingers move towards each other; exclude their mutual contacts.
    for i, first in enumerate(mimic_bodies):
        for second in mimic_bodies[i + 1:]:
            spec.add_exclude(bodyname1=first, bodyname2=second)


def add_imu(spec, settings):
    imu = settings['imu']
    spec.body(imu['link']).add_site(name=IMU_SITE, size=[0.01, 0, 0], group=VISUAL_GROUP + 4)
    for suffix, sensor_type, noise in (('_quat', mujoco.mjtSensor.mjSENS_FRAMEQUAT, 0.0),
                                       ('_gyro', mujoco.mjtSensor.mjSENS_GYRO, imu['gyro_noise']),
                                       ('_accel', mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                                        imu['accel_noise'])):
        spec.add_sensor(name=IMU_SITE + suffix, type=sensor_type, objtype=mujoco.mjtObj.mjOBJ_SITE,
                        objname=IMU_SITE, noise=noise)


def _unit(vector):
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def add_roller_wheels(spec, urdf_root, settings):
    """Replace each driven wheel's collision cylinder with a ring of passive rollers.

    The rollers grip in the rolling direction and roll sideways against the axle friction,
    giving the anisotropic wheel friction a skid-steer base needs to turn.
    """
    base = settings['base']
    rollers = base['rollers']
    joints = _urdf_joints(urdf_root)
    for name, interfaces in ros2_control_joints(urdf_root).items():
        if 'velocity' not in interfaces:
            continue
        wheel = spec.body(joints[name].find('child').get('link'))
        axis = _unit(spec.joint(name).axis)
        cylinders = [g for g in wheel.geoms
                     if g.contype and g.type == mujoco.mjtGeom.mjGEOM_CYLINDER]
        if len(cylinders) != 1:
            raise RuntimeError(f'Wheel {wheel.name}: expected one collision cylinder')
        cylinder = cylinders[0]
        cylinder.contype = cylinder.conaffinity = 0
        wheel_radius, centre = cylinder.size[0], list(cylinder.pos)

        count = rollers['count']
        roller_radius = 1.25 * math.pi * wheel_radius / count  # neighbouring rollers overlap
        ring_radius = wheel_radius - roller_radius + rollers['sink_compensation']
        e1 = _unit(_cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 1, 0]))
        e2 = _cross(axis, e1)
        for i in range(count):
            angle = 2 * math.pi * i / count
            radial = [math.cos(angle) * a + math.sin(angle) * b for a, b in zip(e1, e2)]
            roller = wheel.add_body(
                name=f'{wheel.name}_roller{i}',
                pos=[c + ring_radius * r for c, r in zip(centre, radial)],
                mass=0.005, inertia=[1e-6] * 3)
            roller.explicitinertial = True  # otherwise to_xml() omits the mass
            roller.add_joint(name=f'{wheel.name}_roller{i}_joint', type=mujoco.mjtJoint.mjJNT_HINGE,
                             axis=_cross(axis, radial), frictionloss=rollers['axle_friction'],
                             armature=1e-4, damping=1e-3)
            roller.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[roller_radius, 0, 0],
                            contype=0, conaffinity=ENVIRONMENT_BIT, group=COLLISION_GROUP,
                            friction=rollers['friction'], priority=1, density=0,
                            rgba=[0.2, 0.2, 0.2, 1])


def root_link(urdf_root):
    children = {joint.find('child').get('link') for joint in urdf_root.findall('joint')}
    roots = [link.get('name') for link in urdf_root.findall('link')
             if link.get('name') not in children]
    if len(roots) != 1:
        raise RuntimeError(f'Expected one root link, found {roots}')
    return roots[0]


def robot_spec(urdf_xml, settings, base_pose):
    """Return the robot MjSpec at base_pose {x, y, z, orientation (yaw)}."""
    urdf_root = ET.fromstring(urdf_xml)
    prepared, visuals = prepare_urdf(urdf_xml)
    spec = mujoco.MjSpec.from_string(prepared)
    hide_collision_geoms(spec)
    add_visuals(spec, visuals)
    add_free_joint(spec, root_link(urdf_root), base_pose)
    add_actuators(spec, urdf_root, settings)
    add_mimic_constraints(spec, urdf_root)
    add_imu(spec, settings)
    add_lidars(spec, gazebo_lidars(urdf_xml), VISUAL_GROUP)
    add_roller_wheels(spec, urdf_root, settings)
    return spec
