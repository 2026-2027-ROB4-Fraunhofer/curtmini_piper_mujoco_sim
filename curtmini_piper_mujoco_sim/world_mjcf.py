"""Convert an SDFormat world to MuJoCo.

The SDF is parsed with libsdformat; models, links, collisions, visuals and directional lights
are added to the MjSpec. Static models are fixed to the world, others get a free joint.
Physics and plugin settings are ignored.
"""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import sdformat as sdf

from .meshes import mujoco_mesh_parts, mujoco_texture_file
from .robot_mjcf import COLLISION_GROUP, ENVIRONMENT_CONTYPE, VISUAL_GROUP

DEFAULT_RGBA = [0.7, 0.7, 0.7, 1.0]


def load_world(world_file, model_dirs):
    root = sdf.Root()
    root.load(str(world_file), _parser_config(model_dirs))
    if root.world_count() != 1:
        raise RuntimeError(f'{world_file}: expected one <world>, found {root.world_count()}')
    return root.world_by_index(0)


def _pose(gz_pose):
    """Convert a gz.math Pose3d to (pos, quat wxyz)."""
    p, q = gz_pose.pos(), gz_pose.rot()
    return np.array([p.x(), p.y(), p.z()]), np.array([q.w(), q.x(), q.y(), q.z()])


def _compose(parent, child):
    pos, quat = np.zeros(3), np.zeros(4)
    mujoco.mju_mulPose(pos, quat, parent[0], parent[1], child[0], child[1])
    return pos, quat


def _relative(element):
    """Return the pose of an SDF element relative to its parent frame."""
    return _pose(element.semantic_pose().resolve())


def _vector(gz_vector):
    return [gz_vector.x(), gz_vector.y(), gz_vector.z()]


def _rgba(material):
    if material is None:
        return None
    color = material.diffuse()
    return [color.r(), color.g(), color.b(), color.a()]


def _resolve_uri(uri, declared_in, model_dirs):
    """Resolve a model://, file:// or relative URI to a file path."""
    if uri.startswith('file://'):
        uri = uri[len('file://'):]
    if uri.startswith('model://'):
        name, _, rest = uri[len('model://'):].partition('/')
        # libsdformat resolves model:// only for <include>, not for mesh or texture URIs.
        candidates = [Path(d) / name / rest for d in model_dirs]
        if declared_in:
            candidates += [p / rest for p in Path(declared_in).resolve().parents if p.name == name]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        raise RuntimeError(f'Cannot resolve {uri} (declared in {declared_in or "a string"})')
    path = Path(uri)
    if not path.is_absolute() and declared_in:
        path = Path(declared_in).resolve().parent / path
    return str(path)


def _asset_name(prefix, key):
    """Return a stable asset name for a source file, so builders can share assets."""
    return f'{prefix}_{hashlib.sha1(repr(key).encode()).hexdigest()[:12]}'


class WorldBuilder:
    def __init__(self, spec, settings, model_dirs, sdf_file=None):
        self.spec = spec
        self.settings = settings
        self.model_dirs = model_dirs
        self.sdf_file = sdf_file
        self.skipped = []

    def _textured_material(self, material):
        """Return a MuJoCo material for the SDF PBR albedo map, or None."""
        pbr = material.pbr_material() if material is not None else None
        workflow = None
        if pbr is not None:
            workflow = pbr.workflow(sdf.PbrWorkflowType.METAL) or \
                pbr.workflow(sdf.PbrWorkflowType.SPECULAR)
        if workflow is None or not workflow.albedo_map():
            return None
        path = mujoco_texture_file(_resolve_uri(
            workflow.albedo_map(), material.file_path() or self.sdf_file, self.model_dirs))
        name = _asset_name('world_texture', path)
        if self.spec.texture(name) is None:
            self.spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
            material = self.spec.add_material(name=name)
            material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
        return name

    def _mesh(self, path, scale):
        parts = []
        for part in mujoco_mesh_parts(path):
            name = _asset_name('world_mesh', (part['file'], tuple(scale)))
            if self.spec.mesh(name) is None:
                self.spec.add_mesh(name=name, file=part['file'], scale=scale,
                                   inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            parts.append((name, part['rgba']))
        return parts

    def _geoms(self, geometry):
        """Return [(geom attributes, mesh rgba or None)] for an SDF <geometry>."""
        kind = geometry.type()
        types = sdf.GeometryType
        if kind == types.BOX:
            return [(dict(type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[s / 2 for s in _vector(geometry.box_shape().size())]), None)]
        if kind == types.CYLINDER:
            shape = geometry.cylinder_shape()
            return [(dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                          size=[shape.radius(), shape.length() / 2, 0]), None)]
        if kind == types.SPHERE:
            return [(dict(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[geometry.sphere_shape().radius(), 0, 0]), None)]
        if kind == types.CAPSULE:
            shape = geometry.capsule_shape()
            return [(dict(type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          size=[shape.radius(), shape.length() / 2, 0]), None)]
        if kind == types.ELLIPSOID:
            return [(dict(type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                          size=_vector(geometry.ellipsoid_shape().radii())), None)]
        if kind == types.PLANE:
            shape = geometry.plane_shape()
            if _vector(shape.normal()) != [0.0, 0.0, 1.0]:
                raise RuntimeError('Only horizontal planes (normal 0 0 1) are supported')
            size = shape.size()
            return [(dict(type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[size.x() / 2, size.y() / 2, 0.1]), None)]
        if kind == types.MESH:
            mesh = geometry.mesh_shape()
            path = _resolve_uri(mesh.uri(), mesh.file_path() or self.sdf_file, self.model_dirs)
            parts = self._mesh(path, _vector(mesh.scale()))
            return [(dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name), rgba)
                    for name, rgba in parts]
        self.skipped.append(str(kind))
        return []

    def add_link(self, link, pose, static, name):
        body = self.spec.worldbody.add_body(name=name, pos=pose[0], quat=pose[1])
        if not static:
            body.add_freejoint()
            inertial = link.inertial()
            mass = inertial.mass_matrix().mass()
            if mass > 0:
                body.mass = mass
                diagonal = inertial.mass_matrix().diagonal_moments()
                body.inertia = _vector(diagonal)
                body.ipos = _vector(inertial.pose().pos())
                body.explicitinertial = True
        for i in range(link.collision_count()):
            collision = link.collision_by_index(i)
            pos, quat = _relative(collision)
            for attributes, _ in self._geoms(collision.geometry()):
                body.add_geom(pos=pos, quat=quat, group=COLLISION_GROUP,
                              contype=ENVIRONMENT_CONTYPE, conaffinity=1,
                              friction=self.settings['floor']['friction'], **attributes)
        for i in range(link.visual_count()):
            visual = link.visual_by_index(i)
            pos, quat = _relative(visual)
            material_rgba = _rgba(visual.material())
            texture = self._textured_material(visual.material())
            for attributes, mesh_rgba in self._geoms(visual.geometry()):
                if texture:
                    attributes = dict(attributes, material=texture, rgba=[1, 1, 1, 1])
                else:
                    attributes = dict(attributes, rgba=mesh_rgba or material_rgba or DEFAULT_RGBA)
                body.add_geom(pos=pos, quat=quat, group=VISUAL_GROUP, contype=0, conaffinity=0,
                              density=0, **attributes)

    def add_model(self, model, parent_pose, parent_static=False, prefix=''):
        pose = _compose(parent_pose, _relative(model))
        static = parent_static or model.static()
        scope = f'{prefix}{model.name()}'
        for i in range(model.link_count()):
            link = model.link_by_index(i)
            self.add_link(link, _compose(pose, _relative(link)), static, f'{scope}::{link.name()}')
        for i in range(model.model_count()):
            self.add_model(model.model_by_index(i), pose, static, f'{scope}::')

    def add_lights(self, world):
        for i in range(world.light_count()):
            light = world.light_by_index(i)
            if light.type() != sdf.LightType.DIRECTIONAL:
                self.skipped.append(f'light {light.name()} ({light.type()})')
                continue
            diffuse = light.diffuse()
            self.spec.worldbody.add_light(
                name=light.name(), type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                pos=_vector(light.raw_pose().pos()), dir=_vector(light.direction()),
                diffuse=[diffuse.r(), diffuse.g(), diffuse.b()],
                castshadow=light.cast_shadows())


IDENTITY = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


def _parser_config(model_dirs):
    config = sdf.ParserConfig()
    for model_dir in model_dirs:
        config.add_uri_path('model://', str(model_dir))
    return config


def add_world(spec, settings, world_file, model_dirs):
    """Add all models and directional lights of an SDF world; return skipped features."""
    world = load_world(world_file, model_dirs)
    builder = WorldBuilder(spec, settings, model_dirs, world_file)
    for i in range(world.model_count()):
        builder.add_model(world.model_by_index(i), IDENTITY)
    builder.add_lights(world)
    return builder.skipped


def add_sdf_model(spec, settings, sdf_string, pos, quat, model_dirs):
    """Add an SDF <model> string at the given world pose; return skipped features."""
    root = sdf.Root()
    root.load_sdf_string(sdf_string, _parser_config(model_dirs))
    if root.model() is None:
        raise RuntimeError('SDF string has no top-level <model>')
    builder = WorldBuilder(spec, settings, model_dirs)
    builder.add_model(root.model(), (np.asarray(pos, float), np.asarray(quat, float)))
    return builder.skipped
