"""Convert meshes and textures to formats MuJoCo can load, with an on-disk cache.

Meshes: DAE and STL files above MuJoCo's face limit become OBJ, one file per colour.
Textures: non-PNG images become PNG; small images are upscaled without filtering so that
low-resolution textures (e.g. AprilTags) stay sharp.
"""

import hashlib
import json
import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / \
    'curtmini_piper_mujoco_sim' / 'meshes'
CACHE_VERSION = 3
MIN_TEXTURE_SIZE = 512  # px
STL_FACE_LIMIT = 200000  # MuJoCo's STL decoder limit


def _cache_dir_for(path):
    stat = path.stat()
    key = f'{CACHE_VERSION}:{path}:{stat.st_size}:{stat.st_mtime_ns}'
    return CACHE_DIR / f'{path.stem}-{hashlib.sha1(key.encode()).hexdigest()[:12]}'


def _stl_face_count(path):
    with path.open('rb') as stream:
        header = stream.read(84)
    if len(header) < 84 or header[:5] == b'solid' and \
            path.stat().st_size != 84 + 50 * int.from_bytes(header[80:84], 'little'):
        return None  # ASCII STL
    return int.from_bytes(header[80:84], 'little')


def _rgba(geometry):
    import numpy as np
    visual = geometry.visual
    material = getattr(visual, 'material', None)
    color = None
    if material is not None:
        color = getattr(material, 'main_color', None)
    if color is None and hasattr(visual, 'main_color'):
        color = visual.main_color
    if color is None:
        return [0.7, 0.7, 0.7, 1.0]
    return [round(float(c), 4) for c in np.asarray(color, dtype=float) / 255.0]


def _convert(path, out_dir):
    """Write one OBJ per colour into out_dir; return [{'file', 'rgba'}]."""
    import trimesh

    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = trimesh.load(str(path), force='scene')
    by_colour = {}
    for geometry in loaded.dump():
        if len(geometry.faces):
            by_colour.setdefault(tuple(_rgba(geometry)), []).append(
                trimesh.Trimesh(vertices=geometry.vertices, faces=geometry.faces, process=False))
    parts = []
    for index, (rgba, meshes) in enumerate(by_colour.items()):
        name = f'part{index}.obj'
        trimesh.util.concatenate(meshes).export(str(out_dir / name))
        parts.append({'file': str(out_dir / name), 'rgba': list(rgba)})
    if not parts:
        raise RuntimeError(f'No triangles found in mesh {path}')
    (out_dir / 'parts.json').write_text(json.dumps(parts, indent=1))
    return parts


def mujoco_mesh_parts(filename):
    """Return [{'file', 'rgba'}] for a mesh; rgba is None when the file has no colour."""
    path = Path(filename).resolve()
    suffix = path.suffix.lower()
    if suffix == '.obj':
        return [{'file': str(path), 'rgba': None}]
    if suffix == '.stl':
        faces = _stl_face_count(path)
        if faces is not None and faces <= STL_FACE_LIMIT:
            return [{'file': str(path), 'rgba': None}]
    elif suffix != '.dae':
        raise RuntimeError(f'Unsupported mesh format for MuJoCo: {path}')

    out_dir = _cache_dir_for(path)
    index = out_dir / 'parts.json'
    if index.is_file():
        return json.loads(index.read_text())
    parts = _convert(path, out_dir)
    if suffix == '.stl':
        for part in parts:
            part['rgba'] = None
        index.write_text(json.dumps(parts, indent=1))
    return parts


def mujoco_texture_file(filename):
    """Return a PNG path for an image file, converting it if needed."""
    from PIL import Image

    path = Path(filename).resolve()
    with Image.open(path) as image:
        small = min(image.size) < MIN_TEXTURE_SIZE
    if path.suffix.lower() == '.png' and not small:
        return str(path)
    out_dir = _cache_dir_for(path)
    png = out_dir / (path.stem + '.png')
    if not png.is_file():
        out_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(path).convert('RGBA')
        if small:
            factor = -(-MIN_TEXTURE_SIZE // min(image.size))  # ceiling division
            image = image.resize((image.width * factor, image.height * factor),
                                 Image.NEAREST)
        image.save(png)
    return str(png)
