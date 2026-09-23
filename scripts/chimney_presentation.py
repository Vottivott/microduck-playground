"""Visual-only cream palette and shadow helpers for the chimney demo."""
from copy import deepcopy
from typing import Sequence
import mujoco
import numpy as np

DUCK_PALETTES = {
    "cream": {
        "shell": (0.88, 0.83, 0.78, 1.0),  # #e1d3c7
        "plate": (0.80, 0.74, 0.67, 1.0),  # #cbbcac
        "beak": (0.89, 0.32, 0.07, 1.0),  # #e45111
        # "foot" is the top plate of the foot, "sole" the big bottom block
        "foot": (0.89, 0.32, 0.07, 1.0),  # orange-red top of the foot
        "ankle": (0.89, 0.32, 0.07, 1.0),  # orange-red ankle brackets
        "sole": (0.96, 0.63, 0.04, 1.0),  # #f5a00a yellow-orange sole block
        "eye": (0.96, 0.63, 0.04, 1.0),  # eye ring, same yellow-orange as the sole
        "face": (0.78, 0.78, 0.78, 1.0),  # light grey face part
        "mouth": (0.58, 0.17, 0.04, 1.0),  # #942c0a
    },
}
_MATERIAL_ROLES = {
"shell": ("top_head_shell_material", "right_shell_material", "left_shell_material", "power_support_material", "np_f970_material", "yaw_roll_motion_material"),
    "plate": ("hip_l_material", "upper_leg_left_material", "upper_leg_right_material", "leg_material", "upper_leg_rigidity_plate_material"),
    "beak": ("jaw_material", "bottom_head_shell_material"),
    "foot": ("foot_left_material", "foot_right_material"),
    "ankle": ("ankle_left_material", "ankle_right_material"),
    "sole": ("sole_left_material", "sole_right_material"),
    "eye": ("m12_lens_holder_material", "noenoeil_material"),
    "face": ("face_part_material",),
    "mouth": ("jaw_soft_material", "soft_mouth_top_material"),
}

def recolored_robot_cfg(robot_cfg, colourway):
    cfg = deepcopy(robot_cfg)
    base = cfg.spec_fn
    palette = DUCK_PALETTES[colourway]
    wanted = {name: palette[role] for role, names in _MATERIAL_ROLES.items() for name in names}
    def paint():
        spec = base()
        for mat in spec.materials:
            if mat.name in wanted:
                mat.rgba = list(wanted[mat.name])
        return spec
    cfg.spec_fn = paint
    return cfg

def configure_video_cfg(env_cfg, width: int = 1280, height: int = 720) -> None:
    """720p output and no floating command arrows / sensor markers."""
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    for command_cfg in getattr(env_cfg, "commands", {}).values():
        if hasattr(command_cfg, "debug_vis"):
            command_cfg.debug_vis = False
    for sensor_cfg in getattr(env_cfg.scene, "sensors", ()) or ():
        if hasattr(sensor_cfg, "debug_vis"):
            sensor_cfg.debug_vis = False


def fix_render_shadows(raw_env, min_extent: float = 4.0, light_dir: Sequence[float] | None = None) -> None:
    """MuJoCo sizes the directional-light shadow box from ``stat.extent``;
    with the default extent the shadow map ends a metre or two from the
    tracked robot and shows as a hard cut in the background.  Enlarge the
    shadow frustum; normal contact shadows are unchanged.

    ``light_dir`` re-aims every directional light (the scene sun).  The
    default sun points straight down, so tall off-screen structures (the
    upper ladder treads) cast detached rectangles onto the floor; aiming the
    sun along a structure's incline collapses those shadows at its foot.
    """
    renderer = getattr(raw_env, "_offline_renderer", None)
    if renderer is None:
        return
    model = renderer._model
    model.stat.extent = max(float(model.stat.extent), min_extent)
    model.vis.map.shadowclip = 1.5
    model.vis.map.shadowscale = 1.0
    model.vis.quality.shadowsize = max(int(model.vis.quality.shadowsize), 8192)
    if light_dir is not None:
        d = np.asarray(light_dir, dtype=np.float64)
        d /= np.linalg.norm(d)
        for i in range(model.nlight):
            if int(model.light_type[i]) == int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL):
                model.light_dir[i] = d
