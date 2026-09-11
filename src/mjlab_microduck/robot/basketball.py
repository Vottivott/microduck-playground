"""Cream Microduck colourway used by the basketball release (visual only)."""
from copy import deepcopy

PALETTE = {'shell': (0.88, 0.83, 0.78, 1.0), 'plate': (0.8, 0.74, 0.67, 1.0), 'beak': (0.89, 0.32, 0.07, 1.0), 'foot': (0.89, 0.32, 0.07, 1.0), 'ankle': (0.89, 0.32, 0.07, 1.0), 'sole': (0.96, 0.63, 0.04, 1.0), 'eye': (0.96, 0.63, 0.04, 1.0), 'face': (0.78, 0.78, 0.78, 1.0), 'mouth': (0.58, 0.17, 0.04, 1.0)}
MATERIAL_ROLES = {'shell': ('top_head_shell_material', 'right_shell_material', 'left_shell_material', 'power_support_material', 'np_f970_material', 'yaw_roll_motion_material'), 'plate': ('hip_l_material', 'upper_leg_left_material', 'upper_leg_right_material', 'leg_material', 'upper_leg_rigidity_plate_material'), 'beak': ('jaw_material', 'bottom_head_shell_material'), 'foot': ('foot_left_material', 'foot_right_material'), 'ankle': ('ankle_left_material', 'ankle_right_material'), 'sole': ('sole_left_material', 'sole_right_material'), 'eye': ('m12_lens_holder_material', 'noenoeil_material'), 'face': ('face_part_material',), 'mouth': ('jaw_soft_material', 'soft_mouth_top_material')}

def basketball_robot_cfg(robot_cfg):
    cfg = deepcopy(robot_cfg)
    base_fn = cfg.spec_fn
    def spec_fn():
        spec = base_fn()
        wanted = {name: PALETTE[role] for role, names in MATERIAL_ROLES.items() for name in names}
        for mat in spec.materials:
            if mat.name in wanted:
                mat.rgba = list(wanted[mat.name])
        return spec
    cfg.spec_fn = spec_fn
    return cfg
