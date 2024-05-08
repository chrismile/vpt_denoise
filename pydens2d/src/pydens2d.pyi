from __future__ import annotations
import numpy as np
import pydens2d
import typing

__all__ = [
    "update_visibility_field"
]


def update_visibility_field(
        density_field: np.ndarray, visibility_field: np.ndarray, aabb: np.ndarray,
        cam_res: int, cam_pos: np.ndarray, theta: float, fov: float):
    """
    Updates the visibility field.
    """
