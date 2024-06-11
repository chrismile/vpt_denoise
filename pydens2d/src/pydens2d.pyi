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


def compute_energy(
        num_iterations: int, gamma: float, non_empty_voxel_pos_field: np.ndarray,
        obs_freq_field: np.ndarray, angular_obs_freq_field: int):
    """
    Calculates the camera view combination energy term field.
    """


def update_observation_frequency_fields(
        density_field: np.ndarray, obs_freq_field: np.ndarray, angular_obs_freq_field: int, aabb: np.ndarray,
        cam_res: int, cam_pos: np.ndarray, theta: float, fov: float):
    """
    Updates the observation frequency fields.
    """
