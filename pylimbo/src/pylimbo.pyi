from __future__ import annotations
import torch
import pylimbo
import typing

__all__ = [
    "maximize"
]


class BayOptSettings:
    def __init__(self) -> None: ...
    pbounds: int = ...
    num_iterations: int = ...
    num_optimizer_iterations: int = ...
    ucb_kappa: float = ...

def maximize(settings: BayOptSettings, init_points: list, callback: typing.Callable) -> tuple:
    """
    Applies Bayesian optimization (single-threaded).
    """
