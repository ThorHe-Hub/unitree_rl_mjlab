"""Hold evaluation commands at their source, before command-dependent observations.

The inherited resampling schedule and discarded proposals retain the original
Torch RNG consumption. They never become effective commands or standing flags.
This permits paired comparisons with the previous evaluation implementation.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
import math

import torch
from mjlab.managers.command_manager import CommandTerm
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand, UniformVelocityCommandCfg,
)


class PinnedVelocityCommand(UniformVelocityCommand):
    def __init__(self, cfg, env):
        if cfg.init_velocity_prob != 0:
            raise ValueError("Pinned evaluation requires init_velocity_prob=0; random proposals must not write physical state")
        super().__init__(cfg, env)
        self.held_command = torch.zeros_like(self.vel_command_b)
        self.held_command[:, 0] = cfg.pin_vx
        self._update_command()

    def _update_command(self):
        self.vel_command_b.copy_(self.held_command)
        self.is_heading_env.zero_()
        self.is_standing_env.copy_(torch.linalg.vector_norm(self.held_command, dim=1) < 0.1)

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        self._update_command()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        ids = env_ids
        self.held_command[ids] = 0.0
        self.held_command[ids, 0] = self.cfg.pin_vx
        return super().reset(env_ids)

    def compute(self, dt):
        # Bypass UniformVelocityCommand's post-compute viewer joystick override.
        CommandTerm.compute(self, dt)

    def set_velocity(self, vx, vy, vyaw):
        if not all(math.isfinite(v) for v in (vx, vy, vyaw)):
            raise ValueError("Non-finite pinned command")
        self.held_command[:, 0] = vx
        self.held_command[:, 1] = vy
        self.held_command[:, 2] = vyaw
        self._update_command()


@dataclass(kw_only=True)
class PinnedVelocityCommandCfg(UniformVelocityCommandCfg):
    pin_vx: float = 0.5

    def build(self, env):
        return PinnedVelocityCommand(self, env)


def make_pinned_velocity_cfg(source, pin_vx):
    if not isinstance(source, UniformVelocityCommandCfg):
        raise TypeError("--pin-cmd requires a UniformVelocityCommandCfg named twist")
    if not math.isfinite(pin_vx):
        raise ValueError("pin_vx must be finite")
    if source.init_velocity_prob != 0:
        raise ValueError("Pinned evaluation requires init_velocity_prob=0")
    values = {f.name: deepcopy(getattr(source, f.name))
              for f in fields(UniformVelocityCommandCfg) if f.init}
    return PinnedVelocityCommandCfg(**values, pin_vx=pin_vx)
