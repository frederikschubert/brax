# Copyright 2021 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trains a humanoid to run in the +x direction."""

import jax
import jax.numpy as jnp

import brax
from brax.envs import env
from brax.physics import bodies
from brax.physics.base import take

from google.protobuf import text_format


class Humanoid(env.Env):
  """Trains a humanoid to run in the +x direction."""

  def __init__(self, **kwargs):
    config = text_format.Parse(_SYSTEM_CONFIG, brax.Config())
    super().__init__(config, **kwargs)
    body = bodies.Body.from_config(config)
    body = take(body, body.idx[:-1])  # skip the floor body
    self.mass = body.mass.reshape(-1, 1)
    self.inertia = body.inertia

  def reset(self, rng: jnp.ndarray) -> env.State:
    """Resets the environment to an initial state."""
    qp = self.sys.default_qp()
    qp, info = self.sys.step(qp,
                             jax.random.uniform(rng, (self.action_size,)) * .5)
    obs = self._get_obs(qp, info, jnp.zeros(self.action_size))
    reward, done, steps = jnp.zeros(3)
    metrics = {
        'reward_linvel': jnp.zeros(()),
        'reward_quadctrl': jnp.zeros(()),
        'reward_alive': jnp.zeros(()),
        'reward_impact': jnp.zeros(())
    }
    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def step(self, state: env.State, action: jnp.ndarray) -> env.State:
    """Run one timestep of the environment's dynamics."""
    rng = state.rng
    # note the minus sign.  reverse torque improves performance over a range of
    # hparams.  as to why: ¯\_(ツ)_/¯
    qp, info = self.sys.step(state.qp, -action)
    obs = self._get_obs(qp, info, action)

    pos_before = state.qp.pos[:-1]  # ignore floor at last index
    pos_after = qp.pos[:-1]  # ignore floor at last index
    com_before = jnp.sum(pos_before * self.mass, axis=0) / jnp.sum(self.mass)
    com_after = jnp.sum(pos_after * self.mass, axis=0) / jnp.sum(self.mass)
    lin_vel_cost = 1.25 * (com_after[0] - com_before[0]) / self.sys.config.dt
    quad_ctrl_cost = .01 * jnp.sum(jnp.square(action))
    # can ignore contact cost, see: https://github.com/openai/gym/issues/1541
    quad_impact_cost = 0.0
    alive_bonus = 5.0
    reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

    steps = state.steps + self.action_repeat
    done = jnp.where(steps >= self.episode_length, x=1.0, y=0.0)
    done = jnp.where(qp.pos[0, 2] < 0.6, x=1.0, y=done)
    done = jnp.where(qp.pos[0, 2] > 2.1, x=1.0, y=done)
    metrics = {
        'reward_linvel': lin_vel_cost,
        'reward_quadctrl': -quad_ctrl_cost,
        'reward_alive': alive_bonus,
        'reward_impact': -quad_impact_cost
    }

    return env.State(rng, qp, info, obs, reward, done, steps, metrics)

  def _get_obs(self, qp: brax.QP, info: brax.Info,
               action: jnp.ndarray) -> jnp.ndarray:
    """Observe humanoid body position, velocities, and angles."""
    # some pre-processing to pull joint angles and velocities
    joint_1d_angle, joint_1d_vel = self.sys.joint_revolute.angle_vel(qp)
    joint_2d_angle, joint_2d_vel = self.sys.joint_universal.angle_vel(qp)
    joint_3d_angle, joint_3d_vel = self.sys.joint_spherical.angle_vel(qp)

    # qpos:
    # Z of the torso (1,)
    # orientation of the torso as quaternion (4,)
    # joint angles (8,)
    qpos = [
        qp.pos[0, 2:], qp.rot[0], joint_1d_angle[0], joint_2d_angle[0],
        joint_2d_angle[1], joint_3d_angle[0], joint_3d_angle[1],
        joint_3d_angle[2]
    ]

    # qvel:
    # velocity of the torso (3,)
    # angular velocity of the torso (3,)
    # joint angle velocities (8,)
    qvel = [
        qp.vel[0], qp.ang[0], joint_1d_vel[0], joint_2d_vel[0],
        joint_2d_vel[1], joint_3d_vel[0], joint_3d_vel[1],
        joint_3d_vel[2]
    ]

    # actuator forces
    torque_1d = take(action, self.sys.torque_1d.act_idx)
    torque_1d *= self.sys.torque_1d.strength
    torque_2d = take(action, self.sys.torque_2d.act_idx)
    torque_2d = torque_2d.reshape(torque_2d.shape[:-2] + (-1,))
    torque_2d *= jnp.repeat(self.sys.torque_2d.strength, 2)
    torque_3d = take(action, self.sys.torque_3d.act_idx)
    torque_3d = torque_3d.reshape(torque_3d.shape[:-2] + (-1,))
    torque_3d *= jnp.repeat(self.sys.torque_3d.strength, 3)
    qfrc_actuator = [torque_1d, torque_2d, torque_3d]

    # external contact forces:
    # delta velocity (3,), delta ang (3,) * num bodies in the system
    cfrc_ext = [info.contact.vel, info.contact.ang]
    # flatten bottom dimension
    cfrc_ext = [x.reshape(x.shape[:-2] + (-1,)) for x in cfrc_ext]

    # center of mass obs:
    body_pos = qp.pos[:-1]  # ignore floor at last index
    body_vel = qp.vel[:-1]  # ignore floor at last index

    com_vec = jnp.sum(body_pos * self.mass, axis=0) / jnp.sum(self.mass)
    com_vel = body_vel * self.mass / jnp.sum(self.mass)

    def v_outer(a):
      return jnp.outer(a, a)

    def v_cross(a, b):
      return jnp.cross(a, b)

    v_outer = jax.vmap(v_outer, in_axes=[0])
    v_cross = jax.vmap(v_cross, in_axes=[0, 0])

    disp_vec = body_pos - com_vec
    com_inert = self.inertia + self.mass.reshape(
        (11, 1, 1)) * ((jnp.linalg.norm(disp_vec, axis=1)**2.).reshape(
            (11, 1, 1)) * jnp.stack([jnp.eye(3)] * 11) - v_outer(disp_vec))

    cinert = [com_inert.reshape(-1)]

    square_disp = (1e-7 + (jnp.linalg.norm(disp_vec, axis=1)**2.)).reshape(
        (11, 1))
    com_angular_vel = (v_cross(disp_vec, body_vel) / square_disp)
    cvel = [com_vel.reshape(-1), com_angular_vel.reshape(-1)]

    return jnp.concatenate(qpos + qvel + cinert + cvel + qfrc_actuator +
                           cfrc_ext)

_SYSTEM_CONFIG = """
bodies {
  name: "torso"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.07
      length: 0.28
    }
  }
  colliders {
    position {
      z: 0.19
    }
    capsule {
      radius: 0.09
      length: 0.18
    }
  }
  colliders {
    position {
      x: -0.01
      z: -0.12
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 8.907463
}
bodies {
  name: "lwaist"
  colliders {
    position {
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.06
      length: 0.24
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 2.2619467
}
bodies {
  name: "pelvis"
  colliders {
    position {
      x: -0.02
    }
    rotation {
      x: -90.0
    }
    capsule {
      radius: 0.09
      length: 0.32
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 6.6161942
}
bodies {
  name: "right_thigh"
  colliders {
    position {
      y: 0.005
      z: -0.17
    }
    rotation {
      x: -178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "right_shin"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "left_thigh"
  colliders {
    position {
      y: -0.005
      z: -0.17
    }
    rotation {
      x: 178.31532
    }
    capsule {
      radius: 0.06
      length: 0.46014702
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.751751
}
bodies {
  name: "left_shin"
  colliders {
    position {
      z: -0.15
    }
    rotation {
      x: -180.0
    }
    capsule {
      radius: 0.049
      length: 0.398
      end: -1
    }
  }
  colliders {
    position {
      z: -0.35
    }
    capsule {
      radius: 0.075
      length: 0.15
      end: 1
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 4.5228419
}
bodies {
  name: "right_upper_arm"
  colliders {
    position {
      x: 0.08
      y: -0.08
      z: -0.08
    }
    rotation {
      x: 135.0
      y: 35.26439
      z: -75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "right_lower_arm"
  colliders {
    position {
      x: 0.09
      y: 0.09
      z: 0.09
    }
    rotation {
      x: -45.0
      y: 35.26439
      z: 15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: 0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
bodies {
  name: "left_upper_arm"
  colliders {
    position {
      x: 0.08
      y: 0.08
      z: -0.08
    }
    rotation {
      x: -135.0
      y: 35.26439
      z: 75.0
    }
    capsule {
      radius: 0.04
      length: 0.35712814
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.6610805
}
bodies {
  name: "left_lower_arm"
  colliders {
    position {
      x: 0.09
      y: -0.09
      z: 0.09
    }
    rotation {
      x: 45.0
      y: 35.26439
      z: -15.0
    }
    capsule {
      radius: 0.031
      length: 0.33912814
    }
  }
  colliders {
    position {
      x: 0.18
      y: -0.18
      z: 0.18
    }
    capsule {
      radius: 0.04
      length: 0.08
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.2295402
}
bodies {
  name: "floor"
  colliders {
    plane {
    }
  }
  inertia {
    x: 1.0
    y: 1.0
    z: 1.0
  }
  mass: 1.0
  frozen { all: true }
}
joints {
  name: "abdomen_z"
  stiffness: 15000.0
  parent: "torso"
  child: "lwaist"
  parent_offset {
    x: -0.01
    z: -0.195
  }
  child_offset {
    z: 0.065
  }
  rotation {
    y: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -45.0
    max: 45.0
  }
  angle_limit {
    min: -75.0
    max: 30.0
  }
}
joints {
  name: "abdomen_x"
  stiffness: 15000.0
  parent: "lwaist"
  child: "pelvis"
  parent_offset {
    z: -0.065
  }
  child_offset {
    z: 0.1
  }
  rotation {
    x: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -35.0
    max: 35.0
  }
}
joints {
  name: "right_hip_x"
  stiffness: 8000.0
  parent: "pelvis"
  child: "right_thigh"
  parent_offset {
    y: -0.1
    z: -0.04
  }
  child_offset {
  }
  rotation {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "right_knee"
  stiffness: 15000.0
  parent: "right_thigh"
  child: "right_shin"
  parent_offset {
    y: 0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "left_hip_x"
  stiffness: 8000.0
  parent: "pelvis"
  child: "left_thigh"
  parent_offset {
    y: 0.1
    z: -0.04
  }
  child_offset {
  }
  angular_damping: 20.0
  limit_strength: 2000.0
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -10.0
    max: 10.0
  }
  angle_limit {
    min: -30.0
    max: 70.0
  }
}
joints {
  name: "left_knee"
  stiffness: 15000.0
  parent: "left_thigh"
  child: "left_shin"
  parent_offset {
    y: -0.01
    z: -0.383
  }
  child_offset {
    z: 0.02
  }
  rotation {
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -160.0
    max: -2.0
  }
}
joints {
  name: "right_shoulder1"
  stiffness: 15000.0
  parent: "torso"
  child: "right_upper_arm"
  parent_offset {
    y: -0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 135.0
    y: 35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -85.0
    max: 60.0
  }
  angle_limit {
    min: -85.0
    max: 60.0
  }
}
joints {
  name: "right_elbow"
  stiffness: 15000.0
  parent: "right_upper_arm"
  child: "right_lower_arm"
  parent_offset {
    x: 0.18
    y: -0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 135.0
    z: 90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
joints {
  name: "left_shoulder1"
  stiffness: 15000.0
  parent: "torso"
  child: "left_upper_arm"
  parent_offset {
    y: 0.17
    z: 0.06
  }
  child_offset {
  }
  rotation {
    x: 45.0
    y: -35.26439
  }
  angular_damping: 20.0
  angle_limit {
    min: -60.0
    max: 85.0
  }
  angle_limit {
    min: -60.0
    max: 85.0
  }
}
joints {
  name: "left_elbow"
  stiffness: 15000.0
  parent: "left_upper_arm"
  child: "left_lower_arm"
  parent_offset {
    x: 0.18
    y: 0.18
    z: -0.18
  }
  child_offset {
  }
  rotation {
    x: 45.0
    z: -90.0
  }
  angular_damping: 20.0
  angle_limit {
    min: -90.0
    max: 50.0
  }
}
actuators {
  name: "abdomen_z"
  joint: "abdomen_z"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "abdomen_x"
  joint: "abdomen_x"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_hip_x"
  joint: "right_hip_x"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_knee"
  joint: "right_knee"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_hip_x"
  joint: "left_hip_x"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "left_knee"
  joint: "left_knee"
  strength: 300.0
  torque {
  }
}
actuators {
  name: "right_shoulder1"
  joint: "right_shoulder1"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "right_elbow"
  joint: "right_elbow"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_shoulder1"
  joint: "left_shoulder1"
  strength: 75.0
  torque {
  }
}
actuators {
  name: "left_elbow"
  joint: "left_elbow"
  strength: 75.0
  torque {
  }
}
collide_include {
  first: "floor"
  second: "left_shin"
}
collide_include {
  first: "floor"
  second: "right_shin"
}
friction: 1.0
gravity {
  z: -9.81
}
angular_damping: -0.05
baumgarte_erp: 0.1
dt: 0.015
substeps: 8
"""
