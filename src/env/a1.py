import pathlib
import time
import math

from gym.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np


path = pathlib.Path(__file__)

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}
A1_XML_PATH = path.parents[2].resolve().joinpath("robot_spec/xml/a1_mj210.xml").as_posix()
FRAME_SKIP = 5

class A1Env(MujocoEnv):
    """
    ### Description

    This environment is based on the environment model by unitree robotics,
    (https://github.com/unitreerobotics/unitree_mujoco).
    The A1 is a 3D robot consisting of one trunk (free body) with
    four legs attached to it with each leg having 3 bodies. The goal is to
    coordinate the four legs to move in any $the forward (right)$ direction by applying
    torques on the 12 hinge joints connecting the three bodies of each leg and the trunk
    (13 bodies and 12 hinge joints).

    ### Action Space
    The agent take a 12-element vector for actions.

    The action space is a continuous `(action, action, action, action, action, action,
    action, action, action, action, action, action)` with specific ranges as given in table below, 
    where `action` represents the numerical torques applied at the hinge joints.

    | Num | Action                                                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    |-----|--------------------------------------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the rotor between the trunk and front right hip  | -0.8        | 0.8         | FR_hip_joint                     | hinge | torque (N m) |
    | 1   | Torque applied on the rotor between the front right hip and thigh  | -1          | 4.2         | FR_thigh_joint                   | hinge | torque (N m) |
    | 2   | Torque applied on the rotor between the front right thigh and calf | -2.7        | -0.92       | FR_calf_joint                    | hinge | torque (N m) |
    | 3   | Torque applied on the rotor between the trunk and front left hip   | -8          | 8           | FL_hip_joint                     | hinge | torque (N m) |
    | 4   | Torque applied on the rotor between the front left hip and thigh   | -1          | 4.2         | FL_thigh_joint                   | hinge | torque (N m) |
    | 5   | Torque applied on the rotor between the front left thigh and calf  | -2.7        | -0.92       | FL_calf_joint                    | hinge | torque (N m) |
    | 6   | Torque applied on the rotor between the trunk and rear right hip   | -0.8        | 0.8         | RR_hip_joint                     | hinge | torque (N m) |
    | 7   | Torque applied on the rotor between the rear right hip and thigh   | -1          | 4.2         | RR_thigh_joint                   | hinge | torque (N m) |
    | 8   | Torque applied on the rotor between rear right thigh and calf      | -2.7        | -0.92       | RR_calf_joint                    | hinge | torque (N m) |
    | 9   | Torque applied on the rotor between the the trunk and rear left hip| -0.8        | 0.8         | RL_hip_joint                     | hinge | torque (N m) |
    | 10  | Torque applied on the rotor between the rear left hip and thigh    | -1          | 4.2         | RL_thigh_joint                   | hinge | torque (N m) |
    | 11  | Torque applied on the rotor between rear left thigh and calf       | -2.7        | -0.92       | RL_calf_joint                    | hinge | torque (N m) |

    ### Observation Space

    The state space consists of positional values of different body parts of the A1,
    followed by the velocities of those individual parts (their derivatives) with all
    the positions ordered before all the velocities.

    The observation is a `ndarray` with shape `(119,)` where the elements correspond to the following:

    | Num | Observation                                                         | $Min | Max$| Name (in corresponding XML file) | Joint | Unit                     |
    |-----|---------------------------------------------------------------------|------|-----|----------------------------------|-------|--------------------------|
    | 0   | x-coordinate of the trunk (centre)                                  | -Inf | Inf | trunk                            | free  | position (m)             |
    | 1   | y-coordinate of the trunk (centre)                                  | -Inf | Inf | trunk                            | free  | position (m)             |
    | 2   | z-coordinate of the trunk (centre)                                  | -Inf | Inf | trunk                            | free  | position (m)             |
    | 3   | x-orientation of the trunk (centre)                                 | -Inf | Inf | trunk                            | free  | angle (rad)              |
    | 4   | y-orientation of the trunk (centre)                                 | -Inf | Inf | trunk                            | free  | angle (rad)              |
    | 5   | z-orientation of the trunk (centre)                                 | -Inf | Inf | trunk                            | free  | angle (rad)              |
    | 6   | w-orientation of the trunk (centre)                                 | -Inf | Inf | trunk                            | free  | angle (rad)              |
    | 7   | angle between trunk and hip front right                             | -Inf | Inf | FR_hip                           | hinge | angle (rad)              |
    | 8   | angle between hip and thigh on the front right                      | -Inf | Inf | FR_thigh                         | hinge | angle (rad)              |
    | 9   | angle between thigh and calf on front right                         | -Inf | Inf | FR_calf                          | hinge | angle (rad)              |
    | 10  | angle between trunk and hip front left                              | -Inf | Inf | FL_hip                           | hinge | angle (rad)              |
    | 11  | angle between hip and thigh on the front left                       | -Inf | Inf | FL_thigh                         | hinge | angle (rad)              |
    | 12  | angle between thigh and calf on front left                          | -Inf | Inf | FL_calf                          | hinge | angle (rad)              |
    | 13  | angle between trunk and hip rear right                              | -Inf | Inf | RR_hip                           | hinge | angle (rad)              |
    | 14  | angle between hip and thigh on the rear right                       | -Inf | Inf | RR_thigh                         | hinge | angle (rad)              |
    | 15  | angle between thigh and calf on rear right                          | -Inf | Inf | RR_calf                          | hinge | angle (rad)              |
    | 16  | angle between trunk and hip rear left                               | -Inf | Inf | RL_hip                           | hinge | angle (rad)              |
    | 17  | angle between hip and thigh on the rear left                        | -Inf | Inf | RL_thigh                         | hinge | angle (rad)              |
    | 18  | angle between thigh and calf on rear left                           | -Inf | Inf | RL_calf                          | hinge | angle (rad)              |    
    | 19  | x-coordinate velocity of the trunk                                  | -Inf | Inf | trunk                            | free  | velocity (m/s)           |
    | 20  | y-coordinate velocity of the trunk                                  | -Inf | Inf | trunk                            | free  | velocity (m/s)           |
    | 21  | z-coordinate velocity of the trunk                                  | -Inf | Inf | trunk                            | free  | velocity (m/s)           |
    | 22  | x-coordinate angular velocity of the trunk                          | -Inf | Inf | trunk                            | free  | angular velocity (rad/s) |
    | 23  | y-coordinate angular velocity of the trunk                          | -Inf | Inf | trunk                            | free  | angular velocity (rad/s) |
    | 24  | z-coordinate angular velocity of the trunk                          | -Inf | Inf | trunk                            | free  | angular velocity (rad/s) |
    | 25  | angular velocity of angle between trunk and hip front right         | -Inf | Inf | FR_hip                           | hinge | angular velocity (rad/s) |
    | 26  | angular velocity of angle between hip and thigh on the front right  | -Inf | Inf | FR_thigh                         | hinge | angular velocity (rad/s) |
    | 27  | angular velocity of angle between thigh and calf on front right     | -Inf | Inf | FR_calf                          | hinge | angular velocity (rad/s) |
    | 28  | angular velocity of angle between trunk and hip front left          | -Inf | Inf | FL_hip                           | hinge | angular velocity (rad/s) |
    | 29  | angular velocity of angle between hip and thigh on the front left   | -Inf | Inf | FL_thigh                         | hinge | angular velocity (rad/s) |
    | 30  | angular velocity of angle between thigh and calf on front left      | -Inf | Inf | FL_calf                          | hinge | angular velocity (rad/s) |
    | 31  | angular velocity of angle between trunk and hip rear right          | -Inf | Inf | RR_hip                           | hinge | angular velocity (rad/s) |
    | 32  | angular velocity of angle between hip and thigh on the rear right   | -Inf | Inf | RR_thigh                         | hinge | angular velocity (rad/s) |
    | 33  | angular velocity of angle between thigh and calf on rear right      | -Inf | Inf | RR_calf                          | hinge | angular velocity (rad/s) |
    | 34  | angular velocity of angle between trunk and hip rear left           | -Inf | Inf | RL_hip                           | hinge | angular velocity (rad/s) |
    | 35  | angular velocity of angle between hip and thigh on the rear left    | -Inf | Inf | RL_thigh                         | hinge | angular velocity (rad/s) |
    | 36  | angular velocity of angle between thigh and calf on the rear left   | -Inf | Inf | RL_calf                          | hinge | angular velocity (rad/s) |


    The remaining 14*6 = 84 elements in the state are contact forces
    (external forces - force x, y, z and torque x, y, z) applied to the
    center of mass of each of the bodies. The 14 bodies are: the word body,
    the trunk body, and 3 bodies for each leg (1 + 1 + 12) with the 6 external forces.

    The (x,y,z) coordinates are translational DOFs while the orientations are rotational
    DOFs expressed as quaternions. One can read more about free joints on the [Mujoco Documentation]
    (https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    **Note:** There are 37 elements in the table above - giving rise to `(121,)` elements
    in the state space. In practice (and Gym implementation), the first two positional
    elements are omitted from the state space since the reward function is calculated based
    on the x-coordinate value. This value is hidden from the algorithm, which in turn has to
    develop an abstract understanding of it from the observed rewards. Therefore, observation
    space has shape `(119,)` instead of `(121,)` and the table should not have the first two rows.

    $**Note:** Ant-v4 environment no longer has the following contact forces issue.
    If using previous Ant versions from v4, there have been reported issues that using a Mujoco-Py version > 2.0 results
    in the contact forces always being 0. As such we recommend to use a Mujoco-Py version < 2.0
    when using the Ant environment if you would like to report results with contact forces (if
    contact forces are not used in your experiments, you can use version > 2.0).

    **Note:** Ant-v4 has the option of including contact forces in the observation space. To add contact forces set the argument
    'use_contact_forces" to True. The default value is False. Also note that training including contact forces can perform worse
    than not using them as shown in (https://github.com/openai/gym/pull/2762).$

    ### Rewards
    The reward consists of three parts:
    - *survive_reward*: Every timestep that the ant is alive, it gets a reward of 1.
    - *forward_reward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the time
    between actions and is dependent on the frame_skip parameter (default is 5),
    where the *dt* for one frame is 0.01 - making the default *dt = 5 * 0.01 = 0.05*.
    This reward would be positive if the ant moves forward (right) desired.
    - *ctrl_cost*: A negative reward for penalising the ant if it takes actions
    that are too large. It is measured as *coefficient **x** sum(action<sup>2</sup>)*
    where *coefficient* is a parameter set for the control and has a default value of 0.5.
    - *contact_cost*: A negative reward for penalising the ant if the external contact
    force is too large. It is calculated *0.5 * 0.001 * sum(clip(external contact
    force to [-1,1])<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *alive survive_reward + forward_reward - ctrl_cost - contact_cost*

    ### Starting State
    All observations start in state
    (0.0, 0.0,  0.75, 1.0, 0.0  ... 0.0) with a uniform noise in the range
    of [-0.1, 0.1] added to the positional values and standard normal noise
    with 0 mean and 0.1 standard deviation added to the velocity values for
    stochasticity. Note that the initial z coordinate is intentionally selected
    to be slightly high, thereby indicating a standing up ant. The initial orientation
    is designed to make it face forward as well.

    ### Episode Termination
    The episode terminates when any of the following happens:

    1. The episode duration reaches a 1000 timesteps
    2. Any of the state space values is no longer finite
    3. The z-position (index 2) in the state is **not** in the range `[0.2, 1.0]` or
        z-orientation (index 5) is **not** in range `[-0.2, 0.2]`
    ``
    """
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 20,
    }
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=True,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        #utils.EzPickle.__init__(**locals())

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        self._contact_force_range = contact_force_range

        self._reset_noise_scale = reset_noise_scale

        self._use_contact_forces = use_contact_forces

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(self, xml_file, FRAME_SKIP)

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        # self.data.cfrc_ext : com-based external force on body, (nbody x 6)
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        min_z_orient, max_z_orient = (-0.2, 0.2)
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z and min_z_orient <= state[5] <= max_z_orient
        return is_healthy

    @property
    def done(self):
        done = not self.is_healthy if self._terminate_when_unhealthy else False
        return done

    def step(self, action):
        xy_position_before = self.get_body_com("trunk")[:2].copy()
        
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        
        xy_position_after = self.get_body_com("trunk")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        done = self.done
        
        # print(observation.shape)
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        return observation, reward, done, info

    def _get_reward(self):
        pass

    def get_body_com_xquat(self, body_name):
        if self._mujoco_bindings.__name__ == "mujoco_py":
            return self.data.get_body_xquat(body_name)
        else:
            return self.data.body(body_name).xquat

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        # print("positions in A1Env._get_obs(), shape : ", position.shape)
        velocity = self.data.qvel.flat.copy()
        # print("velocities in A1Env._get_obs(), shape : ", velocity.shape)

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            # print("Contact force in A1Env._get_obs(), shape : ", contact_force.shape, contact_force)
            return np.concatenate((position, velocity, contact_force))
        else:
            return np.concatenate((position, velocity))

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)

def r2d(rad):
    """To convert radian to degree"""
    return math.degrees(rad)

def d2r(deg):
    """To convert degree to radian"""
    return math.radians(deg)

if __name__ == "__main__":
    env = A1Env(A1_XML_PATH, FRAME_SKIP)
    for _ in range(500):
        try:
            env.render()
            time.sleep(0.05)
            observation, reward, done, _ = env.step(env.action_space.sample())

            if done:
                exit(0)
        except KeyboardInterrupt:
            env.close()
