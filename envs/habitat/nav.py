from typing import Optional, Type
import numpy as np
import habitat
from habitat import Config, Dataset
import envs.utils.pose as pu
import quaternion

class NavRLEnv(habitat.RLEnv):
    def __init__(self, args, config: Config, dataset: Optional[Dataset] = None):
        self.args = args
        self._core_env_config = config
        self._reward_measure_name = "distance_to_goal"
        self._success_measure_name = "spl"
        self._slack_reward = -0.001
        self._success_reward = 2.5
        self._previous_measure = None
        self._previous_action = None
        self.last_sim_location = None
        super().__init__(self._core_env_config, dataset)
        self.stopped = None
        self.info = {}
        self.info['distance_to_goal'] = None
        self.info['spl'] = None
        self.info['success'] = None
        self.info['softspl'] = None
        self.timestep = None
        self.goal_dict =  {0: "chair", 1: "bed", 2: "plant", 3: "toilet", 4: "tv_monitor", 5: "sofa"}
        

    def reset(self):
        self.timestep = 0
        self.stopped = False
        self.info['sensor_pose'] = [0., 0., 0.]
        self._previous_action = None
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        self.last_sim_location = self.get_sim_location()
        
        rgb = observations["rgb"]
        depth = observations["depth"]
        sem = observations["semantic"]
        sem = np.expand_dims(sem, -1)
        objectgoal = observations["objectgoal"]
        self.info['goal_cat_id'] = int(objectgoal)
        self.info['goal_name'] = self.goal_dict[int(objectgoal)]

        compass = observations["compass"]
        gps = observations["gps"]
        obs = np.concatenate((rgb, depth, sem), axis=2).transpose(2, 0, 1)

        self.info['time'] = self.timestep

        return obs, self.info

    def step(self, *args, **kwargs):
        if len(kwargs) >= 1:
            self._previous_action = kwargs["action"]
        else:
            self._previous_action = None
        if args[0]['action'] == 0:
            self.stopped = True

        observations, rew, done, _ =  super().step(*args, **kwargs)
        rgb = observations["rgb"]
        depth = observations["depth"]
        sem = observations["semantic"]
        sem = np.expand_dims(sem, -1)
        obs = np.concatenate((rgb, depth, sem), axis=2).transpose(2, 0, 1)
        dx, dy, do = self.get_pose_change()
        self.info['sensor_pose'] = [dx, dy, do]

        if done:
            spl = self._env.get_metrics()[self._success_measure_name]
            dist = self._env.get_metrics()[self._reward_measure_name]
            softspl = self._env.get_metrics()["softspl"]
            success = self._env.get_metrics()["success"] 
            self.info["distance_to_goal"] = dist
            self.info["success"] = success
            self.info["spl"] = spl
            self.info["softspl"] = softspl
        
        self.timestep += 1
        self.info['time'] = self.timestep

        return obs, rew, done, self.info

    def get_reward_range(self):
        return (
            self._slack_reward - 1.0,
            self._success_reward + 1.0,
        )

    def get_reward(self, observations):
        reward = self._slack_reward

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        reward += (self._previous_measure - current_measure) * self.args.reward_coeff

        self._previous_measure = current_measure

        if self._episode_success():
            reward += self._success_reward

        return reward

    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def get_done(self, observations):
        done = False
        if self.info['time'] >= self.args.max_episode_length - 1:
            done = True
        # if self._env.episode_over or self._episode_success():
        #     done = True
        if self.stopped:
            done = True

        return done

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        return info

    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o
    
    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do

