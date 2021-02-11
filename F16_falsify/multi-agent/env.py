from typing import Dict, List, Tuple

import gym
import ray
import CoRec

from ray.rllib import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, AgentID


class MultiAgentF16GCAS(MultiAgentEnv):
    def __init__(self, plant_num=2):
        self.plant_dict = {f"plant_{i}": gym.make("F16GCAS-v3") for i in range(plant_num)}
        for k in self.plant_dict:
            self.plant_dict[k].initial_space.low[10] += 200 * int(k.split("_")[-1])
            self.plant_dict[k].initial_space.high[10] += 200 * int(k.split("_")[-1])

    def reset(self) -> MultiAgentDict:
        obs_dict = {}
        for k in self.plant_dict:
            obs_dict[k] = self.plant_dict[k].reset()

        return obs_dict

    def step(self, action_dict: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for k in action_dict:
            obs[k], rewards[k], dones[k], infos[k] = self.plant_dict[k].step(action_dict[k])

        return obs, rewards, dones, infos

    def get_all_states(self) -> MultiAgentDict:
        return {k: env.states for k, env in self.plant_dict.items()}


if __name__ == '__main__':
    env = MultiAgentF16GCAS()
    print(env.reset())
    print(env.step({"plant_0": env.plant_dict["plant_0"].action_space.sample(),
                    "plant_1": env.plant_dict["plant_1"].action_space.sample()}))
    print(env.get_all_states())

    for _ in range(1000):
        env.step({"plant_0": env.plant_dict["plant_0"].action_space.sample(),
                  "plant_1": env.plant_dict["plant_1"].action_space.sample()})

    from plot import multi_agent_plot3d, multi_agent_plot3d_anim

    # multi_agent_plot3d(env)
    multi_agent_plot3d_anim(env, skip=20, filename="multi_agent_f16.gif")
