import os
from stable_baselines import PPO2, DDPG

import warnings
warnings.filterwarnings("ignore")


def get_dir_root():
    return os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]


def get_policy(name="ddpg"):
    """
    Note: ppo requires the NeuralShield package in the docker.
    :param name: pretrained policy name
    :return: stable baselines policy
    """
    if name == "ppo":
        return PPO2.load(get_dir_root() + "/pretrained/ppo.pkl")
    elif name == "ddpg":
        return DDPG.load(get_dir_root() + "/pretrained/ddpg.pkl")
