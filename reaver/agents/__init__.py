from .base import *
from .random import RandomAgent
from .a2c import AdvantageActorCriticAgent
from .ppo import ProximalPolicyOptimizationAgent
from .testagent import RandomActorCriticAgent

A2C = AdvantageActorCriticAgent
PPO = ProximalPolicyOptimizationAgent
testagent = RandomActorCriticAgent
