from envs.gridworld import GridWorld
from models.model import WorldModel, Policy
import torch

env = GridWorld()
state = torch.tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
print("Test environment reset. State shape:", state.shape)

wm = WorldModel(25, 4)
policy = Policy(25, 4)
print("Test models loaded!")

