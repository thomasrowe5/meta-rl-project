import matplotlib.pyplot as plt
from envs.gridworld import GridWorld

env = GridWorld()
state = env.reset().reshape(5,5)

plt.imshow(state, cmap='Greys', interpolation='none')
plt.title("GridWorld Initial State")
plt.show()

