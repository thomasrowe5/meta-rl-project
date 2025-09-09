import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Example environment settings
state_size = 10
action_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- WORLD MODEL ----
class WorldModel(nn.Module):
    def __init__(self, state_size, hidden_size):
        super(WorldModel, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, state_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ---- POLICY NETWORK ----
class Policy(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

# Dummy environment step
def env_step(states, actions):
    noise = torch.randn_like(states) * 0.1
    return states + actions @ torch.randn(action_size, state_size) + noise

# Example bonus functions
def exploration_bonus(states):
    return torch.rand(states.size(0), device=states.device)

def interaction_bonus(states, actions):
    return torch.rand(states.size(0), device=states.device)

# ---- INITIALIZE ----
world_model = WorldModel(state_size, hidden_size=64).to(device)
policy = Policy(state_size, hidden_size=64, action_size=action_size).to(device)

optimizer_wm = optim.Adam(world_model.parameters(), lr=1e-3)
optimizer_policy = optim.Adam(policy.parameters(), lr=1e-3)

# Initial states
batch_size = 32
states = torch.randn(batch_size, state_size, device=device)

# ---- TRAIN LOOP ----
for episode in range(100):
    total_loss = 0

    for step in range(10):
        # Policy picks actions
        action_probs = policy(states)
        dist = torch.distributions.Categorical(action_probs)
        actions = dist.sample()
        one_hot_actions = F.one_hot(actions, num_classes=action_size).float()

        # Step environment
        next_states = env_step(states, one_hot_actions)

        # World model predicts next state
        predicted_next_states = world_model(states)

        # Prediction loss
        pred_loss = F.mse_loss(predicted_next_states, next_states)

        # Exploration & interaction bonuses
        exp_bonus = exploration_bonus(states).mean()
        int_bonus = interaction_bonus(states, one_hot_actions).mean()

        # World model total loss
        wm_loss = pred_loss - 0.01 * exp_bonus - 0.01 * int_bonus

        optimizer_wm.zero_grad()
        wm_loss.backward(retain_graph=True)
        optimizer_wm.step()

        # Policy update (REINFORCE)
        reward = exp_bonus + int_bonus
        log_probs = dist.log_prob(actions)
        policy_loss = -(log_probs * reward.detach()).mean()  # detach reward from WM graph

        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # Move to next state
        states = next_states.detach()
        total_loss += wm_loss.item()

    if episode % 10 == 0:
        print(f"Episode {episode}, World Model Loss: {total_loss:.4f}")

