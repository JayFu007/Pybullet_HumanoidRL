import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network model for reinforcement learning.
    Combines a policy network (actor) and a value network (critic).
    """
    def __init__(self, state_dim, action_dim):
        """
        Initializes the ActorCritic model.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
        """
        super(ActorCritic, self).__init__()
        # Actor network to predict the mean of the action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, 256),  # Hidden layer with 256 units
            nn.Tanh(),
            nn.Linear(256, 128),       # Hidden layer with 128 units
            nn.Tanh(),
            nn.Linear(128, action_dim),  # Output layer for action mean
            nn.Tanh()  # Ensures actions are in the range [-1, 1]
        )
        # Learnable parameter for the log standard deviation of the action distribution
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        # Critic network to predict the state value
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # Output layer for state value
        )

    def forward(self, state):
        """
        Forward pass through the ActorCritic model.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            tuple: Action mean, action standard deviation, and state value.
        """
        action_mean = self.actor_mean(state)
        action_std = self.actor_log_std.exp()  # Convert log std to std
        state_value = self.critic(state)
        return action_mean, action_std, state_value


class PPO:
    """
    Proximal Policy Optimization (PPO) algorithm implementation.
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, buffer_size=10000):
        """
        Initializes the PPO agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor for rewards.
            eps_clip (float): Clipping parameter for PPO.
            k_epochs (int): Number of epochs for policy updates.
            buffer_size (int): Size of the preallocated buffer for transitions.
        """
        # Check device availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on device: {self.device}")

        # Initialize the policy network
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.mse_loss = nn.MSELoss()  # Loss function for the critic

        # Symmetry constraint matrix
        self.symmetry_matrix = np.eye(action_dim)  # Initialize symmetry constraint matrix as identity matrix

        # Preallocated buffer for transitions
        self.buffer_size = buffer_size
        self.buffer_index = 0
        self.states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(buffer_size, dtype=torch.bool, device=self.device)

    def set_symmetry_matrix(self, symmetry_matrix):
        """
        Sets the symmetry constraint matrix.

        Args:
            symmetry_matrix (np.ndarray): Symmetry constraint matrix, shape (action_dim, action_dim).
        """
        self.symmetry_matrix = symmetry_matrix

    def select_action(self, state):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            np.ndarray: Selected action.
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_mean, action_std, _ = self.policy(state)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()

        # Clamp actions to the range [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)

        # Apply symmetry constraint
        symmetry_matrix = torch.FloatTensor(self.symmetry_matrix).to(self.device)
        action = torch.matmul(symmetry_matrix, action.T).T  # 修复形状不匹配问题

        log_prob = dist.log_prob(action).sum(dim=-1)
        self.log_probs[self.buffer_index] = log_prob
        return action.cpu().detach().numpy().flatten()

    def store_transition(self, transition):
        """
        Stores a transition in the preallocated buffer.

        Args:
            transition (tuple): A tuple containing (state, action, reward, done, info).
        """
        state, action, reward, done, info = transition
        combined_reward = reward + info.get("visualize_reward", 0)

        # Store data in the preallocated tensors
        self.states[self.buffer_index] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.buffer_index] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.buffer_index] = combined_reward
        self.dones[self.buffer_index] = done
        self.log_probs[self.buffer_index] = 0  # Initialize log_prob to 0
        self.buffer_index += 1

        # Raise an exception if the buffer overflows
        if self.buffer_index >= self.buffer_size:
            raise RuntimeError("Buffer overflow: Increase buffer_size or process transitions more frequently.")

    def _compute_discounted_rewards(self):
        """
        Computes discounted rewards for the stored transitions.

        Returns:
            torch.Tensor: Discounted rewards.
        """
        discounted_rewards = torch.zeros_like(self.rewards, device=self.device)
        cumulative_reward = 0
        for i in reversed(range(self.buffer_index)):
            if self.dones[i]:
                cumulative_reward = 0
            cumulative_reward = self.rewards[i] + self.gamma * cumulative_reward
            discounted_rewards[i] = cumulative_reward

        # Normalize rewards
        mean = discounted_rewards[:self.buffer_index].mean()
        std = discounted_rewards[:self.buffer_index].std() + 1e-8
        discounted_rewards[:self.buffer_index] = (discounted_rewards[:self.buffer_index] - mean) / std

        return discounted_rewards[:self.buffer_index]

    def train(self):
        """
        Trains the PPO agent using the stored transitions.
        """
        states = self.states[:self.buffer_index]
        actions = self.actions[:self.buffer_index]
        discounted_rewards = self._compute_discounted_rewards()
        old_log_probs = self.log_probs[:self.buffer_index]

        for epoch in range(self.k_epochs):
            action_means, action_stds, state_values = self.policy(states)
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # Compute advantage
            advantages = discounted_rewards - state_values.squeeze().detach()

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute PPO loss
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.mse_loss(state_values.squeeze(), discounted_rewards)
            loss = actor_loss + 0.5 * critic_loss

            # Optimize the policy
            self.optimizer.zero_grad()
            # loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.buffer_index = 0