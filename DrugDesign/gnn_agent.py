import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.distributions import Categorical

class GNNActorCritic(nn.Module):
    def __init__(self, num_node_features, num_actions, hidden_dim=64):
        super(GNNActorCritic, self).__init__()
        
        # === 1. Graph Encoder (The "Eyes") ===
        # We use Graph Convolutional Networks (GCN) to process the molecule
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # === 2. Shared Body ===
        # After pooling the graph into a single vector, we process it here
        self.shared_fc = nn.Linear(hidden_dim, hidden_dim)

        # === 3. Actor Head (The Policy) ===
        # Decides which action to take (0-10)
        self.actor_head = nn.Linear(hidden_dim, num_actions)

        # === 4. Critic Head (The Value) ===
        # Estimates "How good is this molecule?" (Scalar output)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch_vector, action_mask=None):
        """
        Args:
            x: Node features [Total_Atoms, Num_Features]
            edge_index: Graph connectivity [2, Total_Edges]
            batch_vector: Tells us which atom belongs to which molecule [Total_Atoms]
            action_mask: Boolean mask [Batch_Size, Num_Actions]
        """
        
        # 1. GNN Layers (Message Passing)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # 2. Global Pooling (Readout)
        # Aggregates all atoms in a molecule into a single vector
        # x shape becomes: [Batch_Size, Hidden_Dim]
        x = global_mean_pool(x, batch_vector)
        
        # 3. Shared Processing
        x = F.relu(self.shared_fc(x))

        # 4. Heads
        logits = self.actor_head(x)
        state_value = self.critic_head(x)

        # 5. Apply Action Masking
        if action_mask is not None:
            # Create a huge negative number for invalid actions
            huge_neg = torch.tensor(-1e8, device=x.device, dtype=x.dtype)
            # Use where: if mask is True keep logit, else replace with -1e8
            logits = torch.where(action_mask.bool(), logits, huge_neg)

        return logits, state_value

    def get_action(self, x, edge_index, batch_vector, action_mask=None):
        """
        Helper to sample an action during training
        """
        logits, value = self.forward(x, edge_index, batch_vector, action_mask)
        
        # Create probability distribution
        probs = Categorical(logits=logits)
        
        # Sample action
        action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value