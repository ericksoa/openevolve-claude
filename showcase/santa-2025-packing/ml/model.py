"""
Value Function Model for Tree Packing

Predicts final side length from partial packing state.
Used as heuristic to guide search/beam search.
"""

import torch
import torch.nn as nn


class PackingValueNet(nn.Module):
    """
    MLP that predicts final side length from packing state.

    Input: [n_trees_norm, x1, y1, rot1, x2, y2, rot2, ..., target_n]
    Output: predicted final side length
    """

    def __init__(self, max_n: int = 50, hidden_dims: list = [256, 128, 64]):
        super().__init__()

        self.max_n = max_n
        # Input: max_n * 3 (positions) + 1 (n_trees) + 1 (target_n)
        input_dim = max_n * 3 + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, max_n * 3 + 1] - packing state
            target_n: [batch, 1] - normalized target n

        Returns:
            predicted_side: [batch, 1]
        """
        x = torch.cat([features, target_n], dim=-1)
        return self.network(x)


class PackingValueNetWithAttention(nn.Module):
    """
    More sophisticated model using attention over placed trees.
    Better for variable-length inputs.
    """

    def __init__(self, max_n: int = 50, embed_dim: int = 64, num_heads: int = 4):
        super().__init__()

        self.max_n = max_n
        self.embed_dim = embed_dim

        # Embed each tree (x, y, rot) -> embed_dim
        self.tree_embed = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Positional encoding for tree order
        self.pos_embed = nn.Parameter(torch.randn(1, max_n, embed_dim) * 0.02)

        # Self-attention over trees
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # Final prediction
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim + 2, 128),  # +2 for n_trees and target_n
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, features: torch.Tensor, target_n: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch, max_n * 3 + 1]
            target_n: [batch, 1]
        """
        batch_size = features.shape[0]

        # Extract n_trees and tree features
        n_trees_norm = features[:, 0:1]  # [batch, 1]
        tree_features = features[:, 1:].view(batch_size, self.max_n, 3)  # [batch, max_n, 3]

        # Embed trees
        tree_embeds = self.tree_embed(tree_features)  # [batch, max_n, embed_dim]
        tree_embeds = tree_embeds + self.pos_embed

        # Create mask for padded trees (where all features are 0)
        mask = (tree_features.abs().sum(dim=-1) == 0)  # [batch, max_n]

        # Self-attention
        attended, _ = self.attention(
            tree_embeds, tree_embeds, tree_embeds,
            key_padding_mask=mask
        )

        # Pool over trees (mean of non-padded)
        mask_expanded = (~mask).unsqueeze(-1).float()
        pooled = (attended * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)

        # Predict
        combined = torch.cat([pooled, n_trees_norm, target_n], dim=-1)
        return self.predictor(combined)


def create_model(model_type: str = "mlp", max_n: int = 50) -> nn.Module:
    """Factory function for creating models."""
    if model_type == "mlp":
        return PackingValueNet(max_n=max_n)
    elif model_type == "attention":
        return PackingValueNetWithAttention(max_n=max_n)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
