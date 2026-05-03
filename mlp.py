import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset

FEATURE_COLS = [
    "rank_diff",
    "rank_pts_diff",
    "age_diff",
    "height_diff",
    "ace_vs_df_diff",
    "first_in_diff",
    "first_won_diff",
    "second_won_diff",
    "bp_converted_pct_diff",
    "win_pct_diff",
    "games_played_diff",
]

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 50),
            nn.ReLU(),
            nn.Linear(50, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    df = pd.read_csv("data/cleaned/atp_match_features_2*.csv")

    train_df = df[df["tourney_date"] < 20220101]
    test_df  = df[df["tourney_date"] >= 20220101]

    X_train = torch.tensor(train_df[FEATURE_COLS].to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(train_df["p1_won"].to_numpy(), dtype=torch.float32)
    X_test  = torch.tensor(test_df[FEATURE_COLS].to_numpy(), dtype=torch.float32)
    y_test  = torch.tensor(test_df["p1_won"].to_numpy(), dtype=torch.float32)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    dataset   = TensorDataset(X_train, y_train)
    loader    = DataLoader(dataset, batch_size=256, shuffle=True)
    model     = MLP(input_size=len(FEATURE_COLS), hidden_size=128, output_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn   = nn.BCELoss()

    for epoch in range(20):
        total_loss = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(X_batch).squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                acc = ((model(X_test).squeeze() >= 0.5).float() == y_test).float().mean().item()
            print(f"Epoch {epoch:>3} | loss: {total_loss / len(loader):.4f} | test acc: {acc:.4f}")

    model.eval()
    with torch.no_grad():
        final_acc = ((model(X_test).squeeze() >= 0.5).float() == y_test).float().mean().item()
    print(f"\nFinal test accuracy: {final_acc:.4f}")

    rank_baseline = (test_df["rank_diff"] < 0).astype(int).to_numpy()
    print(f"Rank baseline:       {(rank_baseline == test_df['p1_won'].to_numpy()).mean():.4f}")
