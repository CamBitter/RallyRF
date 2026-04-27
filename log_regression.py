# logistic regression prediction for tennis matches
# for comparison purposes with random forest

# past project/basis: 
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from data import getDF, getPlayerStatLastYrAvg

# read in data:

# this is a truncated version since using the whole data was taking forever
def build_features(df):
    rows = []
    labels = []

    for _, match in df.iterrows():
        features = {
            "rank_diff":     (match["winner_rank"]        or 0) - (match["loser_rank"]        or 0),
            "rank_pts_diff": (match["winner_rank_points"] or 0) - (match["loser_rank_points"] or 0),
            "age_diff":      (match["winner_age"]         or 0) - (match["loser_age"]         or 0),
        }

        for col in ["match_type_main", "match_type_futures", "match_type_challenger"]:
            features[col] = match.get(col, 0)

        rows.append(features)
        labels.append(1)

    X_df = pd.DataFrame(rows).fillna(0)
    y    = np.array(labels, dtype=np.float32)

    flip = np.random.rand(len(y)) > 0.5
    diff_cols = ["rank_diff", "rank_pts_diff", "age_diff"]
    X_df.loc[flip, diff_cols] *= -1
    y[flip] = 0

    return X_df, y

    
# adapted from lecture 7: Assessment of Classifiers

def binary_cross_entropy(q, y, model, lambda_reg=0.001):
    loss = -(y * torch.log(q) + (1 - y) * torch.log(1 - q)).mean()
    return loss + lambda_reg * torch.sum(model.w ** 2) 

def sigmoid(z): 
    return 1 / (1 + torch.exp(-z))

# model class
class BinaryLogisticRegression: 
    def __init__(self, n_features): 
        self.w = torch.zeros(n_features, 1, requires_grad=True)

    def forward(self, X): 
        return sigmoid(X @ self.w)    

# optimizer class
class GradientDescentOptimizer: 
    def __init__(self, model, lr=0.1): 
        self.model = model
        self.lr = lr

    def grad_func(self, X, y): 
        q = self.model.forward(X)
        return 1/X.shape[0] * ((q - y).T @ X).T
        
    def step(self, X, y): 
        grad = self.grad_func(X, y)
        with torch.no_grad(): 
            self.model.w -= self.lr * grad
        
# copied from previous project, not ever running this on colab but not harmful to have
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"



# Main

if __name__ == "__main__":
    # was taking forever so i printed a lot
    print("getting data")

    df = getDF()
    print("data wrangled!")

    X_df, y = build_features(df)

    print("dataset built")

    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

    print("scaling features")

    scaler          = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled   = scaler.transform(X_test)

    device = get_device()
    print(f"Running on {device}")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train).unsqueeze(1).to(device)  # (n, 1)

    d_features = X_train_tensor.shape[1]
    model      = BinaryLogisticRegression(d_features)
    model.w    = model.w.to(device)
    opt        = GradientDescentOptimizer(model, lr=0.01)

    # training loop
    batch_size = 64
    n_samples  = X_train_tensor.shape[0]
    losses     = []

    # proof of concept run with very few epochs
    for epoch in range(50):
        perm            = torch.randperm(n_samples, device=device)
        X_train_tensor  = X_train_tensor[perm]
        y_train_tensor = y_train_tensor[perm]

        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n_samples, batch_size):
            X_batch = X_train_tensor[start:start + batch_size]
            y_batch = y_train_tensor[start:start + batch_size]

            loss        = binary_cross_entropy(model.forward(X_batch), y_batch, model)
            epoch_loss  += loss.item()
            num_batches += 1

            opt.step(X_batch, y_batch)

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/27500, Loss: {avg_loss:.4f}")

    # evaluate on test set
    # currently bad, could be improved for much longer run
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_probs = model.forward(X_test_tensor).squeeze()
        test_preds = (test_probs >= 0.5).float()
        accuracy   = (test_preds == y_test_tensor).float().mean().item()

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

