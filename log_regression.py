# logistic regression prediction for tennis matches
# for comparison purposes with random forest

import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from features import FEATURE_SETS

feature_set = "all_diffs.csv"
FEATURE_COLS = FEATURE_SETS[feature_set]

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
    print("getting data")

    df = pd.read_csv(f"data/cleaned/{feature_set}")
    train_df = df[df["tourney_date"] < 20220101]
    test_df  = df[df["tourney_date"] >= 20220101]

    X_train = train_df[FEATURE_COLS].to_numpy()
    Y_train = train_df["p1_won"].to_numpy()

    X_test  = test_df[FEATURE_COLS].to_numpy()
    Y_test  = test_df["p1_won"].to_numpy()

    print("scaling features")

    scaler          = StandardScaler()
    X_train_scaled  = scaler.fit_transform(X_train)
    X_test_scaled   = scaler.transform(X_test)

    device = get_device()
    print(f"Running on {device}")
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    Y_train_tensor = torch.tensor(Y_train).unsqueeze(1).to(device)  # (n, 1)

    d_features = X_train_tensor.shape[1]
    model      = BinaryLogisticRegression(d_features)
    model.w    = model.w.to(device)
    opt        = GradientDescentOptimizer(model, lr=0.0001)

    # training loop
    batch_size = 128
    n_samples  = X_train_tensor.shape[0]
    losses     = []
    n_epochs   = 200

    # proof of concept run with very few epochs
    for epoch in range(n_epochs):
        perm            = torch.randperm(n_samples, device=device)
        X_train_tensor  = X_train_tensor[perm]
        Y_train_tensor = Y_train_tensor[perm]

        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n_samples, batch_size):
            X_batch = X_train_tensor[start:start + batch_size]
            y_batch = Y_train_tensor[start:start + batch_size]

            loss        = binary_cross_entropy(model.forward(X_batch), y_batch, model)
            epoch_loss  += loss.item()
            num_batches += 1

            opt.step(X_batch, y_batch)

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    # evaluate on test set
    # currently bad, could be improved for much longer run
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(Y_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        test_probs = model.forward(X_test_tensor).squeeze()
        test_preds = (test_probs >= 0.5).float()
        accuracy   = (test_preds == y_test_tensor).float().mean().item()


    rank_baseline = (test_df["rank_diff"] < 0).astype(float).to_numpy()
    seed_accuracy = (rank_baseline == Y_test).mean()

    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    print(f"seed baseline: {seed_accuracy * 100:.2f}%")

