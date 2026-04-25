# logistic regression prediction for tennis matches
# for comparison purposes with random forest

# past project/basis: 
import pickle
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Pipeline

class DataPrepPipeline:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=50000, min_df=5)
        self.one_hot_encoder  = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler           = StandardScaler()

    def fit(self, X, y=None):
        self.tfidf_vectorizer.fit(X['lyrics'])
        self.one_hot_encoder.fit(X[['topic']])
        self.numeric_cols = [c for c in X.columns if c not in ['lyrics', 'topic', 'track_name']]
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        if 'genre' in X_transformed.columns:
            X_transformed = X_transformed.drop(columns=['genre'])

        # scale numeric columns first, then engineer features from scaled values
        X_transformed[self.numeric_cols] = self.scaler.transform(X_transformed[self.numeric_cols])

        # engineered features
        X_transformed['hype']                = (X_transformed['danceability'] * X_transformed['energy']) + X_transformed['loudness']
        X_transformed['emotionness']         = (X_transformed['sadness'] + X_transformed['dating']) * X_transformed['feelings']
        X_transformed['intensity']           = X_transformed['violence'] + X_transformed['obscene'] + X_transformed['shake the audience']
        X_transformed['spirituality']        = X_transformed['family/gospel'] + X_transformed['family/spiritual']
        X_transformed['contrast']            = X_transformed['romantic'] - X_transformed['violence']
        X_transformed['night-energy']        = X_transformed['night/time'] * X_transformed['danceability']
        X_transformed['vibe']                = X_transformed['valence'] * X_transformed['energy']
        X_transformed['performance']         = X_transformed['shake the audience'] + X_transformed['music'] + X_transformed['movement/places']
        X_transformed['softness']            = X_transformed['acousticness'] * (1 - X_transformed['energy'])
        X_transformed['chill']               = X_transformed['romantic'] * X_transformed['acousticness']
        X_transformed['vintage']             = X_transformed['age'] * X_transformed['acousticness']
        X_transformed['modern']              = (1 - X_transformed['age']) * X_transformed['energy']
        X_transformed['dense_lyrics']        = X_transformed['len'] * (1 - X_transformed['instrumentalness'])

        # text and categorical features
        lyrics_tfidf = self.tfidf_vectorizer.transform(X_transformed['lyrics'])
        tfidf_df = pd.DataFrame(
            lyrics_tfidf.toarray(),
            columns=self.tfidf_vectorizer.get_feature_names_out(),
            index=X_transformed.index
        )
        topic_encoded = self.one_hot_encoder.transform(X_transformed[['topic']])
        topic_df = pd.DataFrame(
            topic_encoded,
            columns=self.one_hot_encoder.get_feature_names_out(['topic']),
            index=X_transformed.index
        )

        X_transformed = X_transformed.drop(columns=['lyrics', 'topic', 'track_name'])
        return pd.concat([X_transformed, tfidf_df, topic_df], axis=1)

# Model
    
# adapted from lecture 9: Multinomial Classification

def softmax_rows(S, dim=-1):
    exp_S = torch.exp(S)
    return exp_S / torch.sum(exp_S, dim=dim, keepdim=True)

def cross_entropy_loss(Q, Y, model, lambda_reg=0.001):
    Q_clamped  = torch.clamp(Q, min=1e-7)
    ce_loss    = -torch.mean(Y * torch.log(Q_clamped))
    l2_penalty = torch.sum(model.W ** 2)
    return ce_loss + lambda_reg * l2_penalty

class GenreModel:
    def __init__(self, d_features, k_classes):
        self.W = torch.randn(d_features, k_classes) * 0.01

    def forward(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.values, dtype=torch.float32)
        return softmax_rows(X @ self.W, dim=1)

class GradientDescentOptimizer:
    def __init__(self, model, learning_rate=0.01):
        self.model         = model
        self.learning_rate = learning_rate

    def step(self, X, y):
        self.model.W -= self.learning_rate * self.grad_func(X, y)

    def grad_func(self, X, y):
        q = self.model.forward(X)
        return X.T @ (q - y) / X.shape[0]

# Main

if __name__ == "__main__":

    # load data
    url   = "https://middcs.github.io/csci-0451-s26/data/music-genre/train.csv"
    train = pd.read_csv(url)

    X_df = train.drop(columns=["genre"])
    y_df = train["genre"]

    # split for local evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

    pipeline = DataPrepPipeline()
    pipeline.fit(X_train)
    with open("pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    X_train_processed = pipeline.transform(X_train)
    X_test_processed  = pipeline.transform(X_test)

    # fit and save the data prep pipeline on FULL dataset
    pipeline = DataPrepPipeline()
    pipeline.fit(X_df)
    with open("pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    # apply pipeline to full dataset
    X_train_processed = pipeline.transform(X_df)

    # device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # encode labels on full dataset
    label_encoder   = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_df)
    num_genres      = len(label_encoder.classes_)

    # build tensors
    y_train_one_hot = F.one_hot(torch.tensor(y_train_encoded), num_classes=num_genres).float().to(device)
    X_train_tensor  = torch.tensor(X_train_processed.values, dtype=torch.float32).to(device)


    torch.manual_seed(42)
    d_features = X_train_processed.shape[1]
    model      = GenreModel(d_features=d_features, k_classes=num_genres)
    model.W    = model.W.to(device)
    opt        = GradientDescentOptimizer(model, learning_rate=0.0005)

    # training loop
    batch_size = 64
    n_samples  = X_train_tensor.shape[0]
    losses     = []

    for epoch in range(25000):
        perm            = torch.randperm(n_samples, device=device)
        X_train_tensor  = X_train_tensor[perm]
        y_train_one_hot = y_train_one_hot[perm]

        epoch_loss, num_batches = 0.0, 0

        for start in range(0, n_samples, batch_size):
            X_batch = X_train_tensor[start:start + batch_size]
            y_batch = y_train_one_hot[start:start + batch_size]

            loss        = cross_entropy_loss(model.forward(X_batch), y_batch, model)
            epoch_loss  += loss.item()
            num_batches += 1

            opt.step(X_batch, y_batch)

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch + 1}/27500, Loss: {avg_loss:.4f}")

    print("Training complete.")
    print(f"Final Loss: {losses[-1]:.4f}")

    # save model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)