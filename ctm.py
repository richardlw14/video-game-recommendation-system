import pandas as pd
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import spacy
from gensim.models import LdaMulticore, TfidfModel
from gensim.corpora import Dictionary
from tqdm import trange
import multiprocessing

class CTR:
    def __init__(self, epochs=200, learning_rate=1e-5, sigma2=10, sigma2_P=50, sigma2_Q=50):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sigma2 = sigma2
        self.sigma2_P = sigma2_P
        self.sigma2_Q = sigma2_Q

    def fit(self, theta, X_train, X_val):
        K, I = theta.shape
        U, I = X_train.shape

        self.P = np.random.rand(K, U).astype(np.float32)
        self.Q = theta.copy().astype(np.float32)

        self.train_error = []
        self.val_error = []

        users, items = X_train.nonzero()

        for epoch in trange(self.epochs, desc="Training CTR"):
            for u, i in zip(users, items):
                pred = np.dot(self.P[:, u], self.Q[:, i])
                error = X_train[u, i] - pred
                self.P[:, u] += self.learning_rate * (error * self.Q[:, i] - self.P[:, u] / self.sigma2_P)
                self.Q[:, i] += self.learning_rate * (error * self.P[:, u] - (self.Q[:, i] - theta[:, i]) / self.sigma2_Q)

            train_predictions = self.predict_ratings(X_train)
            val_predictions = self.predict_ratings(X_val)

    def predict_ratings(self, X):
        rows, cols = X.nonzero()
        predictions = np.zeros(len(rows), dtype=np.float32)

        for idx, (u, i) in enumerate(zip(rows, cols)):
            predictions[idx] = np.dot(self.P[:, u], self.Q[:, i])

        predictions = np.round(predictions, decimals=6)

        return csr_matrix((predictions, (rows, cols)), shape=X.shape)

    def mse(self, prediction, ground_truth):
        pred_rows, pred_cols = prediction.nonzero()
        gt_rows, gt_cols = ground_truth.nonzero()
        
        pred_data = prediction.data
        gt_data = ground_truth.data

        return np.mean((pred_data - gt_data) ** 2)


    def predict_out_of_matrix(self, topics):
        return np.dot(self.P.T, topics)

def load_data():
    games = pd.read_csv("C:\\Users\\yanch\\Desktop\\AI_Project\\games.csv") # .head(5000)
    
    recommendations = pd.read_csv("C:\\Users\\yanch\\Desktop\\AI_Project\\recommendations.csv")
    # recommendations = recommendations[recommendations['app_id'].isin(games['app_id'])]
    
    users = pd.read_csv("C:\\Users\\yanch\\Desktop\\AI_Project\\users.csv")
    # users = users[users['user_id'].isin(recommendations['user_id'])]
    
    with open("C:\\Users\\yanch\\Desktop\\AI_Project\\games_metadata.json", "r", encoding="utf-8") as file:
        metadata = [json.loads(line) for line in file]
    metadata_df = pd.DataFrame(metadata)
    # metadata_df = metadata_df[metadata_df['app_id'].isin(games['app_id'])]
    
    return games, recommendations, users, metadata_df

def preprocess_data(games, recommendations, metadata_df):
    games = games.dropna(subset=["title", "positive_ratio"])
    metadata_df['title'] = metadata_df['app_id'].map(games.set_index('app_id')['title'])
    metadata_df = metadata_df.dropna(subset=['description'])

    recommendations = recommendations[recommendations['is_recommended'] == True]

    user_ids = recommendations['user_id'].astype('category')
    app_ids = recommendations['app_id'].astype('category')

    user_ids = user_ids.cat.set_categories(user_ids.cat.categories, ordered=True)
    app_ids = app_ids.cat.set_categories(app_ids.cat.categories, ordered=True)

    ratings_matrix = csr_matrix(
        (recommendations['hours'], (user_ids.cat.codes, app_ids.cat.codes)), dtype=np.float32
    )
    ratings_matrix.sum_duplicates()

    max_users, max_items = metadata_df.shape[0], metadata_df['app_id'].nunique()
    if ratings_matrix.shape[1] > max_items:
        ratings_matrix = ratings_matrix[:, :max_items]
    if ratings_matrix.shape[0] > max_users:
        ratings_matrix = ratings_matrix[:max_users, :]

    valid_user_indices = np.arange(ratings_matrix.shape[0])
    user_ids = user_ids.cat.set_categories(user_ids.cat.categories[valid_user_indices])

    return games, metadata_df, ratings_matrix, user_ids, app_ids

def prepare_topic_model(metadata_df):
    nlp = spacy.load("en_core_web_sm")
    metadata_df['lemmas'] = metadata_df['description'].apply(lambda x: [token.lemma_ for token in nlp(x) if token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]])

    dictionary = Dictionary(metadata_df['lemmas'])
    corpus = [dictionary.doc2bow(text) for text in metadata_df['lemmas']]
    tfidf_model = TfidfModel(corpus)
    corpus_tfidf = tfidf_model[corpus]

    num_topics = 15
    lda_model = LdaMulticore(corpus=corpus_tfidf, id2word=dictionary, num_topics=num_topics, passes=10, workers=multiprocessing.cpu_count() - 1)
    
    topic_proportions = []
    for doc in corpus:
        topic_proportions.append([topic_prob for _, topic_prob in lda_model.get_document_topics(doc, minimum_probability=0)])
    
    theta = np.array(topic_proportions).T
    return theta, lda_model

def train_ctm_model(theta, ratings_matrix):
    num_topics, num_items = theta.shape
    num_users, num_items_in_matrix = ratings_matrix.shape

    if num_items_in_matrix != num_items:
        ratings_matrix = ratings_matrix[:, :num_items]

    if num_users > theta.shape[0]:
        ratings_matrix = ratings_matrix[:theta.shape[0], :]

    X_train, X_val = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

    ctm = CTR(epochs=200, learning_rate=1e-5, sigma2_P=50, sigma2_Q=50)
    ctm.fit(theta, X_train, X_val)

    return ctm

def preprocess_data(games, recommendations, metadata_df):
    games = games.dropna(subset=["title", "positive_ratio"])
    metadata_df['title'] = metadata_df['app_id'].map(games.set_index('app_id')['title'])
    metadata_df = metadata_df.dropna(subset=['description'])

    recommendations = recommendations[recommendations['is_recommended'] == True]

    user_ids = recommendations['user_id'].astype('category')
    app_ids = recommendations['app_id'].astype('category')

    user_ids = user_ids.cat.set_categories(user_ids.cat.categories, ordered=True)
    app_ids = app_ids.cat.set_categories(app_ids.cat.categories, ordered=True)

    ratings_matrix = csr_matrix(
        (recommendations['hours'], (user_ids.cat.codes, app_ids.cat.codes)), dtype=np.float32
    )
    ratings_matrix.sum_duplicates()

    max_users, max_items = metadata_df.shape[0], metadata_df['app_id'].nunique()
    if ratings_matrix.shape[1] > max_items:
        ratings_matrix = ratings_matrix[:, :max_items]
    if ratings_matrix.shape[0] > max_users:
        valid_user_indices = np.arange(max_users)
        ratings_matrix = ratings_matrix[:max_users, :]
        user_ids = user_ids.cat.remove_unused_categories()
        user_ids = user_ids.cat.set_categories(user_ids.cat.categories[valid_user_indices])

    return games, metadata_df, ratings_matrix, user_ids, app_ids

def recommend_games(user_id, ctm, ratings_matrix, games_df, user_ids, app_ids, top_n=10):
    if user_id not in user_ids.cat.categories:
        print(f"Error: User ID {user_id} not found. Returning fallback recommendations.")
        fallback_games = games_df.sort_values("positive_ratio", ascending=False)
        return fallback_games[["app_id", "title", "positive_ratio"]]

    print(f"User ID {user_id} exists. Proceeding.")

    try:
        user_row_index = np.where(user_ids.cat.categories == user_id)[0][0]
        if user_row_index >= ratings_matrix.shape[0]:
            raise IndexError(f"User row index {user_row_index} exceeds ratings matrix dimensions.")
    except IndexError as e:
        return pd.DataFrame({"Error": [f"Invalid user_id: {user_id}"]})
    
    predicted_ratings = ctm.P.T @ ctm.Q[:, user_row_index]
    print("Predicted ratings calculated.")

    rated_games = ratings_matrix.getrow(user_row_index).nonzero()[1]
    unrated_indices = np.setdiff1d(np.arange(ratings_matrix.shape[1]), rated_games, assume_unique=True)

    if len(unrated_indices) == 0:
        return pd.DataFrame({"Error": ["No unrated games available for recommendations."]})

    unrated_ratings = np.take(predicted_ratings, unrated_indices, mode='clip')

    top_n = min(top_n, len(unrated_indices))

    sorted_indices = np.argsort(unrated_ratings)[-top_n:][::-1]
    recommendations = np.array(unrated_indices)[sorted_indices]

    recommended_games = pd.DataFrame(
        [
            {
                "app_id": app_ids.cat.categories[i],
                "title": games_df.loc[games_df['app_id'] == app_ids.cat.categories[i], 'title'].iloc[0],
            }
            for idx, i in enumerate(recommendations)
        ]
    )
    return recommended_games

def hit_ratio_test(ctm, ratings_matrix, user_ids, app_ids, top_k=100):
    hits = 0
    total_evaluated = 0

    for user_idx in range(ratings_matrix.shape[0]):
        user_interactions = ratings_matrix.getrow(user_idx).toarray().flatten()
        rated_items = np.where(user_interactions > 0)[0]

        if len(rated_items) == 0:
            continue

        test_item = rated_items[0]
        train_items = rated_items[1:]

        test_ratings_matrix = ratings_matrix.copy()
        test_ratings_matrix[user_idx, test_item] = 0

        predicted_ratings = ctm.P.T @ ctm.Q[:, user_idx]
        
        unrated_indices = np.setdiff1d(np.arange(ratings_matrix.shape[1]), train_items, assume_unique=True)
        predicted_ratings = np.take(predicted_ratings, unrated_indices, mode='clip')
        
        top_indices = np.argsort(predicted_ratings)[-top_k:][::-1]

        if test_item in unrated_indices[top_indices]:
            hits += 1

        total_evaluated += 1

    hit_ratio = hits / total_evaluated if total_evaluated > 0 else 0
    return hit_ratio

if __name__ == "__main__":
    games, recommendations, users, metadata_df = load_data()
    games, metadata_df, ratings_matrix, user_ids, app_ids = preprocess_data(games, recommendations, metadata_df)

    theta, lda_model = prepare_topic_model(metadata_df)
    ctm_model = train_ctm_model(theta, ratings_matrix)

    hit_ratio = hit_ratio_test(ctm_model, ratings_matrix, user_ids, app_ids, top_k=100)
    print(f"Hit Ratio: {hit_ratio * 100:.2f}%")

    user_id = 13103
    print(f"Testing recommendations for User ID: {user_id}")
    recommended_games = recommend_games(user_id, ctm_model, ratings_matrix, games, user_ids, app_ids)

    if "Error" in recommended_games.columns:
        print(recommended_games["Error"].iloc[0])
    else:
        print("Recommended games:")
        print(recommended_games)