from scipy.stats import pearsonr
from scipy.stats import ortho_group
from scipy.sparse import issparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectFromModel
from transformers import pipeline as transformer_pipeline
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import copy
import scipy.linalg as alg
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utility_funcs import (
    SHARED_RANDOM_SEED,
    load_or_create_peer_sus,
    load_or_create_platform_sus,
    predicting,
    run_simulation,
    seed_everything,
    run_sus_var,
)

from pokec_preprocessing import (
    DATA_DIR as POKEC_DATA_DIR,
    load_or_compute_features,
    load_or_compute_scores,
    load_profiles_and_graph,
    parse_args,
)



feature_labels = [
    "user_id",
    "public",
    "completion_percentage", 
    "gender", # 0,1
    "region", #categorical, string
    "last_login",
    "registration",
    "age", #numberical, 0-age attribute not set
    "body", #cm, kg, can also contain text
    "I_am_working_in_field", #text
    "spoken_languages", #text
    "hobbies", #text
    "I_most_enjoy_good_food", #text
    "pets", #text
    "body_type", #text
    "my_eyesight", #text
    "eye_color", #text
    "hair_color", #text
    "hair_type", #text
    "completed_level_of_education", #text
    "favourite_color", #text
    "relation_to_smoking", #text
    "relation_to_alcohol", #text
    "sign_in_zodiac", #text
    "on_pokec_i_am_looking_for", #text
    "love_is_for_me", #text
    "relation_to_casual_sex", #text
    "my_partner_should_be", #text
    "marital_status", #text
    "children", #text
    "relation_to_children", #text
    "I_like_movies", #text
    "I_like_watching_movie", #text
    "I_like_music", #text
    "I_mostly_like_listening_to_music", #text
    "the_idea_of_good_evening", #text
    "I_like_specialties_from_kitchen", #text
    "fun", #text, but contains a lot of link
    "I_am_going_to_concerts", #text
    "my_active_sports", #text
    "my_passive_sports", #text
    "profession", #text
    "I_like_books", #text
    "life_style", #text, jason file, contain links
    "music", #text, jason, link
    "cars", #text, jason, link
    "politics", #text, jason, link
    "relationships", #text, jason, link
    "art_culture", #text, jason, link
    "hobbies_interests", #text, jason, link
    "science_technologies", #text, jason, link
    "computers_internet", #text, jason, link
    "education", #text, jason, link
    "sport", #text, jason, link
    "movies", #text, jason, link
    "travelling", #text, jason, link
    "health", #text, jason, link
    "companies_brands", #text, jason, link
    "more" #text, jason, link
]

numerical_features = [
    "age"
    ]
categorical_features = [
    "gender",
    # "region"
    ]
textual_features = [
    # "body",
    # "I_am_working_in_field",
    # "spoken_languages",
    # "hobbies",
    # "I_most_enjoy_good_food",
    # "pets",
    # "body_type",
    # "my_eyesight",
    # "eye_color",
    # "hair_color",
    # "hair_type",
    # "completed_level_of_education",
    # "favourite_color",
    # "relation_to_smoking",
    "relation_to_alcohol",
    # "sign_in_zodiac",
    # "on_pokec_i_am_looking_for",
    # "love_is_for_me",
    # "relation_to_casual_sex",
    # "my_partner_should_be",
    # "marital_status",
    # "children",
    # "relation_to_children",
    # "I_like_movies",
    # "I_like_watching_movie",
    # "I_like_music",
    # "I_mostly_like_listening_to_music",
    # "the_idea_of_good_evening",
    # "I_like_specialties_from_kitchen",
    # "fun",
    # "I_am_going_to_concerts",
    # "my_active_sports",
    # "my_passive_sports",
    # "profession",
    # "I_like_books"
    # "life_style",
    # "music",
    # "cars",
    # "politics",
    # "relationships",
    # "art_culture",
    # "hobbies_interests",
    # "science_technologies",
    # "computers_internet",
    # "education",
    # "sport",
    # "movies",
    # "travelling",
    # "health",
    # "companies_brands",
    # "more"
]




class TextConcatEmbedder(BaseEstimator, TransformerMixin):
    def __init__(self, model_name, batch_size=16, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.model = None

    def fit(self, X, y=None):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            rows = X.astype(str).fillna("").agg(" ".join, axis=1).tolist()
        else:
            rows = pd.DataFrame(X).astype(str).fillna("").agg(" ".join, axis=1).tolist()
        rows = [s.strip() for s in rows]
        return np.asarray(
            self.model.encode(
                rows,
                batch_size=self.batch_size,
                show_progress_bar=True,
            )
        )


def build_pipeline(numerical_features, categorical_features, filtered_textual_features, model_name, batch_size, device):
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    text_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("embed", TextConcatEmbedder(model_name, batch_size=batch_size, device=device)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numerical_features),
            ("cat", cat_transformer, categorical_features),
            ("text", text_transformer, filtered_textual_features),
        ],
        sparse_threshold=0.3,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kinit/slovakbert-sts-stsb")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    return parser.parse_args()


def preprocess(df, target_column, edge_path="pokec_dataset/relationships.txt"):

    # we only focus individuals with public friendships
    df_public = df[df["public"] != 0]
    df_public_ids = set(df_public["user_id"].values)
    edges = pd.read_csv(edge_path, sep="\t", header=None)
    network_g = nx.Graph()
    for edge in edges.itertuples():
        if edge[1] not in df_public_ids or edge[2] not in df_public_ids:
            continue 
        else:
            network_g.add_edge(edge[1], edge[2])
    
    components = list(nx.connected_components(network_g))
    lcc = max(components, key=len)
    network_lcc = network_g.subgraph(lcc).copy()
    with open("pokec_dataset/lcc_graph_" + target_column + ".pk", "wb") as f:
        pickle.dump(network_lcc, f)
    lcc_public_df = df_public[df_public["user_id"].isin(lcc)]
    print("number of users in lcc with public friendships:", len(lcc_public_df))
    nodelist = list(lcc_public_df["user_id"].values)  # 0..n-1 in row order
    w = nx.to_numpy_array(network_lcc, nodelist=nodelist, dtype=int)
    return lcc_public_df, network_lcc

def sentiment_scores(texts, sentiment_pipe, batch_size=32):
    scores = sentiment_pipe(texts, batch_size=batch_size)
    return np.array([
        r["score"] if r["label"] == "positive"
        else 0.5 if r["label"] == "neutral"
        else 1 - r["score"]
        for r in scores
    ], dtype=float)


def extract_features(df, args, filtered_textual_features, numerical_features_extended):
    
    
    # enforce data types
    df[numerical_features_extended] = df[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df[categorical_features] = df[categorical_features].astype(str)
    df[filtered_textual_features] = df[filtered_textual_features].astype(str)

    use_sentiment_scores = True
    if use_sentiment_scores:

        # sentiment model (same as y_label)
        sentiment = transformer_pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            device=0 if args.device and "cuda" in args.device else -1,
            use_fast=False,
        )

        text_feat_matrix = {}
        for col in filtered_textual_features:
            text_feat_matrix[f"{col}_sent"] = sentiment_scores(
                df[col].fillna("").astype(str).tolist(),
                sentiment
            )

        text_feat_df = pd.DataFrame(text_feat_matrix, index=df.index)

        # numeric + categorical
        num_df = df[numerical_features].apply(pd.to_numeric, errors="coerce").fillna(0)
        cat_df = df[categorical_features].astype(str)

        # one-hot categorical
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_ohe = ohe.fit_transform(cat_df)
        cat_ohe_df = pd.DataFrame(cat_ohe, index=df.index, columns=ohe.get_feature_names_out())

        # final feature matrix
        X_features = pd.concat([num_df, cat_ohe_df, text_feat_df], axis=1)
        X_features = X_features.to_numpy()
        print("Feature matrix shape:", X_features.shape)

    else:

        pipeline = build_pipeline(
            numerical_features_extended,
            categorical_features,
            filtered_textual_features,
            model_name=args.model,
            batch_size=args.batch_size,
            device=args.device,
        )

        
        X = df[numerical_features_extended + categorical_features + filtered_textual_features]

        X_features = pipeline.fit_transform(X)  


    print("Feature matrix shape:", X_features.shape)
    return X_features


def compute_score(df, target_column, args):
    
    sentiment = transformer_pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
        device=0 if args.device and "cuda" in args.device else -1,
    )
    
    scores = sentiment(df[target_column].fillna("").astype(str).tolist(), batch_size=32)
    results = [
        r["score"] if r["label"] == "positive"
        else 0.5 if r["label"] == "neutral"
        else 1 - r["score"]
        for r in scores
    ]
    
    return results

def add_graph_features(df, graph_path):
    
    with open(graph_path, "rb") as f:
        network_lcc = pickle.load(f)
    
    degree = dict(network_lcc.degree())
    clustering = nx.clustering(network_lcc)
    pagerank = nx.pagerank(network_lcc, alpha=0.85)

    df["deg"] = df["user_id"].map(degree).fillna(0)
    df["clust"] = df["user_id"].map(clustering).fillna(0)
    df["pr"] = df["user_id"].map(pagerank).fillna(0)
    return df



    

def create_plot():

    param_folder = POKEC_DATA_DIR / "parametric_params"
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    enforced_sus = np.arange(0.0, 1.1, 0.1)
    
    models = ["perfect", "ridge", "mean", "lightgbm"]
    labels = ["Perfect", "OLS", "Mean", "LightGBM"]
    # models = ["perfect"]
    # labels = ["Perfect"]

    fig, ax = plt.subplots()
    
    for i in range(len(models)):
        with (POKEC_DATA_DIR / "results" / f"variance_{models[i]}_platform.pk").open("rb") as f:
            variances = pickle.load(f)

        ax.plot(enforced_sus, variances, linewidth=1.5, label=labels[i], color=colors[i], linestyle="-", marker="o")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xticks(enforced_sus)
    ax.set_xlabel(r"Platform susceptibility $\beta$", fontsize=18)
    ax.set_ylabel(r"Var($x_{PS}$)", fontsize=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(loc="lower left", frameon=False, fontsize=15)
    plt.savefig(param_folder / "platform_variance_sl.pdf", bbox_inches='tight')
    
    fig, ax = plt.subplots()

    for i in range(len(models)):
        with (POKEC_DATA_DIR / "results" / f"variance_{models[i]}_peer.pk").open("rb") as f:
            variances = pickle.load(f)

        ax.plot(enforced_sus, variances, linewidth=1.5, label=labels[i], color=colors[i], linestyle="-", marker="o")


    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax.set_xticks(enforced_sus)
    ax.set_xlabel(r"Peer susceptibility $\alpha$", fontsize=18)
    ax.set_ylabel(r"Var($x_{PS}$)", fontsize=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)

    ax.legend(loc="upper right", frameon=False, fontsize=15)
    
    plt.savefig(param_folder / "peer_variance_sl.pdf", bbox_inches='tight')
    



def main():
    args = parse_args()
    seed_everything(use_cuda=bool(args.device and "cuda" in args.device))
    target_column = "relation_to_smoking"
    include_graph_features = False
    df, network_lcc = load_profiles_and_graph(target_column)
    param_folder = POKEC_DATA_DIR / "parametric_params"
    param_folder.mkdir(exist_ok=True, parents=True)
    results_folder = POKEC_DATA_DIR / "results"
    results_folder.mkdir(exist_ok=True, parents=True)

    n = int(len(df) * 0.8)
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()
    y_label, y_unlabel_label = load_or_compute_scores(
        df_labeled, df_unlabeled, target_column, args, len(df), cache_dir=param_folder
    )
    X_features_labeled, X_features_unlabeled = load_or_compute_features(
        df_labeled,
        df_unlabeled,
        target_column,
        include_graph_features,
        args,
        cache_dir=POKEC_DATA_DIR,
    )
    innate_opinions = np.array(y_label + y_unlabel_label)
    adjust_plot = True
    
    retrain_T = 50
    fj_K = 100
    if adjust_plot:
        create_plot()
    else:
        
        for model_name in ["perfect", "mean", "ridge", "lightgbm"]:
            for test_sus in ["platform", "peer"]:
                run_sus_var(
                    retrain_T=retrain_T,
                    fj_K=fj_K,
                    DATA_DIR=POKEC_DATA_DIR,
                    test_sus=test_sus,
                    innate_opinions=innate_opinions,
                    network_lcc=network_lcc,
                    nodelist=df["user_id"].values,
                    model_name=model_name,
                    X_features_labeled=X_features_labeled,
                    X_features_unlabeled=X_features_unlabeled)
    
        create_plot()



if __name__ == "__main__":
    main()
