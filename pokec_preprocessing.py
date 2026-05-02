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
    run_simulation,
    seed_everything,
    add_graph_features,
)


SHARED_RANDOM_SEED = 2026
DATA_DIR = Path("pokec_dataset")


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
    # print("number of users with public friendships:", len(df_public_ids))
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




def load_profiles_and_graph(
    target_column,
    sample_size=30000,
    random_state=SHARED_RANDOM_SEED,
    data_dir=DATA_DIR,
):
    profiles_path = data_dir / f"lcc_profiles_{target_column}.pk"
    graph_path = data_dir / f"lcc_graph_{target_column}.pk"

    if profiles_path.exists() and graph_path.exists():
        with profiles_path.open("rb") as f:
            df = pickle.load(f)
        with graph_path.open("rb") as f:
            network_lcc = pickle.load(f)
        print("lcc user num:", len(df))
        return df, network_lcc

    df = pd.read_csv(data_dir / "profiles.txt", sep="\t", header=None)
    feature_len = len(feature_labels)
    df = df.iloc[:, :feature_len]
    df.columns = feature_labels
    df = df.replace(r"^\s*$", np.nan, regex=True)
    mask = df[target_column].isna() | (df[target_column].astype(str).str.strip() == "")
    df = df[~mask].copy()
    df = df.sample(n=sample_size, random_state=random_state)
    df, network_lcc = preprocess(df, target_column, edge_path=str(data_dir / "relationships.txt"))
    print("number of nodes in lcc:", len(network_lcc))

    with profiles_path.open("wb") as f:
        pickle.dump(df, f)

    return df, network_lcc


def load_or_compute_scores(
    df_labeled,
    df_unlabeled,
    target_column,
    args,
    total_n,
    cache_dir=DATA_DIR / "parametric_params",
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    y_label_path = cache_dir / f"y_label{total_n}.pk"
    y_unlabel_path = cache_dir / f"y_unlabel_label{total_n}.pk"

    if y_label_path.exists() and y_unlabel_path.exists():
        with y_label_path.open("rb") as f:
            y_label = pickle.load(f)
        with y_unlabel_path.open("rb") as f:
            y_unlabel_label = pickle.load(f)
        return y_label, y_unlabel_label

    y_label = compute_score(df_labeled, target_column, args)
    y_unlabel_label = compute_score(df_unlabeled, target_column, args)

    with y_label_path.open("wb") as f:
        pickle.dump(y_label, f)
    with y_unlabel_path.open("wb") as f:
        pickle.dump(y_unlabel_label, f)

    return y_label, y_unlabel_label


def load_or_compute_features(
    df_labeled,
    df_unlabeled,
    target_column,
    include_graph_features,
    args,
    cache_dir=DATA_DIR,
):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_key = f"{target_column}_{include_graph_features}"
    labeled_path = cache_dir / f"labeled_feature_matrix_{cache_key}.pk"
    unlabeled_path = cache_dir / f"unlabeled_feature_matrix_{cache_key}.pk"

    if labeled_path.exists() and unlabeled_path.exists():
        with labeled_path.open("rb") as f:
            X_features_labeled = pickle.load(f)
        with unlabeled_path.open("rb") as f:
            X_features_unlabeled = pickle.load(f)
        # print("Feature matrix shape:", X_features_labeled.shape)
        return X_features_labeled, X_features_unlabeled

    filtered_textual_features = [text for text in textual_features if text != target_column]
    numerical_features_extended = numerical_features.copy()
    if include_graph_features:
        numerical_features_extended += ["deg", "clust", "pr"]
        df_labeled = add_graph_features(df_labeled.copy(), graph_path=cache_dir / f"lcc_graph_{target_column}.pk")
        df_unlabeled = add_graph_features(df_unlabeled.copy(), graph_path=cache_dir / f"lcc_graph_{target_column}.pk")

    df_labeled[numerical_features_extended] = df_labeled[numerical_features_extended].apply(
        pd.to_numeric, errors="coerce"
    )
    df_labeled[categorical_features] = df_labeled[categorical_features].astype(str)
    df_labeled[filtered_textual_features] = df_labeled[filtered_textual_features].astype(str)

    df_unlabeled[numerical_features_extended] = df_unlabeled[numerical_features_extended].apply(
        pd.to_numeric, errors="coerce"
    )
    df_unlabeled[categorical_features] = df_unlabeled[categorical_features].astype(str)
    df_unlabeled[filtered_textual_features] = df_unlabeled[filtered_textual_features].astype(str)

    X_features_labeled = extract_features(
        df_labeled, args, filtered_textual_features, numerical_features_extended
    )
    X_features_unlabeled = extract_features(
        df_unlabeled, args, filtered_textual_features, numerical_features_extended
    )

    with labeled_path.open("wb") as f:
        pickle.dump(X_features_labeled, f)
    with unlabeled_path.open("wb") as f:
        pickle.dump(X_features_unlabeled, f)

    return X_features_labeled, X_features_unlabeled


def run_opinion_dynamics(innate_opinions, network_lcc, nodelist, model_name, X_features_labeled, X_features_unlabeled):
    
    agent_num = len(innate_opinions)
    fj_K = 100
    retrain_T = 20
    x_initial = copy.deepcopy(innate_opinions)

    param_folder = "pokec_dataset/parametric_params/"
    realworld_params = Path(param_folder)
    realworld_params.mkdir(exist_ok=True)

    # generate heterogeneous parameters
    platform_file_path = Path(param_folder + "hetero_platform_sus" + str(agent_num) + ".pkl")

    if not platform_file_path.exists():
        
        platform_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(platform_file_path, "wb") as file:
            pickle.dump(platform_sus, file)
    else:
        with open(platform_file_path, "rb") as file:
            platform_sus = pickle.load(file)

    peer_file_path = Path(param_folder + "hetero_peer_sus" + str(agent_num) + ".pkl")

    if not peer_file_path.exists():
        
        peer_sus = np.clip(np.random.normal(loc=0.9, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(peer_file_path, "wb") as file:
            pickle.dump(peer_sus, file)
    else:
        with open(peer_file_path, "rb") as file:
            peer_sus = pickle.load(file)


    steer_file_path = Path(param_folder + "hetero_steer_sus" + str(agent_num) + ".pkl")

    if not steer_file_path.exists():
        steer_strength = np.clip(np.random.normal(loc=0.1, scale=0.1, size=agent_num), 0.01, 0.99)
        with open(steer_file_path, "wb") as file:
            pickle.dump(steer_strength, file)

    else:
        with open(steer_file_path, "rb") as file:
            steer_strength = pickle.load(file)

    # for illustrating supervised learning policy, strong performativity
    platform_sus = np.ones(agent_num)
    steer_strength = np.zeros(agent_num)
    results_folder = "pokec_dataset/results/"
    if os.path.exists(results_folder + model_name + "_equilibrium.pk"):
        with open(results_folder + model_name + "_equilibrium.pk", "rb") as f:
            perform_equilibrium = pickle.load(f)
        with open(results_folder + model_name + "_FJequilibrium.pk", "rb") as f:
            FJ_equilibrium = pickle.load(f)
    else:

        equilibrium_opinions = run_simulation(network=network_lcc, nodelist=nodelist, platform_params=platform_sus, 
                                            peer_params=peer_sus, steering_params=steer_strength, 
                                            steering_vector=None, fj_steps=fj_K, retrain_steps=retrain_T, 
                                            x_star=innate_opinions, policy='sl', model_name=model_name, 
                                            X_features_labeled=X_features_labeled, X_features_unlabeled=X_features_unlabeled)

        interval_num = retrain_T
        heatmap_res1 = np.zeros((agent_num, interval_num+1))
        heatmap_res1[:, 0] = copy.deepcopy(x_initial)
        for k in range(interval_num):
            heatmap_res1[:, k+1] = copy.deepcopy(equilibrium_opinions[:, (k+1)*int(equilibrium_opinions.shape[1]/interval_num)])


        x = np.arange(1, agent_num+1)
        FJ_equilibrium = heatmap_res1[:, 1]
        perform_equilibrium = heatmap_res1[:, -1]
        with open(results_folder + model_name + "_equilibrium.pk", "wb") as f:
            pickle.dump(perform_equilibrium, f)
        with open(results_folder + model_name + "_FJequilibrium.pk", "wb") as f:
            pickle.dump(FJ_equilibrium, f)
        
    
    # reference with perfect predictions are pre-generated
    if model_name == 'perfect':
        with open("pokec_dataset/results/perfect_equilibrium.pk", "wb") as f:
            pickle.dump(perform_equilibrium, f)
        perfect_equilibrium = copy.deepcopy(perform_equilibrium)
    else:
        if os.path.exists("pokec_dataset/results/perfect_equilibrium.pk"):
            with open("pokec_dataset/results/perfect_equilibrium.pk", "rb") as f:
                perfect_equilibrium = pickle.load(f)
        else:
            print("Reference missing! Please generate with policy== perfect first!")
            perfect_equilibrium = np.full(agent_num, np.nan)



    innate_mean = innate_opinions.mean(axis=0)
    innate_std = innate_opinions.std(axis=0)

    fj_mean = FJ_equilibrium.mean(axis=0)
    fj_std = FJ_equilibrium.std(axis=0)

    performative_mean = perform_equilibrium.mean(axis=0)
    performative_std = perform_equilibrium.std(axis=0)

    perfect_mean = perfect_equilibrium.mean(axis=0)
    perfect_std = perfect_equilibrium.std(axis=0)

    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(1, agent_num+1)

    plt.scatter(x, innate_opinions, s=5, color=colors[0], label=r"$x^*$")
    plt.scatter(x, FJ_equilibrium, s=5, color=colors[1], label=r"FJ($x^*$)")
    plt.scatter(x, perfect_equilibrium, s=5, color=colors[3], label=r"$x_{PS}$ (perfect)")
    plt.scatter(x, perform_equilibrium, s=5, color=colors[2], label=r"$x_{PS}$ (imperfect)")


    plt.fill_between(x, innate_mean - innate_std, innate_mean + innate_std, color=colors[0], alpha=0.2)
    plt.fill_between(x, fj_mean - fj_std, fj_mean + fj_std, color=colors[1], alpha=0.2)
    plt.fill_between(x, performative_mean - performative_std, performative_mean + performative_std, color=colors[2], alpha=0.2)
    plt.fill_between(x, perfect_mean - perfect_std, perfect_mean + perfect_std, color=colors[3], alpha=0.2)
    
    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylabel("Opinions", fontsize=13)
    plt.legend(loc="upper left", bbox_to_anchor=(1,1), frameon=False, fontsize=10)
    plt.savefig(param_folder + model_name + "_parametric_sl.pdf", bbox_inches='tight')


def plot_adjust(model_name):
    results_folder = "pokec_dataset/results/"
    param_folder = "pokec_dataset/parametric_params/"

    with open(results_folder + model_name + "_equilibrium.pk", "rb") as f:
        perform_equilibrium = pickle.load(f)
    with open(results_folder + model_name + "_FJequilibrium.pk", "rb") as f:
        FJ_equilibrium = pickle.load(f)
    with open("pokec_dataset/parametric_params/y_label2163.pk", "rb") as f:
        y_label = pickle.load(f)
    with open("pokec_dataset/parametric_params/y_unlabel_label2163.pk", "rb") as f:
        y_unlabel_label = pickle.load(f)
    innate_opinions = np.array(y_label + y_unlabel_label)
    with open("pokec_dataset/results/perfect_equilibrium.pk", "rb") as f:
        perfect_equilibrium = pickle.load(f)
    innate_mean = innate_opinions.mean(axis=0)
    innate_std = innate_opinions.std(axis=0)

    fj_mean = FJ_equilibrium.mean(axis=0)
    fj_std = FJ_equilibrium.std(axis=0)

    performative_mean = perform_equilibrium.mean(axis=0)
    performative_std = perform_equilibrium.std(axis=0)

    perfect_mean = perfect_equilibrium.mean(axis=0)
    perfect_std = perfect_equilibrium.std(axis=0)

    
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    x = np.arange(1, 2164)

    plt.scatter(x, innate_opinions, s=4, color=colors[0], label=r"$x^*$")
    plt.scatter(x, FJ_equilibrium, s=4, color=colors[1], label=r"FJ($x^*)$")
    plt.scatter(x, perfect_equilibrium, s=4, color=colors[3], label=r"$x_{PS}$ (perfect)")
    plt.scatter(x, perform_equilibrium, s=4, color=colors[2], label=r"$x_{PS}$ (imperfect)")

    x_right = x.max() + 45

    plt.errorbar([x_right+210], [innate_mean], yerr=[innate_std], fmt="o", markersize=4, capsize=4, color=colors[0])
    plt.errorbar([x_right+140], [fj_mean], yerr=[fj_std], fmt="o", capsize=4, markersize=4, color=colors[1])
    plt.errorbar([x_right+70], [perfect_mean], yerr=[perfect_std], fmt="o", markersize=4, capsize=4, color=colors[3])
    plt.errorbar([x_right], [performative_mean], yerr=[performative_std], fmt="o", markersize=4, capsize=4, color=colors[2])

    
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.ylabel("Opinions", fontsize=18)
    plt.gca().set_xticklabels([])
    plt.yticks(fontsize=12)
    plt.legend(loc="upper left", bbox_to_anchor=(0.97,1), frameon=False, fontsize=15)
    plt.savefig(param_folder + model_name + "_parametric_sl.pdf", bbox_inches='tight')

def main():
    args = parse_args()
    seed_everything(use_cuda=bool(args.device and "cuda" in args.device))
    # select the feature to be predicted, choose relation_to_smoking as the target label
    target_column = "relation_to_smoking"

    if os.path.exists("pokec_dataset/lcc_profiles_" + target_column + ".pk"):
        with open("pokec_dataset/lcc_profiles_" + target_column + ".pk", "rb") as f:
            df = pickle.load(f)
        print("lcc user num:", len(df))

        with open("pokec_dataset/lcc_graph_" + target_column + ".pk", "rb") as f:
            network_lcc = pickle.load(f)
    else:
        df = pd.read_csv("pokec_dataset/profiles.txt", sep="\t", header=None)
        feature_len = len(feature_labels)
        df = df.iloc[:, :feature_len]
        df.columns = feature_labels
        df = df.replace(r"^\s*$", np.nan, regex=True)
        mask = df[target_column].isna() | (df[target_column].astype(str).str.strip() == "")
        df = df[~mask].copy()
        df = df.sample(n=30000, random_state=SHARED_RANDOM_SEED)
        # now only users in lcc and public friendships are maintained
        df, network_lcc = preprocess(df, target_column, edge_path="pokec_dataset/relationships.txt")
        print("number of nodes in lcc:", len(network_lcc))
        with open("pokec_dataset/lcc_profiles_" + target_column + ".pk", "wb") as f:
            pickle.dump(df, f)
    
    include_graph_features = False
    if include_graph_features:
        df = add_graph_features(df, graph_path="pokec_dataset/lcc_graph_" + target_column + ".pk")
        numerical_features_extended = numerical_features + ["deg", "clust", "pr"]
    else: 
        numerical_features_extended = numerical_features.copy()
    
    # assume we don't have access to the label of the 20% population
    n = int(len(df) * 0.8)
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()
    # we compute the sentiment scores as innate opinions of individuals
    # the platform has access to the innate opinion of labeled group
    # the platform has no access to the innate opinion of labeled group
    if not os.path.exists("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk"):
        y_label = compute_score(df_labeled, target_column, args)
        y_unlabel_label = compute_score(df_unlabeled, target_column, args)
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_label, f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "wb") as f:
            pickle.dump(y_unlabel_label, f)
    else:
        with open("pokec_dataset/parametric_params/y_label" + str(len(df)) + ".pk", "rb") as f:
            y_label = pickle.load(f)
        with open("pokec_dataset/parametric_params/y_unlabel_label" + str(len(df)) + ".pk", "rb") as f:
            y_unlabel_label = pickle.load(f)
   
    # extract features from mixed data types
    filtered_textual_features = [text for text in textual_features if text != target_column]
    df_labeled[numerical_features_extended] = df_labeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_labeled[categorical_features] = df_labeled[categorical_features].astype(str)
    df_labeled[filtered_textual_features] = df_labeled[filtered_textual_features].astype(str)
    df_unlabeled[numerical_features_extended] = df_unlabeled[numerical_features_extended].apply(pd.to_numeric, errors="coerce")
    df_unlabeled[categorical_features] = df_unlabeled[categorical_features].astype(str)
    df_unlabeled[filtered_textual_features] = df_unlabeled[filtered_textual_features].astype(str)

    
    
    if os.path.exists("pokec_dataset/labled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk"): 
        with open("pokec_dataset/labeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "rb") as f:
            X_features_labeled = pickle.load(f)
        print("Feature matrix shape:", X_features_labeled.shape)
        with open("pokec_dataset/unlabeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "rb") as f:
            X_features_unlabeled = pickle.load(f)
    else:
        X_features_labeled = extract_features(df_labeled, args, filtered_textual_features, numerical_features_extended)
        X_features_unlabeled = extract_features(df_unlabeled, args, filtered_textual_features, numerical_features_extended)
        with open("pokec_dataset/labeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "wb") as f:
            pickle.dump(X_features_labeled, f)
        with open("pokec_dataset/unlabeled_feature_matrix_" + target_column + "_" + str(include_graph_features) + ".pk", "wb") as f:
            pickle.dump(X_features_unlabeled, f)
    
    model_name = "mean"  # "neural_net" or "ridge" or "mean" or "perfect"
    # computed sentiment scores are assumed to be innate opinions, x_star
    innate_opinions = np.array(y_label + y_unlabel_label)
    adjust_plot = True 
    if adjust_plot:
        plot_adjust(model_name)
    else: 
        run_opinion_dynamics(innate_opinions, network_lcc, df["user_id"].values, model_name, X_features_labeled, X_features_unlabeled)

if __name__ == "__main__":
    main()
