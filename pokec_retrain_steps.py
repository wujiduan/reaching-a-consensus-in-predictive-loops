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
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

from utility_funcs import (
    load_or_create_peer_sus,
    load_or_create_platform_sus,
    predicting,
    run_simulation,
    seed_everything,
)
from pokec_preprocessing import (
    DATA_DIR as POKEC_DATA_DIR,
    load_or_compute_features,
    load_or_compute_scores,
    load_profiles_and_graph,
    parse_args,
)


SHARED_RANDOM_SEED = 2026


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

def add_graph_features(df, network_lcc):
    
    
    degree = dict(network_lcc.degree())
    clustering = nx.clustering(network_lcc)
    pagerank = nx.pagerank(network_lcc, alpha=0.85)

    df["deg"] = df["user_id"].map(degree).fillna(0)
    df["clust"] = df["user_id"].map(clustering).fillna(0)
    df["pr"] = df["user_id"].map(pagerank).fillna(0)
    return df

def run_opinion_dynamics(innate_opinions, network_lcc, nodelist, model_name, X_features_labeled, X_features_unlabeled, policy, strong_perform, include_graph_features):
    
    agent_num = len(innate_opinions)
    # this is to reveal the steer effect on the stubborn node.
    # innate_opinions = np.zeros(agent_num)
    fj_K = 100
    retrain_T = 100
    x_initial = copy.deepcopy(innate_opinions)

    param_folder = POKEC_DATA_DIR / "parametric_params"
    param_folder.mkdir(exist_ok=True, parents=True)

    platform_file_path = param_folder / f"hetero_platform_sus{agent_num}.pkl"
    peer_sus = load_or_create_peer_sus(agent_num, param_folder)
    platform_sus = load_or_create_platform_sus(agent_num, param_folder)
    steer_node_file_path = param_folder / f"steer_node_{agent_num}.pkl"

    if steer_node_file_path.exists():
        
        if policy == "sl":
            steer_nodes = []
            stubborn_node = None
        elif policy == "steer":
            with (param_folder / f"steer_node_{agent_num}.pkl").open("rb") as file:
                steer_nodes = pickle.load(file)
            with (param_folder / f"stubborn_node_{agent_num}.pkl").open("rb") as file:
                stubborn_node = pickle.load(file)

    else:
        
        if policy == "steer":
            rng = np.random.default_rng(SHARED_RANDOM_SEED)
            steer_size = int(agent_num / 10)
            selected_nodes = rng.choice(agent_num, size=steer_size + 1, replace=False)
            steer_nodes = selected_nodes[:-1]
            stubborn_node = selected_nodes[-1]
            with (param_folder / f"steer_node_{agent_num}.pkl").open("wb") as file:
                pickle.dump(steer_nodes, file)
            with (param_folder / f"stubborn_node_{agent_num}.pkl").open("wb") as file:
                pickle.dump(stubborn_node, file)
        else:
            steer_nodes = []
            stubborn_node = None

    if strong_perform:
        results_folder = POKEC_DATA_DIR / "results_strong_perform"
        platform_sus = np.ones(agent_num)
    else:
        results_folder = POKEC_DATA_DIR / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    if not include_graph_features:
        record_path = results_folder / f"{model_name}_{policy}_whole_record{retrain_T}.pk"
        gamma0_path = results_folder / f"{model_name}_{policy}_gamma0_whole_record{retrain_T}.pk"
    else:
        record_path = results_folder / f"{model_name}_{policy}_whole_record{retrain_T}_graph_feature.pk"
        gamma0_path = results_folder / f"{model_name}_{policy}_gamma0_whole_record{retrain_T}_graph_feature.pk"
    
    if record_path.exists():
        with record_path.open("rb") as f:
            whole_opinions = pickle.load(f)
       
    else:
        
        if policy == "steer":
            whole_opinions, whole_opinions_gamma0 = run_simulation(network=network_lcc, nodelist=nodelist, platform_params=platform_sus, 
                                                peer_params=peer_sus, 
                                                steer_nodes=steer_nodes, fj_steps=fj_K, retrain_steps=retrain_T, 
                                                x_star=innate_opinions, policy=policy, model_name=model_name, 
                                                X_features_labeled=X_features_labeled, X_features_unlabeled=X_features_unlabeled)
        else:
            whole_opinions = run_simulation(network=network_lcc, nodelist=nodelist, platform_params=platform_sus, 
                                                peer_params=peer_sus, 
                                                steer_nodes=steer_nodes, fj_steps=fj_K, retrain_steps=retrain_T, 
                                                x_star=innate_opinions, policy=policy, model_name=model_name, 
                                                X_features_labeled=X_features_labeled, X_features_unlabeled=X_features_unlabeled)
       
        with record_path.open("wb") as f:
            pickle.dump(whole_opinions, f)
        if policy == "steer":
            with gamma0_path.open("wb") as f:
                pickle.dump(whole_opinions_gamma0, f)

    
    


def plot_adjust(innate_opinions, policy, strong_perform, include_graph_features):
    agent_num = len(innate_opinions)
    retrain_T = 100
    if strong_perform:
        results_folder = "pokec_dataset/results_strong_perform/"
    else:
        results_folder = "pokec_dataset/results/"
    param_folder = "pokec_dataset/parametric_params/"

    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]
    models = ["perfect", "ridge", "mean", "lightgbm"]

   
    x = np.arange(0, retrain_T+1)
    if policy == "steer":
    
        labels = ["Perfect", "OLS", "Mean", "LightGBM"]

        with open(param_folder + "stubborn_node_" + str(agent_num) + ".pkl", "rb") as file:
            stubborn_node = pickle.load(file)

        
        with open(results_folder + "perfect_steer_gamma0_whole_record" + str(retrain_T) + ".pk", "rb") as f:
            x_psl_gamma0 = pickle.load(f)
            x_psl_gamma0 = x_psl_gamma0[stubborn_node, -1]

        plt.hlines(y=x_psl_gamma0, 
                    xmin=0, 
                    xmax=retrain_T, 
                    linestyle='--', 
                    label=r"$(x_{ex}^{(T)})_l$" + "(Perfect,\n" + r"$\beta_k=0,k\notin \{l\}\cup S$)", 
                    color='brown')
        
        for i in range(len(models)):
            
            if os.path.exists(results_folder + models[i] + "_steer_whole_record" + str(retrain_T) + ".pk"):
                with open(results_folder + models[i] + "_steer_whole_record" + str(retrain_T) + ".pk", "rb") as f:
                    whole_opinions = pickle.load(f)
            
            plt.plot(x, whole_opinions[stubborn_node, :], label=labels[i], color=colors[i])

        plt.xticks(range(0, retrain_T+1, 10), fontsize=12)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.ylabel(r"Opinion $(x_{ex}^{(t)})_l$", fontsize=18)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", bbox_to_anchor=(1,0.1), frameon=False, fontsize=12)
        
        plt.savefig(param_folder + "all_parametric_steer_retrain_steps.pdf", bbox_inches='tight')
    else:
        # for supervised learning policy 
        labels = ["Perfect", "OLS", "Mean", "LightGBM"]
        
        fig, ax = plt.subplots()
        step_gap = 15
        box_group_width = 7.5
        
        positions_base = np.arange(retrain_T + 1) * step_gap
        offsets = np.linspace(
            -box_group_width / 2, box_group_width / 2, len(models), endpoint=False
        ) + (box_group_width / len(models)) / 2
        box_width = 0.85 * (box_group_width / len(models))

        df = {}
        for i in range(len(models)):
            
            if not include_graph_features:
                record_path = results_folder + models[i] + "_" + policy + "_whole_record" + str(retrain_T) + ".pk"
            else:
                record_path = results_folder + models[i] + "_" + policy + "_whole_record" + str(retrain_T) + "_graph_feature.pk"
            if os.path.exists(record_path):
                with open(record_path, "rb") as f:
                    whole_opinions = pickle.load(f)
                
            df[labels[i]] = whole_opinions[:, :]
                
                
        
        expanded_rows = []
        for model_name, temp_opinions in df.items():
            temp_df = pd.DataFrame(temp_opinions.T)
            temp_df_expanded = temp_df.melt(var_name="sample", value_name="value", ignore_index=False)
            temp_df_expanded = temp_df_expanded.rename_axis("time").reset_index()
            temp_df_expanded["model"] = model_name
            expanded_rows.append(temp_df_expanded)
        
        all_rows = pd.concat(expanded_rows, ignore_index=True)

        stats = (
            all_rows.groupby(["time", "model"])["value"]
            .agg(mean="mean", var="var")
            .reset_index()
        )
        stats["std"] = np.sqrt(stats["var"])   # use variance-derived error bars


        models_u = [m for m in labels if m in stats["model"].unique()]
        print(models_u)


        for m in models_u:
            s = stats[stats["model"] == m].copy()
            i = labels.index(m)
            x = positions_base + offsets[i]
            ax.errorbar(
                x, s["mean"], yerr=s["std"],
                fmt="s",            # square marker ("box")
                linestyle="none",   # no line between steps
                elinewidth=box_width*0.3,       # error bar line width
                capthick=box_width*0.25,      # error bar cap thickness
                markeredgewidth=box_width*0.3,  # marker edge width
                markersize=box_width*0.5,       # marker size
                capsize=box_width*0.4,
                label=m,
                color=colors[i]
            )
        
        ax.set_xticks(positions_base[::10])
        ax.set_xticklabels(np.arange(retrain_T + 1)[::10], fontsize=12)

        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.ylabel(r"Opinion after peer interaction, $x_{ex}^{(t)}$", fontsize=15)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.yticks(fontsize=12)
        
        plt.legend(loc="upper right", 
                    bbox_to_anchor=(1,1), 
                    frameon=False, 
                    fontsize=15, 
                    columnspacing=0.2, 
                    labelspacing=0.2, 
                    borderpad=0.2, 
                    handletextpad=0.2,
                    markerscale=3)
        if not include_graph_features:
            plt.savefig(param_folder + "all_parametric_sl_retrain_steps.pdf", bbox_inches='tight')
        else:
            plt.savefig(param_folder + "all_parametric_sl_retrain_steps_graph_feature.pdf", bbox_inches='tight')
def main():
    args = parse_args()
    seed_everything(use_cuda=bool(args.device and "cuda" in args.device))
    target_column = "relation_to_smoking"
    include_graph_features = False
    
    df, network_lcc = load_profiles_and_graph(target_column)
    if include_graph_features:
        df = add_graph_features(df, network_lcc)
        
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
    
    model_name = "mean"  # "ridge" or "mean" or "perfect" or "lightgbm"
    
    # computed sentiment scores are assumed to be innate opinions, x_star
    innate_opinions = np.array(y_label + y_unlabel_label)
    adjust_plot = True
    policy = "steer"  # "sl" for supervised learning, "steer" for steering
    strong_perform = False  # when it's true, platform_sus = 1 for all individuals
    if adjust_plot:
        plot_adjust(innate_opinions, policy, strong_perform, include_graph_features)
    else: 
        for model_name in ["perfect", "ridge", "mean", "lightgbm"]:
            run_opinion_dynamics(innate_opinions, network_lcc, df["user_id"].values, model_name, X_features_labeled, X_features_unlabeled, policy, strong_perform, include_graph_features)
        plot_adjust(innate_opinions, policy, strong_perform, include_graph_features)

if __name__ == "__main__":
    main()
