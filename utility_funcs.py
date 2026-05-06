import copy
import pickle
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from lightgbm import LGBMRegressor



SHARED_RANDOM_SEED = 2026


def seed_everything(seed=SHARED_RANDOM_SEED, use_cuda=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def load_or_create_platform_sus(agent_num, param_folder, loc=0.9, scale=0.1, seed=SHARED_RANDOM_SEED):
    param_folder = _ensure_path(param_folder)
    platform_file_path = param_folder / f"hetero_platform_sus{agent_num}.pkl"
    if platform_file_path.exists():
        with platform_file_path.open("rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed)
    platform_sus = np.clip(rng.normal(loc=loc, scale=scale, size=agent_num), 0.01, 0.99)
    with platform_file_path.open("wb") as f:
        pickle.dump(platform_sus, f)
    return platform_sus


def load_or_create_peer_sus(agent_num, param_folder, loc=0.9, scale=0.1, seed=SHARED_RANDOM_SEED):
    param_folder = _ensure_path(param_folder)
    peer_file_path = param_folder / f"hetero_peer_sus{agent_num}.pkl"
    if peer_file_path.exists():
        with peer_file_path.open("rb") as f:
            return pickle.load(f)

    rng = np.random.default_rng(seed + 1)
    peer_sus = np.clip(rng.normal(loc=loc, scale=scale, size=agent_num), 0.01, 0.99)
    with peer_file_path.open("wb") as f:
        pickle.dump(peer_sus, f)
    return peer_sus





class SigmoidMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predicting(model_name, X_features_labeled, y_label, X_features_unlabeled):
    if issparse(X_features_labeled):
        X_features_labeled = X_features_labeled.toarray()
    else:
        X_features_labeled = np.asarray(X_features_labeled)

    if issparse(X_features_unlabeled):
        X_features_unlabeled = X_features_unlabeled.toarray()
    else:
        X_features_unlabeled = np.asarray(X_features_unlabeled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_features_labeled, np.asarray(y_label), test_size=0.2, random_state=SHARED_RANDOM_SEED
    )
    

    if model_name == "ridge":
        model = Ridge(alpha=0.0)
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0.0, 1.0)
        unlabeled_preds = np.clip(model.predict(X_features_unlabeled), 0.0, 1.0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"Ordinary least square with clipping regression, RMSE: {rmse:.4f} | R2: {r2:.4f}")
    elif model_name == "lightgbm":
        model = LGBMRegressor(
            random_state=SHARED_RANDOM_SEED,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
        )
        
        model.fit(X_train, y_train)
        y_pred = np.clip(model.predict(X_test), 0.0, 1.0)
        unlabeled_preds = np.clip(model.predict(X_features_unlabeled), 0.0, 1.0)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"LightGBM regression, RMSE: {rmse:.4f} | R2: {r2:.4f}")
    else:
        mean_rmse = np.sqrt(np.mean((np.array(y_test) - np.mean(y_train)) ** 2))
        print(f"predicting mean RMSE: {mean_rmse:.4f}")
        unlabeled_preds = np.mean(y_label) * np.ones(len(X_features_unlabeled))

    return unlabeled_preds


def run_simulation(
    network,
    nodelist,
    platform_params,
    peer_params,
    retrain_steps,
    fj_steps,
    x_star,
    policy,
    model_name,
    X_features_labeled,
    X_features_unlabeled,
    approximated_equilibrium=False,
    steer_nodes=None,
    stubborn_node=None
):
    agent_num = len(x_star)
    n = int(agent_num * 0.8)
    adj_mat = nx.to_numpy_array(network, nodelist=nodelist)
    weight_mat = copy.deepcopy(adj_mat)
    # set platform_stubborn node
    if stubborn_node is not None:
        platform_params[stubborn_node] = 0.
    # case when nodes other than stubborn/steering nodes are completely stubborn towards the platform
    if steer_nodes is not None:
        platform_params_gamma0 = copy.deepcopy(platform_params)
        for i in range(agent_num):
            if i not in steer_nodes:
                platform_params_gamma0[i] = 0.0

    
    degs_inv = 1 / np.sum(adj_mat, axis=0)
    for i in range(agent_num):
        if np.isinf(degs_inv[i]) or degs_inv[i] > 1.1:
            degs_inv[i] = 0.0

    whole_record = np.zeros((agent_num, retrain_steps + 1))
    x_labeled_prior = x_star[:n].copy()
    x_unlabeled_prior = x_star[n:].copy()
    whole_record[:, 0] = copy.deepcopy(x_star)
    platform_predictions = np.zeros(agent_num)
    if policy == "steer" and steer_nodes is not None:
        whole_record_gamma0 = np.zeros((agent_num, retrain_steps + 1))
        x_labeled_prior_gamma0 = x_star[:n].copy()
        x_unlabeled_prior_gamma0 = x_star[n:].copy()
        whole_record_gamma0[:, 0] = copy.deepcopy(x_star)
        platform_predictions_gamma0 = np.zeros(agent_num)

    for t in range(retrain_steps):
        if policy == "sl":
            platform_predictions[:n] = copy.deepcopy(x_labeled_prior)
            if model_name == "perfect":
                platform_predictions[n:] = copy.deepcopy(x_unlabeled_prior)
                print("Perfect prediction")
            else:
                platform_predictions[n:] = predicting(
                    model_name, X_features_labeled, x_labeled_prior, X_features_unlabeled
                )
            
        else:
            platform_predictions[:n] = copy.deepcopy(x_labeled_prior)
            platform_predictions_gamma0[:n] = copy.deepcopy(x_labeled_prior_gamma0)
            if model_name == 'perfect':
                platform_predictions[n:] = copy.deepcopy(x_unlabeled_prior)
                platform_predictions_gamma0[n:] = copy.deepcopy(x_unlabeled_prior_gamma0)
            else: 
                platform_predictions[n:] = predicting(model_name, X_features_labeled, x_labeled_prior, X_features_unlabeled)
                platform_predictions_gamma0[n:] = predicting(model_name, X_features_labeled, x_labeled_prior_gamma0, X_features_unlabeled)
            

            if steer_nodes is not None:
                for node in steer_nodes:
                    platform_predictions[node] = 1.
                    platform_predictions_gamma0[node] = 1.
        
        x_zero = np.diag(np.ones(agent_num) - platform_params) @ x_star + platform_params * platform_predictions
        x_temp = copy.deepcopy(x_zero)
        if policy == "steer" and steer_nodes is not None:
            x_zero_gamma0 = (
                    np.diag(np.ones(agent_num) - platform_params_gamma0) @ x_star
                    + platform_params_gamma0 * platform_predictions_gamma0
                )
            x_temp_gamma0 = copy.deepcopy(x_zero_gamma0)


        for k in range(fj_steps):
            x_temp = np.diag(np.ones(agent_num) - peer_params) @ x_zero + np.diag(peer_params) @ np.diag(
                degs_inv
            ) @ weight_mat @ x_temp
            if policy == "steer" and steer_nodes is not None:
                x_temp_gamma0 = np.diag(np.ones(agent_num) - peer_params) @ x_zero_gamma0 + np.diag(peer_params) @ np.diag(
                        degs_inv
                    ) @ weight_mat @ x_temp_gamma0
        whole_record[:, t + 1] = copy.deepcopy(x_temp)
        if policy == "steer" and steer_nodes is not None:
            whole_record_gamma0[:, t + 1] = copy.deepcopy(x_temp_gamma0)

        x_labeled_prior = copy.deepcopy(x_temp[:n])
        x_unlabeled_prior = copy.deepcopy(x_temp[n:])
        if policy == "steer" and steer_nodes is not None:
            x_labeled_prior_gamma0 = copy.deepcopy(x_temp_gamma0[:n])
            x_unlabeled_prior_gamma0 = copy.deepcopy(x_temp_gamma0[n:])

            

    if policy == "steer" and steer_nodes is not None:
        return whole_record, whole_record_gamma0
    return whole_record


def _ensure_path(path_like):
    from pathlib import Path

    return path_like if isinstance(path_like, Path) else Path(path_like)


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


def run_sus_var(
    retrain_T,
    fj_K,
    DATA_DIR,
    test_sus,
    innate_opinions,
    network_lcc,
    nodelist,
    model_name,
    X_features_labeled,
    X_features_unlabeled,
):
    enforced_sus = np.arange(0.0, 1.1, 0.1)
    variances = np.zeros(len(enforced_sus))

    results_folder = DATA_DIR / "results"
    results_folder.mkdir(exist_ok=True, parents=True)
    agent_num = len(innate_opinions)
    param_folder = DATA_DIR / "parametric_params"

    for i, temp_sus in enumerate(enforced_sus):
    
        if test_sus == "peer":
            platform_sus = load_or_create_platform_sus(agent_num, param_folder)
            peer_sus = np.ones(agent_num) * temp_sus
        elif test_sus == "platform":
            peer_sus = load_or_create_peer_sus(agent_num, param_folder)
            platform_sus = np.ones(agent_num) * temp_sus
        else:
            raise ValueError("test_sus must be 'peer' or 'platform'")
        
        whole_record = run_simulation(
            network=network_lcc,
            nodelist=nodelist,
            platform_params=platform_sus,
            peer_params=peer_sus,
            retrain_steps=retrain_T,
            fj_steps=fj_K,
            x_star=innate_opinions,
            policy="sl",
            model_name=model_name,
            X_features_labeled=X_features_labeled,
            X_features_unlabeled=X_features_unlabeled,
        )

        perform_equilibrium = whole_record[:, -1]
        performative_std = perform_equilibrium.std(axis=0)
        variances[i] = performative_std ** 2

    with (results_folder / f"variance_{model_name}_{test_sus}.pk").open("wb") as f:
        pickle.dump(variances, f)
