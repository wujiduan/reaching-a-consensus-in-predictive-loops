import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from yelp_preprocessing import (
    DATA_DIR,
    FEATURE_COLUMNS,
    SHARED_RANDOM_SEED,
    TARGET_BUSINESS_SLUG,
    load_or_compute_features,
    load_or_compute_scores,
    load_yelp_dataset,
    parse_args,
)

from utility_funcs import(
    add_graph_features,
    load_or_create_peer_sus,
    load_or_create_platform_sus,
    run_simulation,
)



def main():
    args = parse_args()
    param_folder = DATA_DIR / "parametric_params"
    param_folder.mkdir(exist_ok=True, parents=True)
    results_folder = DATA_DIR / "results"
    results_folder.mkdir(exist_ok=True, parents=True)

    df, network_lcc, business = load_yelp_dataset()
    print("selected business:", business.get("name"))

    include_graph_features = False
    if include_graph_features:
        df = add_graph_features(df, graph_path=str(DATA_DIR / f"lcc_graph_acme_oyster_house.pk"))
        feature_columns = FEATURE_COLUMNS + ["deg", "clust", "pr"]
    else:
        feature_columns = FEATURE_COLUMNS

    n = int(len(df) * 0.8)
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()

    y_label, y_unlabel_label = load_or_compute_scores(
        df_labeled, df_unlabeled, param_folder, len(df), cache_key=TARGET_BUSINESS_SLUG
    )
    X_features_labeled, X_features_unlabeled = load_or_compute_features(
        df_labeled,
        df_unlabeled,
        feature_columns,
        DATA_DIR,
        f"{TARGET_BUSINESS_SLUG}_{include_graph_features}",
    )

    innate_original = np.array(y_label + y_unlabel_label, dtype=float)
    agent_num = len(innate_original)

    platform_sus = load_or_create_platform_sus(agent_num, param_folder)
    peer_sus = load_or_create_peer_sus(agent_num, param_folder)

    rng = np.random.default_rng(SHARED_RANDOM_SEED)
    # set a peer-stubborn individual
    stubborn_idx = int(rng.integers(low=n, high=agent_num))
    
    peer_sus_modified = peer_sus.copy()
    peer_sus_modified[stubborn_idx] = 0.0
    # set to zero for easier illustration. 
    innate_original[stubborn_idx] = 0.0
    # modify the innate opinions
    innate_modified = innate_original.copy()
    labeled_override = max(1, int(0.1 * n))
    override_idx = rng.choice(np.arange(n), size=labeled_override, replace=False)
    innate_modified[override_idx] = 1.0

    retrain_T = 100
    fj_K = 100
    selected_steps = 50
    nodelist = df["user_id"].values
    labels = {"mean": "Mean", "perfect": "Perfect", "ridge": "OLS", "neural_net": "MLP", "lightgbm": "LightGBM"}

    for model_name in ["mean", "ridge", "neural_net", "lightgbm"]:
        modified_path = results_folder / f"{model_name}_sl_modified_stubborn_whole_record{retrain_T}.pk"
        baseline_path = results_folder / f"{model_name}_sl_original_whole_record{retrain_T}.pk"
        stubborn_path = param_folder / f"stubborn_unlabeled_node_{agent_num}.pkl"

        if modified_path.exists() and baseline_path.exists() and stubborn_path.exists():
            print("Results already exist. Skipping simulation.")
            with modified_path.open("rb") as f:
                modified_record = pickle.load(f)
            with baseline_path.open("rb") as f:
                baseline_record = pickle.load(f)
            with stubborn_path.open("rb") as f:
                stubborn_idx = pickle.load(f)
        else:
            baseline_record = run_simulation(
                network=network_lcc,
                nodelist=nodelist,
                platform_params=platform_sus,
                peer_params=peer_sus_modified,
                steer_nodes=None,
                stubborn_node=None,
                retrain_steps=retrain_T,
                fj_steps=fj_K,
                x_star=innate_original,
                policy="sl",
                model_name=model_name,
                X_features_labeled=X_features_labeled,
                X_features_unlabeled=X_features_unlabeled,
            )

            modified_record = run_simulation(
                network=network_lcc,
                nodelist=nodelist,
                platform_params=platform_sus,
                peer_params=peer_sus_modified,
                steer_nodes=None,
                stubborn_node=None,
                retrain_steps=retrain_T,
                fj_steps=fj_K,
                x_star=innate_modified,
                policy="sl",
                model_name=model_name,
                X_features_labeled=X_features_labeled,
                X_features_unlabeled=X_features_unlabeled,
            )

            with modified_path.open("wb") as f:
                pickle.dump(modified_record, f)
            with baseline_path.open("wb") as f:
                pickle.dump(baseline_record, f)
            with stubborn_path.open("wb") as f:
                pickle.dump(stubborn_idx, f)

        x = np.arange(selected_steps + 1)
        stubborn_series = modified_record[stubborn_idx, :selected_steps + 1]
        original_series = baseline_record[stubborn_idx, :selected_steps + 1]

        plt.figure()
        plt.plot(
            x,
            stubborn_series,
            color="tab:blue",
            label=labels[model_name] + r" $\tilde{x}^*$",
            marker="o",
            markersize=3,
        )
        plt.plot(
            x,
            original_series,
            color="tab:orange",
            label=labels[model_name] + r" $x^*$",
            marker="o",
            markersize=3,
        )
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.ylabel(r"Opinion $(x_{ex}^{(t)})_q$", fontsize=18)
        plt.xticks(np.arange(0, selected_steps + 1, 10), fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc="best", frameon=False, fontsize=15)

        out_path = param_folder / f"{model_name}_parametric_sl_retrain_steps_stubborn_unlabeled.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"stubborn unlabeled index: {stubborn_idx} (global index)")
        print(f"modified 10% labeled count: {labeled_override}")
        print(f"saved figure: {out_path}")


if __name__ == "__main__":
    main()
