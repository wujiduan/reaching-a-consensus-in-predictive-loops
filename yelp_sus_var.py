import os
import pickle

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

from utility_funcs import (
    run_simulation,
    load_or_create_peer_sus,
    load_or_create_platform_sus,
    add_graph_features,
    run_sus_var,
)





def create_plot():
    param_folder = DATA_DIR / "parametric_params"
    enforced_sus = np.arange(0.0, 1.1, 0.1)

    
    models = ["perfect", "ridge", "mean", "lightgbm"]
    labels = ["Perfect", "OLS", "Mean", "LightGBM"]
    # models = ["perfect"]
    # labels = ["Perfect"]

    colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]

    fig, ax = plt.subplots()
    for i, model in enumerate(models):
        with (DATA_DIR / "results" / f"variance_{model}_platform.pk").open("rb") as f:
            variances = pickle.load(f)
        ax.plot(enforced_sus, variances, linewidth=1.5, label=labels[i], color=colors[i], linestyle="-", marker="o")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(enforced_sus)
    ax.set_xlabel(r"Platform susceptibility $\beta$", fontsize=18)
    ax.set_ylabel(r"Var($x_{PS}$)", fontsize=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(loc="lower left", frameon=False, fontsize=15)
    plt.savefig(param_folder / f"platform_variance_sl.pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    for i, model in enumerate(models):
        with (DATA_DIR / "results" / f"variance_{model}_peer.pk").open("rb") as f:
            variances = pickle.load(f)
        ax.plot(enforced_sus, variances, linewidth=1.5, label=labels[i], color=colors[i], linestyle="-", marker="o")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xticks(enforced_sus)
    ax.set_xlabel(r"Peer susceptibility $\alpha$", fontsize=18)
    ax.set_ylabel(r"Var($x_{PS}$)", fontsize=18)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(loc="upper right", frameon=False, fontsize=15)
    plt.savefig(param_folder / f"peer_variance_sl.pdf", bbox_inches="tight")


def main():
    df, network_lcc, business = load_yelp_dataset()
    print("selected business:", business.get("name"))

    include_graph_features = False
    if include_graph_features:
        df = add_graph_features(df, graph_path=str(DATA_DIR / "lcc_graph_acme_oyster_house.pk"))
        feature_columns = FEATURE_COLUMNS + ["deg", "clust", "pr"]
    else:
        feature_columns = FEATURE_COLUMNS

    n = int(len(df) * 0.8)
    df_labeled = df.iloc[:n].copy()
    df_unlabeled = df.iloc[n:].copy()

    param_folder = DATA_DIR / "parametric_params"
    param_folder.mkdir(exist_ok=True, parents=True)

    y_label, y_unlabel_label = load_or_compute_scores(
        df_labeled, df_unlabeled, param_folder, len(df), cache_key=TARGET_BUSINESS_SLUG
    )
    X_features_labeled, X_features_unlabeled = load_or_compute_features(
        df_labeled,
        df_unlabeled,
        feature_columns,
        DATA_DIR,
        TARGET_BUSINESS_SLUG,
        include_graph_features
    )

    innate_opinions = np.array(y_label + y_unlabel_label, dtype=float)
    adjust_plot = True
    retrain_T = 50
    fj_K = 100
    if adjust_plot:
        create_plot()
    else:
        for model_name in ["perfect", "ridge", "mean", "lightgbm"]:
            for test_sus in ["platform", "peer"]:
                run_sus_var(
                    retrain_T=retrain_T,
                    fj_K=fj_K,
                    DATA_DIR=DATA_DIR,
                    test_sus=test_sus,
                    innate_opinions=innate_opinions,
                    network_lcc=network_lcc,
                    nodelist=df["user_id"].values,
                    model_name=model_name,
                    X_features_labeled=X_features_labeled,
                    X_features_unlabeled=X_features_unlabeled,
                )
        create_plot()


if __name__ == "__main__":
    main()
