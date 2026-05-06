import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility_funcs import(
    add_graph_features,
    run_simulation,
    load_or_create_peer_sus,
    load_or_create_platform_sus,
)

from yelp_preprocessing import (
    DATA_DIR,
    FEATURE_COLUMNS,
    TARGET_BUSINESS_SLUG,
    SHARED_RANDOM_SEED,
    load_or_compute_features,
    load_or_compute_scores,
    load_yelp_dataset,
    parse_args,
)


def run_opinion_dynamics(
    innate_opinions,
    network_lcc,
    nodelist,
    model_name,
    X_features_labeled,
    X_features_unlabeled,
    policy,
    strong_perform,
    include_graph_features
):
    agent_num = len(innate_opinions)
    fj_K = 100
    retrain_T = 100
    x_initial = copy.deepcopy(innate_opinions)

    param_folder = DATA_DIR / "parametric_params"
    param_folder.mkdir(exist_ok=True, parents=True)

    platform_file_path = param_folder / f"hetero_platform_sus{agent_num}.pkl"
    stubborn_file_path = param_folder / f"stubborn_node_{agent_num}.pkl"
    platform_sus = load_or_create_platform_sus(agent_num, param_folder)
    peer_sus = load_or_create_peer_sus(agent_num, param_folder)
    
    if not stubborn_file_path.exists():
        if policy == "steer":
            
            steer_size = int(agent_num / 10)
            rng = np.random.default_rng(SHARED_RANDOM_SEED)
            selected_nodes = rng.choice(agent_num, size=steer_size + 1, replace=False)
            steer_nodes = selected_nodes[:-1]
            stubborn_node = selected_nodes[-1]
            platform_sus[stubborn_node] = 0.0
            with platform_file_path.open("wb") as file:
                pickle.dump(platform_sus, file)
            with (param_folder / f"steer_node_{agent_num}.pkl").open("wb") as file:
                pickle.dump(steer_nodes, file)
            with (param_folder / f"stubborn_node_{agent_num}.pkl").open("wb") as file:
                pickle.dump(stubborn_node, file)
        else:
            steer_nodes = []
            stubborn_node = None
    else:
        
        if policy == "steer":
            with (param_folder / f"steer_node_{agent_num}.pkl").open("rb") as file:
                steer_nodes = pickle.load(file)
            with (param_folder / f"stubborn_node_{agent_num}.pkl").open("rb") as file:
                stubborn_node = pickle.load(file)
        else:
            steer_nodes = []
            stubborn_node = None


    if strong_perform:
        results_folder = DATA_DIR / "results_strong_perform"
        results_folder.mkdir(exist_ok=True, parents=True)
        platform_sus = np.ones(agent_num)
    else:
        results_folder = DATA_DIR / "results"
        results_folder.mkdir(exist_ok=True, parents=True)

    if not include_graph_features:
        record_path = results_folder / f"{model_name}_{policy}_whole_record{retrain_T}.pk"
        gamma0_path = results_folder / f"{model_name}_{policy}_gamma0_whole_record{retrain_T}.pk"
    else:
        record_path = results_folder / f"{model_name}_{policy}_whole_record{retrain_T}_graph_features.pk"
        gamma0_path = results_folder / f"{model_name}_{policy}_gamma0_whole_record{retrain_T}_graph_features.pk"

    if record_path.exists() and gamma0_path.exists():
        with record_path.open("rb") as f:
            whole_opinions = pickle.load(f)
        with gamma0_path.open("rb") as f:
            whole_opinions_gamma0 = pickle.load(f)
    else:
        simulation_out = run_simulation(
            network=network_lcc,
            nodelist=nodelist,
            platform_params=platform_sus,
            peer_params=peer_sus,
            steer_nodes=steer_nodes if policy == "steer" else None,
            stubborn_node = stubborn_node,
            retrain_steps=retrain_T,
            fj_steps=fj_K,
            x_star=innate_opinions,
            policy=policy,
            model_name=model_name,
            X_features_labeled=X_features_labeled,
            X_features_unlabeled=X_features_unlabeled,
        )
        if isinstance(simulation_out, tuple):
            whole_opinions, whole_opinions_gamma0 = simulation_out
        else:
            whole_opinions = simulation_out
            # whole_opinions_gamma0 = simulation_out

        with record_path.open("wb") as f:
            pickle.dump(whole_opinions, f)
        if policy == "steer":
            with gamma0_path.open("wb") as f:
                pickle.dump(whole_opinions_gamma0, f)

    if policy == "sl" and model_name == "perfect":
        with (results_folder / "perfect_equilibrium.pk").open("wb") as f:
            pickle.dump(whole_opinions[:, -1], f)
        with (results_folder / "perfect_FJequilibrium.pk").open("wb") as f:
            pickle.dump(whole_opinions[:, 1], f)

    


def plot_adjust(innate_opinions, policy, strong_perform, include_graph_features):
    agent_num = len(innate_opinions)
    retrain_T = 100
    selected_steps = 50
    if strong_perform:
        results_folder = DATA_DIR / "results_strong_perform/"
    else:
        results_folder = DATA_DIR / "results/"
    param_folder = DATA_DIR / "parametric_params"

    colors = ["tab:blue", "tab:orange",  "tab:red", "tab:purple"]
    models = ["perfect", "ridge", "mean", "lightgbm"]

    
    x = np.arange(0, selected_steps + 1)
    if policy == "steer":
        labels = ["Perfect", "OLS", "Mean", "LightGBM"]

        with (param_folder / f"stubborn_node_{agent_num}.pkl").open("rb") as file:
            stubborn_node = pickle.load(file)
        
        with (results_folder / f"perfect_steer_gamma0_whole_record{retrain_T}.pk").open("rb") as f:
            x_psl_gamma0 = pickle.load(f)
            x_psl_gamma0 = x_psl_gamma0[stubborn_node, -1]

        plt.hlines(
            y=x_psl_gamma0,
            xmin=0,
            xmax=selected_steps,
            linestyle="--",
            label=r"$(x_{ex}^{(T)})_l$" + "(Perfect,\n" + r"$\beta_k=0,k\notin \{l\}\cup S$)",
            color="brown",
        )

        for i, model in enumerate(models):
            path = results_folder / f"{model}_steer_whole_record{retrain_T}.pk"
            if path.exists():
                with path.open("rb") as f:
                    whole_opinions = pickle.load(f)
                plt.plot(x, whole_opinions[stubborn_node, :selected_steps+1], label=labels[i], color=colors[i])

        plt.xticks(range(0, selected_steps + 1, 10), fontsize=12)
        plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        plt.ylabel(r"Opinion $(x_{ex}^{(t)})_l$", fontsize=18)
        plt.xlabel(r"Retraining step $t$", fontsize=18)
        plt.yticks(fontsize=12)
        plt.legend(loc="lower right", bbox_to_anchor=(1, 0.00), frameon=False, fontsize=12)
        plt.savefig(param_folder / "all_parametric_steer_retrain_steps.pdf", bbox_inches="tight")
        return

    labels = ["Perfect", "OLS", "Mean", "LightGBM"]

    fig, ax = plt.subplots()
    step_gap = 40
    box_group_width = 20

    positions_base = np.arange(selected_steps + 1) * step_gap
    offsets = np.linspace(
        -box_group_width / 2, box_group_width / 2, len(models), endpoint=False
    ) + (box_group_width / len(models)) / 2
    box_width = 0.35 * (box_group_width / len(models))

    
    model_rows = {}
    for i, model in enumerate(models):
        if include_graph_features:
            path = results_folder / f"{model}_{policy}_whole_record{retrain_T}_graph_features.pk"
        else:
            path = results_folder / f"{model}_{policy}_whole_record{retrain_T}.pk"
        if path.exists():
            with path.open("rb") as f:
                whole_opinions = pickle.load(f)
            model_rows[labels[i]] = whole_opinions[:, :selected_steps+1]

    expanded_rows = []
    for model_name, temp_opinions in model_rows.items():
        temp_df = pd.DataFrame(temp_opinions.T)
        temp_df_expanded = temp_df.melt(var_name="sample", value_name="value", ignore_index=False)
        temp_df_expanded = temp_df_expanded.rename_axis("time").reset_index()
        temp_df_expanded["model"] = model_name
        expanded_rows.append(temp_df_expanded)

    all_rows = pd.concat(expanded_rows, ignore_index=True)
    stats = all_rows.groupby(["time", "model"])["value"].agg(mean="mean", var="var").reset_index()
    stats["std"] = np.sqrt(stats["var"])

    models_u = [m for m in labels if m in stats["model"].unique()]
    print(models_u)
    
    for m in models_u:
        s = stats[stats["model"] == m].copy()
        i = labels.index(m)
        x = positions_base + offsets[i]
        print("position shape:", positions_base.shape)
        print("data shape:", s["mean"].shape)
        ax.errorbar(
            x,
            s["mean"],
            yerr=s["std"],
            fmt="s",
            linestyle="none",
            elinewidth=box_width * 0.3,
            capthick=box_width * 0.25,
            markeredgewidth=box_width*0.25,
            markersize=box_width*0.25,
            capsize=box_width * 0.3,
            label=m,
            color=colors[i],
        )

    ax.set_xticks(positions_base[::10])
    ax.set_xticklabels(np.arange(selected_steps + 1)[::10], fontsize=12)
    ax.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.ylabel(r"Opinion after peer interaction, $x_{ex}^{(t)}$", fontsize=15)
    plt.xlabel(r"Retraining step $t$", fontsize=18)
    plt.yticks(fontsize=12)
    leg = plt.legend(
        loc="upper right",
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=15,
        columnspacing=0.2,
        labelspacing=0.2,
        borderpad=0.2,
        handletextpad=0.2,
        markerscale=8,
    )
    for handle in leg.legend_handles:
        handle.set_linewidth(1)

    if include_graph_features:
        plt.savefig(param_folder / "all_parametric_sl_retrain_steps_graph_features.pdf", bbox_inches="tight")
    else:
        plt.savefig(param_folder / "all_parametric_sl_retrain_steps.pdf", bbox_inches="tight")


def main():
    args = parse_args()
    df, network_lcc, business = load_yelp_dataset()
    print("selected business:", business.get("name"))

    include_graph_features = False
    if include_graph_features:
        df = add_graph_features(df, graph_path=str(DATA_DIR / f"lcc_graph_{TARGET_BUSINESS_SLUG}.pk"))
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
    policy = "steer"
    strong_perform = False
    if adjust_plot:
        plot_adjust(innate_opinions, policy, strong_perform, include_graph_features)
    else:
        for model_name in ["perfect", "ridge", "mean", "lightgbm"]:
            run_opinion_dynamics(
                innate_opinions,
                network_lcc,
                df["user_id"].values,
                model_name,
                X_features_labeled,
                X_features_unlabeled,
                policy,
                strong_perform,
                include_graph_features
            )
        plot_adjust(innate_opinions, policy, strong_perform, include_graph_features)


if __name__ == "__main__":
    main()
