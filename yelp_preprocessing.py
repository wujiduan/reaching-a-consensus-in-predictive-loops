import argparse
import ast
import copy
import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import issparse
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("yelp_dataset")
REVIEW_PATH = DATA_DIR / "yelp_academic_dataset_review.json"
USER_PATH = DATA_DIR / "yelp_academic_dataset_user.json"
BUSINESS_PATH = DATA_DIR / "yelp_academic_dataset_business.json"
SHARED_RANDOM_SEED = 2026

TARGET_BUSINESS_ID = "_ab50qdWOk0DdB6XOrBitw"
TARGET_BUSINESS_NAME = "Acme Oyster House"
TARGET_BUSINESS_SLUG = "acme_oyster_house_v3"

USER_NUMERIC_FEATURES = [
    "user_review_count_excl_target",
    "user_useful",
    "user_funny",
    "user_cool",
    "user_fans",
    "user_average_stars_excl_target",
    "user_elite_count",
    "user_friend_count",
    "user_yelping_since_year",
    "user_yelping_since_month",
    "user_compliment_hot",
    "user_compliment_more",
    "user_compliment_profile",
    "user_compliment_cute",
    "user_compliment_list",
    "user_compliment_note",
    "user_compliment_plain",
    "user_compliment_cool",
    "user_compliment_funny",
    "user_compliment_writer",
    "user_compliment_photos",
]

BUSINESS_NUMERIC_FEATURES = [
    "business_stars",
    "business_review_count_excl_target",
    "business_is_open",
    "business_latitude",
    "business_longitude",
    "business_categories_count",
    "business_attribute_count",
    "business_price_range",
]

FEATURE_COLUMNS = USER_NUMERIC_FEATURES + BUSINESS_NUMERIC_FEATURES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="English sentence embedding model used if text features are enabled",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu")
    return parser.parse_args()


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _count_csv_items(value):
    if value in (None, "", "None", "null", "nan"):
        return 0
    return len([item for item in str(value).split(",") if item.strip()])


def _parse_datetime(value):
    if value in (None, "", "None", "null", "nan"):
        return None
    try:
        return datetime.strptime(str(value), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _parse_business_attributes(attributes):
    if attributes in (None, "", "None", "null", "nan"):
        return {}
    if isinstance(attributes, dict):
        return attributes
    try:
        parsed = ast.literal_eval(str(attributes))
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def _target_paths():
    profiles_path = DATA_DIR / f"lcc_profiles_{TARGET_BUSINESS_SLUG}.pk"
    graph_path = DATA_DIR / f"lcc_graph_{TARGET_BUSINESS_SLUG}.pk"
    return profiles_path, graph_path


def load_target_business(business_id=TARGET_BUSINESS_ID):
    with BUSINESS_PATH.open() as f:
        for line in f:
            business = json.loads(line)
            if business.get("business_id") == business_id:
                return business
    raise ValueError(f"Business {business_id} not found in Yelp business file.")


def load_target_reviews(business_id=TARGET_BUSINESS_ID):
    reviews_by_user = {}
    with REVIEW_PATH.open() as f:
        for line in f:
            review = json.loads(line)
            if review.get("business_id") != business_id:
                continue
            user_id = review["user_id"]
            reviews_by_user.setdefault(user_id, []).append(
                {
                    "review_id": review.get("review_id"),
                    "stars": _to_float(review.get("stars")),
                    "date": review.get("date"),
                    "text": review.get("text"),
                }
            )
    total_reviews = sum(len(v) for v in reviews_by_user.values())
    if total_reviews < 2000:
        raise ValueError(
            f"Selected business has only {total_reviews} reviews; "
            "need at least 2000."
        )
    return reviews_by_user


def load_user_review_stats_excluding_target(target_business_id=TARGET_BUSINESS_ID):
    review_count = {}
    rating_sum = {}
    with REVIEW_PATH.open() as f:
        for line in f:
            review = json.loads(line)
            if review.get("business_id") == target_business_id:
                continue
            user_id = review.get("user_id")
            rating = _to_float(review.get("stars"))
            review_count[user_id] = review_count.get(user_id, 0) + 1
            rating_sum[user_id] = rating_sum.get(user_id, 0.0) + rating
    return review_count, rating_sum


def build_business_feature_row(business, target_review_count):
    attributes = _parse_business_attributes(business.get("attributes"))
    categories = str(business.get("categories") or "")
    category_count = len([item for item in categories.split(",") if item.strip()])
    return {
        "business_stars": _to_float(business.get("stars")),
        "business_review_count_excl_target": max(
            0.0, _to_float(business.get("review_count")) - float(target_review_count)
        ),
        "business_is_open": _to_float(business.get("is_open")),
        "business_latitude": _to_float(business.get("latitude")),
        "business_longitude": _to_float(business.get("longitude")),
        "business_categories_count": float(category_count),
        "business_attribute_count": float(len(attributes)),
        "business_price_range": float(
            _to_int(attributes.get("RestaurantsPriceRange2"), default=0)
        ),
    }


def build_user_feature_row(user, rating, business_features, review_count_excl_target, rating_sum_excl_target):
    yelping_since = _parse_datetime(user.get("yelping_since"))
    user_id = user.get("user_id")
    excl_count = float(review_count_excl_target.get(user_id, 0))
    excl_avg = rating_sum_excl_target.get(user_id, 0.0) / excl_count if excl_count > 0 else 0.0
    return {
        "user_id": user_id,
        "rating_raw": _to_float(rating),
        "rating": (_to_float(rating) - 1.0) / 4.0,
        "user_review_count_excl_target": excl_count,
        "user_useful": _to_float(user.get("useful")),
        "user_funny": _to_float(user.get("funny")),
        "user_cool": _to_float(user.get("cool")),
        "user_fans": _to_float(user.get("fans")),
        "user_average_stars_excl_target": excl_avg,
        "user_elite_count": float(_count_csv_items(user.get("elite"))),
        "user_friend_count": float(_count_csv_items(user.get("friends"))),
        "user_yelping_since_year": float(yelping_since.year if yelping_since else 0),
        "user_yelping_since_month": float(yelping_since.month if yelping_since else 0),
        "user_compliment_hot": _to_float(user.get("compliment_hot")),
        "user_compliment_more": _to_float(user.get("compliment_more")),
        "user_compliment_profile": _to_float(user.get("compliment_profile")),
        "user_compliment_cute": _to_float(user.get("compliment_cute")),
        "user_compliment_list": _to_float(user.get("compliment_list")),
        "user_compliment_note": _to_float(user.get("compliment_note")),
        "user_compliment_plain": _to_float(user.get("compliment_plain")),
        "user_compliment_cool": _to_float(user.get("compliment_cool")),
        "user_compliment_funny": _to_float(user.get("compliment_funny")),
        "user_compliment_writer": _to_float(user.get("compliment_writer")),
        "user_compliment_photos": _to_float(user.get("compliment_photos")),
        **business_features,
    }


def load_yelp_dataset(business_id=TARGET_BUSINESS_ID, random_state=SHARED_RANDOM_SEED):
    profiles_path, graph_path = _target_paths()
    business = load_target_business(business_id)

    if profiles_path.exists() and graph_path.exists():
        with profiles_path.open("rb") as f:
            df = pickle.load(f)
        with graph_path.open("rb") as f:
            network_lcc = pickle.load(f)
        print("lcc user num:", len(df))
        return df, network_lcc, business

    reviews_by_user = load_target_reviews(business_id)
    review_count_excl_target, rating_sum_excl_target = load_user_review_stats_excluding_target(business_id)
    reviewer_ids = set(reviews_by_user)
    target_review_count = sum(len(v) for v in reviews_by_user.values())
    business_features = build_business_feature_row(business, target_review_count)

    reviewer_rows = []
    network_g = nx.Graph()
    network_g.add_nodes_from(reviewer_ids)

    with USER_PATH.open() as f:
        for line in f:
            user = json.loads(line)
            user_id = user.get("user_id")
            if user_id not in reviewer_ids:
                continue

            user_reviews = reviews_by_user[user_id]
            user_rating = float(np.mean([review["stars"] for review in user_reviews]))
            reviewer_rows.append(
                build_user_feature_row(
                    user,
                    user_rating,
                    business_features,
                    review_count_excl_target,
                    rating_sum_excl_target,
                )
            )

            friends = user.get("friends") or ""
            if friends:
                for friend in friends.split(", "):
                    if friend in reviewer_ids:
                        network_g.add_edge(user_id, friend)

    df = pd.DataFrame(reviewer_rows)
    if df.empty:
        raise ValueError("No Yelp reviewers were loaded for the selected business.")

    components = list(nx.connected_components(network_g))
    if not components:
        raise ValueError("Reviewer friendship graph is empty.")
    lcc = max(components, key=len)
    network_lcc = network_g.subgraph(lcc).copy()
    df = df[df["user_id"].isin(lcc)].copy()
    df = df.sample(frac=1.0, random_state=SHARED_RANDOM_SEED).reset_index(drop=True)

    with profiles_path.open("wb") as f:
        pickle.dump(df, f)
    with graph_path.open("wb") as f:
        pickle.dump(network_lcc, f)

    print("selected business:", business.get("name"), business_id)
    print("unique raters:", len(reviewer_ids))
    print("lcc user num:", len(df))
    print("lcc graph nodes:", network_lcc.number_of_nodes())
    return df, network_lcc, business


def compute_score(df, rating_column="rating"):
    return df[rating_column].astype(float).tolist()


def extract_features(df, feature_columns=FEATURE_COLUMNS):
    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return X.to_numpy(dtype=float)


def load_or_compute_scores(
    df_labeled,
    df_unlabeled,
    param_folder,
    total_n,
    rating_column="rating",
    cache_key=None,
):
    param_folder = Path(param_folder)
    param_folder.mkdir(exist_ok=True, parents=True)
    cache_suffix = cache_key if cache_key is not None else str(total_n)
    y_label_path = param_folder / f"y_label{cache_suffix}.pk"
    y_unlabel_path = param_folder / f"y_unlabel_label{cache_suffix}.pk"

    if y_label_path.exists() and y_unlabel_path.exists():
        with y_label_path.open("rb") as f:
            y_label = pickle.load(f)
        with y_unlabel_path.open("rb") as f:
            y_unlabel_label = pickle.load(f)
        return y_label, y_unlabel_label

    y_label = compute_score(df_labeled, rating_column=rating_column)
    y_unlabel_label = compute_score(df_unlabeled, rating_column=rating_column)

    with y_label_path.open("wb") as f:
        pickle.dump(y_label, f)
    with y_unlabel_path.open("wb") as f:
        pickle.dump(y_unlabel_label, f)
    return y_label, y_unlabel_label


def load_or_compute_features(df_labeled, df_unlabeled, feature_columns, cache_dir, TARGET_BUSINESS_SLUG, include_graph_features):
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_key = f"{TARGET_BUSINESS_SLUG}_{include_graph_features}"
    labeled_path = cache_dir / f"labeled_feature_matrix_{cache_key}.pk"
    unlabeled_path = cache_dir / f"unlabeled_feature_matrix_{cache_key}.pk"

    if labeled_path.exists() and unlabeled_path.exists():
        with labeled_path.open("rb") as f:
            X_features_labeled = pickle.load(f)
        with unlabeled_path.open("rb") as f:
            X_features_unlabeled = pickle.load(f)
        print("Feature matrix shape:", X_features_labeled.shape)
        return X_features_labeled, X_features_unlabeled


    X_features_labeled = extract_features(df_labeled, feature_columns)
    X_features_unlabeled = extract_features(df_unlabeled, feature_columns)

    with labeled_path.open("wb") as f:
        pickle.dump(X_features_labeled, f)
    with unlabeled_path.open("wb") as f:
        pickle.dump(X_features_unlabeled, f)
    return X_features_labeled, X_features_unlabeled



