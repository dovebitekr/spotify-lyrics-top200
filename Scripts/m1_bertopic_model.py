#!/usr/bin/env python3
"""M1 + BERTopic modeling pipeline for Spotify Top 200 lyrics.

This script operationalizes the notebook's next-step plan:
- M1: release_year + one-hot genre_cleaned
- M1 + BERTopic: M1 features plus topic probabilities extracted from lyrics

It uses a 5-fold StratifiedGroupKFold to prevent artist leakage while reporting
ROC-AUC and Macro F1 across folds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from umap import UMAP

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "spotify_data" / "final_dataset_for_ml_lyrics_en.csv"
OUTPUT_PATH = ROOT / "artifacts" / "m1_bertopic_summary.json"
TOPIC_METRICS_PATH = ROOT / "artifacts" / "bertopic_topic_metrics.csv"
TOPIC_ASSIGNMENTS_PATH = ROOT / "artifacts" / "bertopic_song_topics.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 128
CHUNK_STRIDE = 96


def load_standard_corpus(csv_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(csv_path)

    agg_rules = {
        "best_rank_this_year": "min",
        "all_time_peak_rank": "min",
        "all_time_weeks_on_chart": "max",
        "total_streams_this_year": "sum",
        "release_year": "first",
        "genre": "first",
        "language": "first",
        "track_type": "first",
        "lyrics": "first",
        "lemmas": "first",
        "cleaned_lyrics_transformer": "first",
        "word_count": "first",
        "lexical_diversity": "first",
        "lemma_word_count": "first",
        "lemma_lexical_diversity": "first",
    }
    for col in ["raw_word_count", "raw_lexical_diversity", "repetition_rate"]:
        if col in raw_df.columns:
            agg_rules[col] = "first"

    dedup_df = raw_df.groupby(["artist_names", "track_name"], as_index=False).agg(agg_rules)
    standard_df = dedup_df[~dedup_df["track_type"].isin(["holiday", "event_anthem"])].copy()

    standard_df["genre_cleaned"] = (
        standard_df["genre"].fillna("Unknown").replace({"Hip-Hop": "Hip-Hop/Rap", "Christmas: Classic": "Holiday"})
    )
    standard_df["target_hit"] = (standard_df["best_rank_this_year"] <= 50).astype(int)

    cluster_map = _build_artist_clusters(standard_df)
    standard_df["artist_cluster"] = standard_df["artist_names"].apply(
        lambda s: cluster_map.get(str(s).split(",")[0].strip(), "cluster_1")
    )
    return standard_df


def _build_artist_clusters(df: pd.DataFrame) -> dict[str, str]:
    all_artists = set()
    adjacency: defaultdict[str, set[str]] = defaultdict(set)

    for artist_names in df["artist_names"].dropna().tolist():
        parts = [part.strip() for part in str(artist_names).split(",") if part.strip()]
        for artist in parts:
            all_artists.add(artist)
            adjacency[artist] = adjacency.get(artist, set())
        for idx, artist in enumerate(parts):
            for other in parts[idx + 1 :]:
                adjacency[artist].add(other)
                adjacency[other].add(artist)
        if len(parts) == 1:
            adjacency[parts[0]].add(parts[0])

    visited: set[str] = set()
    artist_to_cluster: dict[str, str] = {}
    cluster_counter = 0

    for artist in sorted(all_artists):
        if artist in visited:
            continue
        cluster_counter += 1
        queue = deque([artist])
        visited.add(artist)
        while queue:
            current = queue.popleft()
            artist_to_cluster[current] = f"cluster_{cluster_counter}"
            for neighbor in adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return artist_to_cluster


def prepare_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    meta = df[["release_year", "genre_cleaned"]].copy()
    meta["release_year"] = pd.to_numeric(meta["release_year"], errors="coerce")
    meta["genre_cleaned"] = meta["genre_cleaned"].fillna("Unknown")
    meta["release_year"] = meta["release_year"].fillna(meta["release_year"].median())

    features = pd.get_dummies(meta, columns=["genre_cleaned"], dtype=float)
    return features.fillna(0.0)


def prepare_m1_bertopic_features(df: pd.DataFrame, topic_probabilities: pd.DataFrame) -> pd.DataFrame:
    base_features = prepare_m1_features(df).reset_index(drop=True)
    topic_df = topic_probabilities.copy().reset_index(drop=True)
    if not topic_df.columns.empty:
        topic_df.columns = [f"topic_{idx}" for idx in range(topic_df.shape[1])]
    if len(base_features) != len(topic_df):
        raise ValueError(
            f"Feature length mismatch: M1 has {len(base_features)} rows but BERTopic has {len(topic_df)} rows."
        )

    combined = pd.concat([base_features, topic_df], axis=1)
    return combined.fillna(0.0)


def evaluate_model(df: pd.DataFrame, model_name: str, include_topics: bool = False) -> dict:
    y = df["target_hit"].to_numpy()
    groups = df["artist_cluster"].to_numpy()
    groupsplit = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics: list[dict] = []
    for fold_number, (train_idx, test_idx) in enumerate(groupsplit.split(df, y, groups), start=1):
        train_df = df.iloc[train_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        train_text = train_df["cleaned_lyrics_transformer"].fillna("").astype(str).tolist()
        test_text = test_df["cleaned_lyrics_transformer"].fillna("").astype(str).tolist()

        if include_topics:
            topic_model = BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                language="english",
                calculate_probabilities=True,
                min_topic_size=12,
                n_gram_range=(1, 2),
                verbose=False,
            )
            _, train_topic_probs = topic_model.fit_transform(train_text)
            _, test_topic_probs = topic_model.transform(test_text)
            train_features = prepare_m1_bertopic_features(train_df, pd.DataFrame(train_topic_probs))
            test_features = prepare_m1_bertopic_features(test_df, pd.DataFrame(test_topic_probs))
            feature_cols = sorted(set(train_features.columns) | set(test_features.columns))
            train_features = train_features.reindex(columns=feature_cols, fill_value=0.0)
            test_features = test_features.reindex(columns=feature_cols, fill_value=0.0)
        else:
            train_features = prepare_m1_features(train_df)
            test_features = prepare_m1_features(test_df)
            feature_cols = sorted(set(train_features.columns) | set(test_features.columns))
            train_features = train_features.reindex(columns=feature_cols, fill_value=0.0)
            test_features = test_features.reindex(columns=feature_cols, fill_value=0.0)

        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        model.fit(train_features, train_df["target_hit"])
        probas = model.predict_proba(test_features)[:, 1]
        predictions = model.predict(test_features)

        fold_metrics.append(
            {
                "fold": fold_number,
                "roc_auc": roc_auc_score(test_df["target_hit"], probas),
                "macro_f1": f1_score(test_df["target_hit"], predictions, average="macro"),
            }
        )

    summary = {
        "model_name": model_name,
        "n_rows": len(df),
        "n_positive": int(df["target_hit"].sum()),
        "fold_metrics": fold_metrics,
        "mean_roc_auc": float(pd.Series([m["roc_auc"] for m in fold_metrics]).mean()),
        "mean_macro_f1": float(pd.Series([m["macro_f1"] for m in fold_metrics]).mean()),
    }
    return summary


def embed_long_documents(documents: list[str]) -> np.ndarray:
    """Embed overlapping lyric chunks and mean-pool them to one vector per song."""
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    document_chunks: list[list[str]] = []
    tokenizer = encoder.tokenizer

    for document in documents:
        token_ids = tokenizer(document, add_special_tokens=False, truncation=False)["input_ids"]
        chunks = []
        for start in range(0, len(token_ids), CHUNK_STRIDE):
            chunks.append(
                tokenizer.decode(token_ids[start : start + CHUNK_SIZE], skip_special_tokens=True)
            )
            if start + CHUNK_SIZE >= len(token_ids):
                break
        document_chunks.append(chunks or [""])

    flat_chunks = [chunk for chunks in document_chunks for chunk in chunks]
    chunk_embeddings = encoder.encode(
        flat_chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    document_embeddings = []
    offset = 0
    for chunks in document_chunks:
        pooled = chunk_embeddings[offset : offset + len(chunks)].mean(axis=0)
        pooled /= np.linalg.norm(pooled)
        document_embeddings.append(pooled)
        offset += len(chunks)

    return np.asarray(document_embeddings)


def build_topic_vectorizer(documents: list[str]) -> CountVectorizer:
    """Filter English/lyric filler and terms appearing in at least a quarter of songs."""
    document_vectorizer = CountVectorizer(stop_words="english", binary=True, min_df=2)
    document_term_matrix = document_vectorizer.fit_transform(documents)
    document_frequencies = np.asarray(document_term_matrix.sum(axis=0)).ravel()
    vocabulary = np.asarray(document_vectorizer.get_feature_names_out())
    corpus_common_terms = set(vocabulary[document_frequencies / len(documents) >= 0.25])

    lyric_fillers = {
        "oh", "yeah", "ooh", "uh", "ah", "la", "na", "hey", "whoa", "woo", "hmm",
        "mm", "ha", "baby", "gonna", "wanna", "gotta", "don", "ll", "ve", "re",
        "ain", "aint", "ill", "im", "ive", "youre", "youll", "youve", "thats",
        "hes", "shes", "theyre", "didnt", "doesnt", "isnt", "wasnt", "wouldnt",
        "couldnt", "shouldnt", "dont", "doesnt", "cant", "wont", "lets", "theres",
    }
    english_stop_words = set(CountVectorizer(stop_words="english").get_stop_words())
    return CountVectorizer(
        stop_words=sorted(english_stop_words | lyric_fillers | corpus_common_terms),
        ngram_range=(1, 2),
        min_df=2,
    )


def create_descriptive_topic_report(df: pd.DataFrame) -> pd.DataFrame:
    """Fit topics from lyric embeddings and describe them with cleaned lemmas."""
    documents = df["cleaned_lyrics_transformer"].fillna("").astype(str).tolist()
    representation_documents = df["lemmas"].fillna("").astype(str).tolist()
    embeddings = embed_long_documents(documents)
    topic_model = BERTopic(
        embedding_model=None,
        language="english",
        calculate_probabilities=False,
        umap_model=UMAP(
            n_neighbors=5,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=42,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=8,
            min_samples=1,
            metric="euclidean",
            cluster_selection_method="eom",
        ),
        vectorizer_model=build_topic_vectorizer(representation_documents),
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(documents, embeddings)
    topic_model.update_topics(
        representation_documents,
        topics=topics,
        vectorizer_model=build_topic_vectorizer(representation_documents),
    )

    assignments = df[
        ["artist_names", "track_name", "best_rank_this_year", "target_hit"]
    ].reset_index(drop=True)
    assignments["topic_id"] = topics
    topic_info = topic_model.get_topic_info().set_index("Topic")

    grouped = assignments.groupby("topic_id", as_index=False).agg(
        song_count=("track_name", "size"),
        mean_best_rank=("best_rank_this_year", "mean"),
        median_best_rank=("best_rank_this_year", "median"),
        top_50_count=("target_hit", "sum"),
        top_50_rate=("target_hit", "mean"),
    )
    grouped["topic_label"] = grouped["topic_id"].map(topic_info["Name"])
    grouped["keywords"] = grouped["topic_id"].map(topic_info["Representation"])
    representative_tracks = {}
    topic_assignments = np.asarray(topics)
    for topic_id in grouped["topic_id"]:
        if topic_id == -1:
            continue
        member_indices = np.flatnonzero(topic_assignments == topic_id)
        centroid = embeddings[member_indices].mean(axis=0)
        closest = member_indices[
            np.argsort(embeddings[member_indices] @ centroid)[::-1][:3]
        ]
        representative_tracks[topic_id] = "; ".join(
            f"{assignments.iloc[index]['artist_names']} — {assignments.iloc[index]['track_name']}"
            for index in closest
        )
    grouped["representative_tracks"] = grouped["topic_id"].map(representative_tracks)
    grouped.loc[grouped["topic_id"].eq(-1), ["topic_label", "keywords"]] = [
        "Outliers (unassigned)",
        None,
    ]
    grouped["share_of_songs_pct"] = grouped["song_count"] / len(assignments) * 100
    grouped["is_outlier"] = grouped["topic_id"].eq(-1)
    grouped = grouped[
        [
            "topic_id",
            "topic_label",
            "keywords",
            "representative_tracks",
            "song_count",
            "share_of_songs_pct",
            "mean_best_rank",
            "median_best_rank",
            "top_50_count",
            "top_50_rate",
            "is_outlier",
        ]
    ].sort_values(["song_count", "topic_id"], ascending=[False, True])

    assignments["topic_label"] = assignments["topic_id"].map(topic_info["Name"])
    assignments["topic_keywords"] = assignments["topic_id"].map(topic_info["Representation"])
    assignments.loc[assignments["topic_id"].eq(-1), ["topic_label", "topic_keywords"]] = [
        "Outliers (unassigned)",
        None,
    ]

    TOPIC_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(TOPIC_METRICS_PATH, index=False)
    assignments.to_csv(TOPIC_ASSIGNMENTS_PATH, index=False)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--topics-only",
        action="store_true",
        help="Generate the full-corpus BERTopic prevalence/rank report without running classifier CV.",
    )
    args = parser.parse_args()

    df = load_standard_corpus(DATA_PATH)
    topic_metrics = create_descriptive_topic_report(df)

    if args.topics_only:
        print(f"Descriptive BERTopic report written to {TOPIC_METRICS_PATH}")
        print(topic_metrics.to_string(index=False))
        return

    m1_summary = evaluate_model(df, "M1", include_topics=False)
    m1_bertopic_summary = evaluate_model(df, "M1 + BERTopic", include_topics=True)

    summary = {
        "dataset": {
            "rows": int(len(df)),
            "target_rate": float(df["target_hit"].mean()),
            "artist_clusters": int(df["artist_cluster"].nunique()),
        },
        "M1": m1_summary,
        "M1_plus_BERTopic": m1_bertopic_summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "M1": {
                    "mean_roc_auc": round(m1_summary["mean_roc_auc"], 4),
                    "mean_macro_f1": round(m1_summary["mean_macro_f1"], 4),
                },
                "M1_plus_BERTopic": {
                    "mean_roc_auc": round(m1_bertopic_summary["mean_roc_auc"], 4),
                    "mean_macro_f1": round(m1_bertopic_summary["mean_macro_f1"], 4),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
