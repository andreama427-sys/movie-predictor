import re
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Performance Predictor",
    page_icon="🎬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def safe_literal_eval(x):
    try:
        if pd.isna(x):
            return []
        return ast.literal_eval(x)
    except Exception:
        return []


def extract_names(json_text, max_items=None):
    items = safe_literal_eval(json_text)
    if not isinstance(items, list):
        return []
    names = [item.get("name", "") for item in items if isinstance(item, dict) and item.get("name")]
    return names[:max_items] if max_items else names


def get_first_name(json_text, default="Unknown"):
    names = extract_names(json_text, max_items=1)
    return names[0] if names else default


def get_count(json_text):
    items = safe_literal_eval(json_text)
    return len(items) if isinstance(items, list) else 0


def extract_director(json_text):
    items = safe_literal_eval(json_text)
    if not isinstance(items, list):
        return "Unknown"
    for item in items:
        if isinstance(item, dict) and item.get("job") == "Director":
            return item.get("name", "Unknown")
    return "Unknown"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentiment_score(text):
    t = clean_text(text)
    return TextBlob(t).sentiment.polarity if t else 0.0


def subjectivity_score(text):
    t = clean_text(text)
    return TextBlob(t).sentiment.subjectivity if t else 0.0


AUDIENCE_WORDS = [
    "love", "revenge", "survival", "hero", "battle", "adventure", "secret",
    "family", "friendship", "romance", "crime", "murder", "mystery", "danger",
    "magic", "war", "dream", "escape", "future", "epic", "thrilling",
    "emotional", "powerful", "funny", "scary", "journey", "betrayal",
    "fight", "haunted", "legend", "discovery", "quest", "mission", "alien",
    "superhero", "villain", "comedy", "heartbreaking", "inspiring", "race",
]


def keyword_count(text):
    t = clean_text(text)
    return sum(1 for w in AUDIENCE_WORDS if w in t)


def word_count(text):
    return len(clean_text(text).split())


def split_entered_names(text):
    if not text:
        return []
    names = [n.strip() for n in re.split(r",|\n", text) if n.strip()]
    return names[:10]


def load_credit_parts():
    """Read the split credits file parts and combine them."""
    required = [
        "tmdb_5000_credits_part1.csv",
        "tmdb_5000_credits_part2.csv",
        "tmdb_5000_credits_part3.csv",
    ]
    missing = [f for f in required if not Path(f).exists()]
    if missing:
        st.error("Missing credit file(s): " + ", ".join(missing))
        st.stop()

    parts = []
    for file in required:
        path = Path(file)
        if path.stat().st_size == 0:
            st.error(f"{file} is empty. Re-split the original credits CSV and upload this file again.")
            st.stop()
        try:
            part = pd.read_csv(file, engine="python", on_bad_lines="skip")
        except Exception as e:
            st.error(f"Could not read {file}: {e}")
            st.stop()
        if part.empty or len(part.columns) == 0:
            st.error(f"{file} has no readable data. Re-split and re-upload it.")
            st.stop()
        parts.append(part)

    return pd.concat(parts, ignore_index=True)


def build_actor_stats(training_df):
    """Build historical star-power stats only from the training split."""
    actor_rows = []
    for _, row in training_df.iterrows():
        for actor in row["top_cast_names"]:
            actor_rows.append({
                "actor": actor,
                "performance_score": row["performance_score"],
                "revenue_log": np.log1p(row["revenue"]),
                "vote_average": row["vote_average"],
                "is_high": 1 if row["performance_label"] == "Likely to Perform Well" else 0,
            })

    if not actor_rows:
        return pd.DataFrame(columns=[
            "actor", "actor_success_score", "actor_avg_revenue_log",
            "actor_avg_rating", "actor_high_success_rate", "actor_appearances"
        ])

    actor_df = pd.DataFrame(actor_rows)
    stats = actor_df.groupby("actor").agg(
        actor_success_score=("performance_score", "mean"),
        actor_avg_revenue_log=("revenue_log", "mean"),
        actor_avg_rating=("vote_average", "mean"),
        actor_high_success_rate=("is_high", "mean"),
        actor_appearances=("actor", "count"),
    ).reset_index()
    return stats


def build_director_stats(training_df):
    director_df = training_df[training_df["director"] != "Unknown"].copy()
    if director_df.empty:
        return pd.DataFrame(columns=["director", "director_success_score", "director_avg_revenue_log", "director_appearances"])
    return director_df.groupby("director").agg(
        director_success_score=("performance_score", "mean"),
        director_avg_revenue_log=("revenue", lambda x: np.log1p(x).mean()),
        director_appearances=("director", "count"),
    ).reset_index()


def get_star_power_features(actor_names, actor_stats, global_actor_score=0.50, global_actor_revenue=0.0):
    names = [n for n in actor_names if n]
    if not names or actor_stats.empty:
        return {
            "actor_success_score": global_actor_score,
            "actor_avg_revenue_log": global_actor_revenue,
            "actor_high_success_rate": 0.0,
            "known_actor_count": 0,
            "actor_appearances_total": 0,
            "star_power_score": global_actor_score,
        }

    matches = actor_stats[actor_stats["actor"].isin(names)]
    if matches.empty:
        return {
            "actor_success_score": global_actor_score,
            "actor_avg_revenue_log": global_actor_revenue,
            "actor_high_success_rate": 0.0,
            "known_actor_count": 0,
            "actor_appearances_total": 0,
            "star_power_score": global_actor_score,
        }

    known_count = len(matches)
    appearances = int(matches["actor_appearances"].sum())
    success = float(matches["actor_success_score"].mean())
    revenue = float(matches["actor_avg_revenue_log"].mean())
    high_rate = float(matches["actor_high_success_rate"].mean())

    # Combined star-power score: performance history + high-performance rate + repeat visibility.
    star_power = (
        success * 0.55
        + high_rate * 0.30
        + min(known_count / 5, 1) * 0.10
        + min(appearances / 20, 1) * 0.05
    )

    return {
        "actor_success_score": success,
        "actor_avg_revenue_log": revenue,
        "actor_high_success_rate": high_rate,
        "known_actor_count": known_count,
        "actor_appearances_total": appearances,
        "star_power_score": float(star_power),
    }


def get_director_features(director, director_stats, global_director_score=0.50, global_director_revenue=0.0):
    if not director or director == "Unknown" or director_stats.empty:
        return {
            "director_success_score": global_director_score,
            "director_avg_revenue_log": global_director_revenue,
            "director_appearances": 0,
        }
    match = director_stats[director_stats["director"] == director]
    if match.empty:
        return {
            "director_success_score": global_director_score,
            "director_avg_revenue_log": global_director_revenue,
            "director_appearances": 0,
        }
    row = match.iloc[0]
    return {
        "director_success_score": float(row["director_success_score"]),
        "director_avg_revenue_log": float(row["director_avg_revenue_log"]),
        "director_appearances": int(row["director_appearances"]),
    }


def sharpen_probabilities(proba, temperature=0.72):
    """Makes displayed class confidence clearer without changing the predicted class."""
    p = np.array(proba, dtype=float)
    p = np.clip(p, 1e-9, 1)
    p = p ** (1 / temperature)
    return p / p.sum()

# ─────────────────────────────────────────────────────────────────────────────
# Load data + train models
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data, building actor star-power features, and training models…")
def load_and_train():
    movies = pd.read_csv("tmdb_5000_movies.csv", engine="python", on_bad_lines="skip")
    credits = load_credit_parts()

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))

    # Feature engineering
    df["main_genre"] = df["genres"].apply(lambda x: get_first_name(x, "Unknown"))
    df["top_cast_names"] = df["cast"].apply(lambda x: extract_names(x, max_items=10))
    df["director"] = df["crew"].apply(extract_director)
    df["cast_count"] = df["cast"].apply(get_count)
    df["crew_count"] = df["crew"].apply(get_count)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    for col in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["overview_word_count"] = df["overview"].apply(word_count)
    df["overview_sentiment"] = df["overview"].apply(sentiment_score)
    df["overview_subjectivity"] = df["overview"].apply(subjectivity_score)
    df["audience_keyword_count"] = df["overview"].apply(keyword_count)
    df["budget_log"] = np.log1p(df["budget"].fillna(0))
    df["roi"] = np.where(
        (df["budget"] > 0) & (df["revenue"] > 0),
        df["revenue"] / df["budget"],
        np.nan,
    )

    mdf = df.dropna(subset=[
        "budget", "revenue", "runtime", "popularity", "vote_average",
        "vote_count", "release_year", "release_month"
    ]).copy()

    mdf = mdf[
        (mdf["budget"] > 0)
        & (mdf["revenue"] > 0)
        & (mdf["runtime"] > 0)
        & (mdf["vote_count"] >= 25)
        & (mdf["main_genre"] != "Unknown")
    ].copy()

    # Performance score combines audience rating, popularity, vote volume, and box office ROI.
    mdf["performance_score"] = (
        mdf["vote_average"].rank(pct=True) * 0.25
        + np.log1p(mdf["popularity"]).rank(pct=True) * 0.20
        + np.log1p(mdf["vote_count"]).rank(pct=True) * 0.20
        + np.log1p(mdf["roi"]).rank(pct=True) * 0.20
        + np.log1p(mdf["revenue"]).rank(pct=True) * 0.15
    )

    low_cut = mdf["performance_score"].quantile(0.33)
    high_cut = mdf["performance_score"].quantile(0.67)

    def label(score):
        if score >= high_cut:
            return "Likely to Perform Well"
        if score <= low_cut:
            return "High Risk of Underperforming"
        return "Average Performance"

    mdf["performance_label"] = mdf["performance_score"].apply(label)

    # Split before creating actor/director historical stats to reduce leakage.
    train_idx, test_idx = train_test_split(
        mdf.index,
        test_size=0.25,
        random_state=42,
        stratify=mdf["performance_label"],
    )
    train_df = mdf.loc[train_idx].copy()
    test_df = mdf.loc[test_idx].copy()

    actor_stats = build_actor_stats(train_df)
    director_stats = build_director_stats(train_df)

    global_actor_score = float(train_df["performance_score"].mean())
    global_actor_revenue = float(np.log1p(train_df["revenue"]).mean())
    global_director_score = global_actor_score
    global_director_revenue = global_actor_revenue

    def add_star_features(frame):
        actor_feature_rows = frame["top_cast_names"].apply(
            lambda names: get_star_power_features(
                names, actor_stats, global_actor_score, global_actor_revenue
            )
        ).apply(pd.Series)
        director_feature_rows = frame["director"].apply(
            lambda name: get_director_features(
                name, director_stats, global_director_score, global_director_revenue
            )
        ).apply(pd.Series)
        return pd.concat([frame.reset_index(drop=True), actor_feature_rows, director_feature_rows], axis=1)

    train_df = add_star_features(train_df)
    test_df = add_star_features(test_df)

    features = [
        "budget_log", "runtime", "cast_count", "crew_count",
        "release_year", "release_month",
        "overview_word_count", "overview_sentiment",
        "overview_subjectivity", "audience_keyword_count",
        "actor_success_score", "actor_avg_revenue_log", "actor_high_success_rate",
        "known_actor_count", "actor_appearances_total", "star_power_score",
        "director_success_score", "director_avg_revenue_log", "director_appearances",
        "main_genre",
    ]

    combined_for_columns = pd.concat([train_df[features], test_df[features]], ignore_index=True)
    all_X = pd.get_dummies(combined_for_columns, columns=["main_genre"], drop_first=False).fillna(0)
    X_train = all_X.iloc[:len(train_df)].copy()
    X_test = all_X.iloc[len(train_df):].copy()
    y_train = train_df["performance_label"]
    y_test = test_df["performance_label"]

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=1500, class_weight="balanced", C=1.8)),
    ])

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced_subsample",
    )

    gb = GradientBoostingClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    )

    model_specs = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
    }

    results, trained = {}, {}
    for name, mdl in model_specs.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        }
        trained[name] = mdl

    # Soft-voting ensemble often produces more stable confidence than one model alone.
    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft",
        weights=[1, 2, 2],
    )
    ensemble.fit(X_train, y_train)
    ens_preds = ensemble.predict(X_test)
    ens_acc = accuracy_score(y_test, ens_preds)
    ens_prec, ens_rec, ens_f1, _ = precision_recall_fscore_support(
        y_test, ens_preds, average="weighted", zero_division=0
    )
    results["Ensemble"] = {
        "Accuracy": round(ens_acc, 4),
        "Precision": round(ens_prec, 4),
        "Recall": round(ens_rec, 4),
        "F1": round(ens_f1, 4),
    }
    trained["Ensemble"] = ensemble

    results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
    best_name = results_df.index[0]
    best_model = trained[best_name]
    best_preds = best_model.predict(X_test)

    # Feature importances: use Random Forest importances for interpretability.
    rf_for_importance = trained["Random Forest"]
    importance_df = (
        pd.DataFrame({"Feature": X_train.columns, "Importance": rf_for_importance.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    genre_list = sorted([
        c.replace("main_genre_", "") for c in X_train.columns if c.startswith("main_genre_")
    ])

    # Popular actor list for the UI.
    popular_actors = actor_stats.sort_values(
        ["actor_high_success_rate", "actor_appearances", "actor_avg_revenue_log"],
        ascending=False,
    )["actor"].head(150).tolist()

    popular_directors = director_stats.sort_values(
        ["director_success_score", "director_appearances"],
        ascending=False,
    )["director"].head(100).tolist()

    return {
        "model": best_model,
        "model_name": best_name,
        "feature_cols": list(X_train.columns),
        "genre_list": genre_list,
        "results_df": results_df,
        "importance_df": importance_df,
        "report": classification_report(y_test, best_preds, zero_division=0),
        "cm": confusion_matrix(y_test, best_preds, labels=sorted(y_train.unique())),
        "cm_labels": sorted(y_train.unique()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "actor_stats": actor_stats,
        "director_stats": director_stats,
        "popular_actors": popular_actors,
        "popular_directors": popular_directors,
        "global_actor_score": global_actor_score,
        "global_actor_revenue": global_actor_revenue,
        "global_director_score": global_director_score,
        "global_director_revenue": global_director_revenue,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Prediction + explanation
# ─────────────────────────────────────────────────────────────────────────────

def make_prediction(state, genre, budget, runtime, year, month, cast_count, crew_count, overview, actor_names, director):
    actor_features = get_star_power_features(
        actor_names,
        state["actor_stats"],
        state["global_actor_score"],
        state["global_actor_revenue"],
    )
    director_features = get_director_features(
        director,
        state["director_stats"],
        state["global_director_score"],
        state["global_director_revenue"],
    )

    base = {
        "budget_log": np.log1p(budget),
        "runtime": runtime,
        "cast_count": cast_count,
        "crew_count": crew_count,
        "release_year": year,
        "release_month": month,
        "overview_word_count": word_count(overview),
        "overview_sentiment": sentiment_score(overview),
        "overview_subjectivity": subjectivity_score(overview),
        "audience_keyword_count": keyword_count(overview),
        "main_genre": genre,
    }
    base.update(actor_features)
    base.update(director_features)

    row = pd.DataFrame([base])
    row = pd.get_dummies(row, columns=["main_genre"], drop_first=False)
    for col in state["feature_cols"]:
        if col not in row.columns:
            row[col] = 0
    row = row[state["feature_cols"]]

    pred = state["model"].predict(row)[0]
    raw_proba = state["model"].predict_proba(row)[0]
    classes = list(state["model"].classes_)

    # Sharpen only the displayed confidence/probabilities so the output is easier to interpret.
    display_proba = sharpen_probabilities(raw_proba, temperature=0.72)
    conf = round(float(np.max(display_proba)) * 100, 1)
    proba_d = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, display_proba)}
    raw_proba_d = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, raw_proba)}

    return pred, conf, proba_d, raw_proba_d, actor_features, director_features


def build_explanation(pred, conf, budget, runtime, overview, actor_names, actor_features, director, director_features):
    sent = sentiment_score(overview)
    subj = subjectivity_score(overview)
    keys = keyword_count(overview)
    wc = word_count(overview)

    reasons = []
    risks = []

    if budget >= 100_000_000:
        reasons.append("a large budget that historically supports bigger production value and marketing reach")
    elif budget <= 10_000_000:
        risks.append("a smaller budget, which can limit box office reach unless the concept breaks out")
    else:
        reasons.append("a moderate commercial budget")

    if 85 <= runtime <= 140:
        reasons.append("a commercially typical runtime")
    else:
        risks.append("an unusual runtime that may affect audience accessibility")

    if sent > 0.20:
        reasons.append("a positive plot tone")
    elif sent < -0.10:
        risks.append("a darker plot tone that may narrow audience appeal")
    else:
        reasons.append("a neutral plot tone")

    if keys >= 3:
        reasons.append("several audience-appeal keywords in the description")
    elif keys == 0:
        risks.append("few obvious audience-appeal keywords in the description")
    else:
        reasons.append("some audience-appeal language in the description")

    if actor_features["known_actor_count"] >= 2 and actor_features["star_power_score"] >= 0.60:
        reasons.append("recognizable actors with stronger historical movie performance in the dataset")
    elif actor_features["known_actor_count"] == 0:
        risks.append("the entered actor names were not found in the TMDB training data, so star power could not be fully measured")
    else:
        reasons.append("some actor history was found in the dataset")

    if director and director != "Unknown" and director_features["director_appearances"] > 0:
        reasons.append("director history from prior movies in the dataset")

    if not reasons:
        reasons.append("the combined historical movie features")
    if not risks:
        risks.append("the model does not include real-time marketing buzz, reviews, streaming deals, or social media trends")

    recommendation = ""
    if pred == "Likely to Perform Well":
        recommendation = "Proceed with a strong release and marketing push."
    elif pred == "Average Performance":
        recommendation = "Use a targeted release strategy and monitor early audience response."
    else:
        recommendation = "Proceed carefully and consider reducing budget, narrowing the release, or strengthening market positioning."

    return {
        "recommendation": recommendation,
        "summary": (
            f"The model predicts **{pred}** with **{conf}% confidence**. "
            f"The strongest signals are {', '.join(reasons)}."
        ),
        "risks": risks,
        "nlp": (
            f"The AI text analysis found a sentiment score of **{sent:+.3f}**, "
            f"subjectivity of **{subj:.3f}**, and **{keys}** audience-appeal "
            f"keywords across **{wc}** words."
        )
    }

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Performance Predictor")
st.caption("Predicts movie performance using TMDB data, plot-text AI signals, and actor star-power history.")

state = load_and_train()

with st.expander("About this tool", expanded=False):
    st.write(
        "This tool trains machine learning models on historical TMDB movie data. "
        "It uses budget, runtime, release timing, genre, cast/crew size, plot summary text, "
        "and actor/director historical performance to classify a movie as likely to perform well, "
        "average performance, or high risk of underperforming. The AI component analyzes the plot "
        "summary for sentiment, subjectivity, word count, and audience-appeal keywords."
    )

# Sidebar
with st.sidebar:
    st.header("Enter Movie Details")

    genre = st.selectbox("Genre", state["genre_list"])

    budget = st.number_input(
        "Production Budget ($)",
        min_value=100_000,
        max_value=500_000_000,
        value=25_000_000,
        step=1_000_000,
        format="%d",
    )

    runtime = st.slider("Runtime (minutes)", 60, 240, 110)

    col_a, col_b = st.columns(2)
    year = col_a.number_input("Release Year", min_value=1980, max_value=2035, value=2026, step=1)
    month = col_b.slider("Release Month", 1, 12, 7)

    cast_count = st.slider("Total Cast Members", 1, 80, 12)
    crew_count = st.slider("Crew Members", 5, 250, 45)

    st.markdown("### Actor Star Power")
    selected_actors = st.multiselect(
        "Select known actors from the dataset",
        options=state["popular_actors"],
        default=state["popular_actors"][:2] if len(state["popular_actors"]) >= 2 else [],
        help="These names are matched against historical TMDB cast data and converted into star-power features.",
    )

    extra_actor_text = st.text_area(
        "Or type actor names, separated by commas",
        value="",
        height=70,
        help="Example: Tom Cruise, Scarlett Johansson, Denzel Washington",
    )

    actor_names = list(dict.fromkeys(selected_actors + split_entered_names(extra_actor_text)))

    director = st.selectbox(
        "Director / Filmmaker History",
        options=["Unknown"] + state["popular_directors"],
        index=0,
    )

    overview = st.text_area(
        "Plot Summary",
        value=(
            "A young woman uncovers a dangerous family secret while racing "
            "across the city during one unforgettable night filled with crime, "
            "revenge, and survival."
        ),
        height=150,
    )

    run_btn = st.button("🎬 Predict Performance", use_container_width=True)

# Tabs
tab_pred, tab_model, tab_data = st.tabs(["📊 Prediction", "🔬 Model Info", "🎭 Actor Data"])

with tab_pred:
    if not run_btn:
        st.info("Fill in the movie details on the left and click **Predict Performance**.")

    if run_btn:
        pred, conf, proba_d, raw_proba_d, actor_features, director_features = make_prediction(
            state, genre, budget, runtime, year, month, cast_count, crew_count, overview, actor_names, director
        )
        sent = sentiment_score(overview)
        subj = subjectivity_score(overview)
        keys = keyword_count(overview)
        wc = word_count(overview)

        if "Well" in pred:
            st.success(f"### ✅ {pred}")
        elif "Average" in pred:
            st.warning(f"### ⚠️ {pred}")
        else:
            st.error(f"### 🔴 {pred}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Displayed Confidence", f"{conf}%")
        c2.metric("Known Actors Matched", int(actor_features["known_actor_count"]))
        c3.metric("Star Power Score", f"{actor_features['star_power_score']:.2f}")

        explanation = build_explanation(
            pred, conf, budget, runtime, overview, actor_names, actor_features, director, director_features
        )

        st.divider()
        st.subheader("Final Recommendation")
        st.markdown(f"### {explanation['recommendation']}")
        st.markdown(explanation["summary"])

        st.divider()
        st.subheader("Class Probabilities")
        st.caption("Displayed probabilities are sharpened for clearer interpretation; the model prediction itself is unchanged.")
        for label, prob in sorted(proba_d.items(), key=lambda x: -x[1]):
            st.write(f"**{label}** — {prob}%")
            st.progress(prob / 100)

        with st.expander("Raw model probabilities"):
            for label, prob in sorted(raw_proba_d.items(), key=lambda x: -x[1]):
                st.write(f"{label}: {prob}%")

        st.divider()
        st.subheader("AI Text Analysis of Plot Summary")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("Sentiment", f"{sent:+.3f}", help="-1 = very negative · +1 = very positive")
        n2.metric("Subjectivity", f"{subj:.3f}", help="0 = factual · 1 = emotional/opinion-based")
        n3.metric("Appeal Keywords", keys)
        n4.metric("Word Count", wc)
        st.markdown(explanation["nlp"])

        st.divider()
        st.subheader("Actor + Director Signals")
        st.write("**Actors entered:** " + (", ".join(actor_names) if actor_names else "None"))
        st.write(f"**Known actor count:** {int(actor_features['known_actor_count'])}")
        st.write(f"**Actor historical success score:** {actor_features['actor_success_score']:.3f}")
        st.write(f"**Actor high-performance rate:** {actor_features['actor_high_success_rate']:.3f}")
        st.write(f"**Actor average box office log revenue:** {actor_features['actor_avg_revenue_log']:.3f}")
        st.write(f"**Director selected:** {director}")
        st.write(f"**Director historical success score:** {director_features['director_success_score']:.3f}")

        st.divider()
        st.subheader("Risk Factors")
        for risk in explanation["risks"]:
            st.markdown(f"- {risk}")

        st.divider()
        st.subheader("Top 12 Predictive Features")
        top12 = state["importance_df"].head(12).set_index("Feature")["Importance"]
        st.bar_chart(top12)

with tab_model:
    st.subheader("Model Comparison")
    st.dataframe(state["results_df"], use_container_width=True)
    st.caption(
        f"Best model selected by weighted F1: **{state['model_name']}** · "
        f"Training rows: {state['n_train']} · Test rows: {state['n_test']}"
    )

    st.divider()
    st.subheader("Classification Report")
    st.code(state["report"], language=None)

    st.divider()
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        state["cm"],
        index=[f"Actual: {l}" for l in state["cm_labels"]],
        columns=[f"Predicted: {l}" for l in state["cm_labels"]],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("All Feature Importances")
    st.dataframe(state["importance_df"], use_container_width=True)

with tab_data:
    st.subheader("Popular Actor History Used by the Model")
    st.write(
        "The app now uses actor names from the TMDB credits data. Actors are converted into star-power features based on "
        "their historical average performance score, average box office revenue, high-performance rate, and number of appearances."
    )
    display_actor_stats = state["actor_stats"].sort_values(
        ["actor_high_success_rate", "actor_appearances", "actor_avg_revenue_log"],
        ascending=False,
    ).head(50)
    st.dataframe(display_actor_stats, use_container_width=True)

    st.divider()
    st.subheader("Limitations")
    st.markdown(
        "- Actor popularity is based only on names available in the TMDB credits dataset.\n"
        "- The model does not know current social media buzz, advertising spend, scandals, streaming deals, or critic reviews.\n"
        "- Box office performance is approximated using historical revenue, ROI, vote count, popularity, and rating.\n"
        "- The confidence score is a model-based estimate, not a guarantee of real-world performance."
    )
