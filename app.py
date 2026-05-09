import re
import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from textblob import TextBlob
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
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

st.set_page_config(
    page_title="Movie Performance Predictor",
    page_icon="🎬",
    layout="wide",
)

# ── Helper functions ──────────────────────────────────────────────────────────

def safe_literal_eval(x):
    try:
        if pd.isna(x):
            return []
        return ast.literal_eval(x)
    except Exception:
        return []


def get_first_name(json_text, default="Unknown"):
    try:
        items = safe_literal_eval(json_text)
        if isinstance(items, list) and len(items) > 0:
            return items[0].get("name", default)
    except Exception:
        pass
    return default


def get_count(json_text):
    try:
        items = safe_literal_eval(json_text)
        return len(items) if isinstance(items, list) else 0
    except Exception:
        return 0


def extract_cast_names(json_text, max_cast=5):
    """Extract top-billed actor names from TMDB cast data."""
    try:
        items = safe_literal_eval(json_text)
        if not isinstance(items, list):
            return []

        sorted_items = sorted(items, key=lambda x: x.get("order", 999))

        return [
            item["name"].strip().lower()
            for item in sorted_items[:max_cast]
            if isinstance(item, dict) and "name" in item
        ]

    except Exception:
        return []


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
    "revenge", "survival", "battle", "war", "fight", "escape", "danger",
    "chase", "explosion", "weapon", "mission", "attack", "enemy", "spy",
    "love", "family", "friendship", "betrayal", "loss", "grief", "sacrifice",
    "redemption", "forgiveness", "emotional", "powerful", "heartwarming",
    "secret", "mystery", "murder", "crime", "conspiracy", "haunted",
    "hidden", "truth", "investigation", "detective", "dark",
    "hero", "adventure", "magic", "legend", "quest", "dragon", "kingdom",
    "epic", "destiny", "power", "ancient", "warrior",
    "funny", "hilarious", "scary", "thrilling", "romance", "dream",
    "journey", "discovery", "future", "hope",
]


def keyword_count(text):
    t = clean_text(text)
    return sum(1 for w in AUDIENCE_WORDS if w in t)


def word_count(text):
    return len(clean_text(text).split())


def release_season(month):
    if month in [6, 7, 8]:
        return "summer"
    if month in [11, 12]:
        return "holiday"
    if month in [3, 4, 5]:
        return "spring"
    return "winter"


def compute_star_power(actor_input: str, lookup: dict, global_mean: float):
    """
    Given comma-separated actor names, look them up in historical TMDB data
    and return the average star power score.
    """
    if not actor_input or not actor_input.strip():
        return global_mean, [], []

    names = [n.strip().lower() for n in actor_input.split(",") if n.strip()]
    found, not_found = [], []
    scores = []

    for name in names:
        if name in lookup:
            scores.append(lookup[name])
            found.append(name.title())
        else:
            not_found.append(name.title())

    avg = float(np.mean(scores)) if scores else global_mean
    return avg, found, not_found


def load_credit_parts():
    """Load and validate the 3 split TMDB credits files."""

    required_files = [
        "tmdb_5000_credits_part1.csv",
        "tmdb_5000_credits_part2.csv",
        "tmdb_5000_credits_part3.csv",
    ]

    parts = []

    for file in required_files:
        path = Path(file)

        if not path.exists():
            st.error(
                f"""
                Missing required file: `{file}`

                Make sure this exact file is uploaded to GitHub:
                `{file}`
                """
            )
            st.stop()

        if path.stat().st_size == 0:
            st.error(
                f"""
                `{file}` exists, but it is empty.

                Re-split the original TMDB credits file and upload the file again.
                """
            )
            st.stop()

        try:
            part = pd.read_csv(
                file,
                engine="python",
                on_bad_lines="skip",
                encoding="utf-8",
                encoding_errors="replace",
            )

            if part.empty or len(part.columns) == 0:
                st.error(
                    f"""
                    `{file}` was found, but it has no readable data.

                    Re-split the original credits CSV and upload a fresh copy.
                    """
                )
                st.stop()

            parts.append(part)

        except pd.errors.EmptyDataError:
            st.error(
                f"""
                `{file}` is corrupted or improperly split.

                Re-split the original credits CSV and upload a fresh copy.
                """
            )
            st.stop()

        except Exception as e:
            st.error(f"Problem reading `{file}`: {e}")
            st.stop()

    return pd.concat(parts, ignore_index=True)


# ── Load data + train models ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data and training models… this may take about 30 seconds.")
def load_and_train():

    movies = pd.read_csv(
        "tmdb_5000_movies.csv",
        engine="python",
        on_bad_lines="skip",
        encoding="utf-8",
        encoding_errors="replace",
    )

    credits = load_credit_parts()

    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))

    # Feature engineering
    df["main_genre"] = df["genres"].apply(lambda x: get_first_name(x, "Unknown"))
    df["cast_count"] = df["cast"].apply(get_count)
    df["crew_count"] = df["crew"].apply(get_count)
    df["tmdb_keyword_count"] = df["keywords"].apply(get_count)
    df["production_company_count"] = df["production_companies"].apply(get_count)
    df["cast_names"] = df["cast"].apply(lambda x: extract_cast_names(x, max_cast=5))

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_season"] = df["release_month"].apply(
        lambda m: release_season(int(m)) if pd.notna(m) else "winter"
    )

    for col in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["overview_word_count"] = df["overview"].apply(word_count)
    df["overview_sentiment"] = df["overview"].apply(sentiment_score)
    df["overview_subjectivity"] = df["overview"].apply(subjectivity_score)

    # This replaces the confusing manual tagged keyword input.
    # The app automatically counts audience-appeal language from the plot summary.
    df["audience_keyword_count"] = df["overview"].apply(keyword_count)

    df["budget_log"] = np.log1p(df["budget"].fillna(0))
    df["budget_per_minute_log"] = np.log1p(
        np.where(
            (df["budget"] > 0) & (df["runtime"] > 0),
            df["budget"] / df["runtime"],
            0,
        )
    )

    # Filter usable rows
    mdf = df[
        df["budget"].notna() & (df["budget"] > 0) &
        df["revenue"].notna() & (df["revenue"] > 0) &
        df["runtime"].notna() & (df["runtime"] > 0) &
        df["popularity"].notna() &
        df["vote_average"].notna() &
        df["vote_count"].notna() & (df["vote_count"] >= 25) &
        df["release_year"].notna() &
        df["release_month"].notna() &
        (df["main_genre"] != "Unknown")
    ].copy()

    mdf["roi"] = mdf["revenue"] / mdf["budget"]

    # Performance score: rating + popularity + vote volume + ROI
    mdf["performance_score"] = (
        mdf["vote_average"].rank(pct=True) * 0.30 +
        np.log1p(mdf["popularity"]).rank(pct=True) * 0.25 +
        np.log1p(mdf["vote_count"]).rank(pct=True) * 0.20 +
        np.log1p(mdf["roi"]).rank(pct=True) * 0.25
    )

    mdf = mdf.dropna(subset=["performance_score"]).copy()

    # Build actor star-power lookup
    actor_scores = {}

    for _, row in mdf[["cast_names", "performance_score"]].iterrows():
        score = row["performance_score"]

        for name in row["cast_names"]:
            if name not in actor_scores:
                actor_scores[name] = []
            actor_scores[name].append(score)

    actor_lookup = {
        name: float(np.mean(scores))
        for name, scores in actor_scores.items()
        if len(scores) >= 2
    }

    global_mean = float(mdf["performance_score"].mean())

    def row_star_power(cast_names):
        scores = [actor_lookup[n] for n in cast_names if n in actor_lookup]
        return float(np.mean(scores)) if scores else global_mean

    mdf["cast_star_power"] = mdf["cast_names"].apply(row_star_power)

    # Labels
    low_cut = mdf["performance_score"].quantile(0.33)
    high_cut = mdf["performance_score"].quantile(0.67)

    def label(score):
        if pd.isna(score):
            return np.nan
        if score >= high_cut:
            return "Likely to Perform Well"
        if score <= low_cut:
            return "High Risk of Underperforming"
        return "Average Performance"

    mdf["performance_label"] = mdf["performance_score"].apply(label)
    mdf = mdf.dropna(subset=["performance_label"]).copy()

    features = [
        "budget_log",
        "budget_per_minute_log",
        "runtime",
        "cast_count",
        "crew_count",
        "tmdb_keyword_count",
        "production_company_count",
        "cast_star_power",
        "release_year",
        "release_month",
        "overview_word_count",
        "overview_sentiment",
        "overview_subjectivity",
        "audience_keyword_count",
        "main_genre",
        "release_season",
    ]

    X = pd.get_dummies(
        mdf[features].copy(),
        columns=["main_genre", "release_season"],
        drop_first=False,
    )

    X = X.fillna(0).reset_index(drop=True)
    y = mdf["performance_label"].reset_index(drop=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    lr = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5)),
    ])

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
    )

    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        random_state=42,
    )

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
        voting="soft",
    )

    model_specs = {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "Ensemble (Voting)": ensemble,
    }

    results, trained = {}, {}

    for name, mdl in model_specs.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test,
            preds,
            average="weighted",
            zero_division=0,
        )

        results[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        }

        trained[name] = mdl

    results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)

    best_model = trained["Ensemble (Voting)"]
    best_preds = best_model.predict(X_test)

    fi = rf.feature_importances_

    importance_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": fi})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    genre_list = sorted([
        c.replace("main_genre_", "")
        for c in X.columns
        if c.startswith("main_genre_")
    ])

    return {
        "model": best_model,
        "model_name": "Ensemble (Voting)",
        "feature_cols": list(X.columns),
        "genre_list": genre_list,
        "actor_lookup": actor_lookup,
        "global_mean": global_mean,
        "results_df": results_df,
        "importance_df": importance_df,
        "report": classification_report(y_test, best_preds, zero_division=0),
        "cm": confusion_matrix(y_test, best_preds, labels=sorted(y.unique())),
        "cm_labels": sorted(y.unique()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_actors": len(actor_lookup),
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def make_prediction(
    state,
    genre,
    budget,
    runtime,
    year,
    month,
    cast,
    crew,
    overview,
    num_companies,
    actor_input,
):

    star_power, found, not_found = compute_star_power(
        actor_input,
        state["actor_lookup"],
        state["global_mean"],
    )

    season = release_season(month)
    bpm = (budget / runtime) if runtime > 0 else 0

    auto_plot_keywords = keyword_count(overview)

    row = pd.DataFrame([{
        "budget_log": np.log1p(budget),
        "budget_per_minute_log": np.log1p(bpm),
        "runtime": runtime,
        "cast_count": cast,
        "crew_count": crew,
        "tmdb_keyword_count": auto_plot_keywords,
        "production_company_count": num_companies,
        "cast_star_power": star_power,
        "release_year": year,
        "release_month": month,
        "overview_word_count": word_count(overview),
        "overview_sentiment": sentiment_score(overview),
        "overview_subjectivity": subjectivity_score(overview),
        "audience_keyword_count": auto_plot_keywords,
        "main_genre": genre,
        "release_season": season,
    }])

    row = pd.get_dummies(
        row,
        columns=["main_genre", "release_season"],
        drop_first=False,
    )

    for col in state["feature_cols"]:
        if col not in row.columns:
            row[col] = 0

    row = row[state["feature_cols"]].fillna(0)

    pred = state["model"].predict(row)[0]
    proba = state["model"].predict_proba(row)[0]
    classes = list(state["model"].classes_)

    conf = round(float(np.max(proba)) * 100, 1)
    proba_d = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

    return pred, conf, proba_d, found, not_found, star_power


def build_explanation(
    pred,
    conf,
    budget,
    runtime,
    month,
    overview,
    num_companies,
    found,
    star_power,
    global_mean,
):

    sent = sentiment_score(overview)
    subj = subjectivity_score(overview)
    keys = keyword_count(overview)
    wc = word_count(overview)
    season = release_season(month)

    reasons = []

    if budget >= 100_000_000:
        reasons.append("a large budget supporting stronger production value and marketing reach")
    elif budget <= 10_000_000:
        reasons.append("a smaller budget, which increases risk unless the concept has breakout potential")
    else:
        reasons.append("a moderate budget compared with most commercial films")

    if 85 <= runtime <= 140:
        reasons.append("a commercially typical runtime")
    else:
        reasons.append("an unusual runtime that may affect audience accessibility")

    if season == "summer":
        reasons.append("a summer release window, which is historically strong for box office performance")
    elif season == "holiday":
        reasons.append("a holiday release window, which can benefit family films and major releases")

    if found:
        if star_power > global_mean * 1.1:
            reasons.append(
                f"a high-performing cast ({', '.join(found)}) whose past films averaged above typical performance"
            )
        elif star_power < global_mean * 0.9:
            reasons.append(
                f"a cast ({', '.join(found)}) whose historical films have tended to underperform on average"
            )
        else:
            reasons.append(f"a cast ({', '.join(found)}) with average historical performance")
    else:
        reasons.append("limited actor star-power information from the entered cast")

    if num_companies >= 3:
        reasons.append("multiple production companies backing the film")

    if sent > 0.20:
        reasons.append("a positive, emotionally appealing plot tone")
    elif sent < -0.10:
        reasons.append("a darker or more negative plot tone")
    else:
        reasons.append("a fairly neutral plot tone")

    if keys >= 3:
        reasons.append("several audience-appeal words in the plot summary")
    elif keys == 0:
        reasons.append("few obvious audience-appeal words in the plot summary")
    else:
        reasons.append("some audience-appeal language in the plot summary")

    return (
        f"The ensemble model predicts **{pred}** with **{conf}% confidence**. "
        f"This prediction is mainly driven by {', '.join(reasons)}. "
        f"The AI text analysis found sentiment of **{sent:+.3f}**, "
        f"subjectivity of **{subj:.3f}**, and **{keys}** audience-appeal "
        f"words across **{wc}** words."
    )


# ── App ───────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Performance Predictor")
st.caption("Ensemble model · Actor star power · Trained on TMDB movie data")

state = load_and_train()

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

    year = col_a.number_input(
        "Release Year",
        min_value=2000,
        max_value=2035,
        value=2026,
        step=1,
    )

    month = col_b.slider("Release Month", 1, 12, 7)

    cast = st.slider("Cast Members", 1, 50, 8)
    crew = st.slider("Crew Members", 5, 200, 30)

    num_companies = st.slider("Production Companies", 0, 10, 2)

    actor_input = st.text_input(
        "Main Cast",
        value="",
        placeholder="e.g. Leonardo DiCaprio, Meryl Streep",
        help=(
            f"Type actor names from the TMDB database. "
            f"The model knows {state['n_actors']:,} actors from historical films."
        ),
    )

    overview = st.text_area(
        "Plot Summary",
        value=(
            "A young woman uncovers a dangerous family secret while racing "
            "across the city during one unforgettable night filled with crime, "
            "revenge, and survival."
        ),
        height=130,
    )

    run_btn = st.button("🎬 Predict Performance", width="stretch")


tab_pred, tab_model = st.tabs(["📊 Prediction", "🔬 Model Info"])


with tab_pred:
    if not run_btn:
        st.info("Fill in the movie details on the left and click **Predict Performance**.")

    if run_btn:
        pred, conf, proba_d, found, not_found, star_power = make_prediction(
            state,
            genre,
            budget,
            runtime,
            year,
            month,
            cast,
            crew,
            overview,
            num_companies,
            actor_input,
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

        st.metric("Ensemble Confidence", f"{conf}%")
        st.divider()

        if actor_input.strip():
            st.subheader("🎭 Cast Analysis")

            if found:
                pct = round((star_power / state["global_mean"] - 1) * 100, 1)
                direction = "above" if pct >= 0 else "below"

                st.success(
                    f"**Recognized:** {', '.join(found)}  \n"
                    f"Combined star power score is **{abs(pct)}% {direction}** "
                    f"the average actor in this dataset."
                )

            if not_found:
                st.warning(
                    f"**Not found in database:** {', '.join(not_found)}  \n"
                    f"These actors were not found in the TMDB training data, "
                    f"so the model used the dataset average for them."
                )

            st.divider()

        st.subheader("Class Probabilities")

        for lbl, prob in sorted(proba_d.items(), key=lambda x: -x[1]):
            st.write(f"**{lbl}** — {prob}%")
            st.progress(prob / 100)

        st.divider()

        st.subheader("AI Text Analysis of Plot Summary")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Sentiment", f"{sent:+.3f}", help="-1 = very negative · +1 = very positive")
        c2.metric("Subjectivity", f"{subj:.3f}", help="0 = objective · 1 = very emotional")
        c3.metric("Appeal Words", keys)
        c4.metric("Word Count", wc)

        st.divider()

        st.subheader("Explanation")

        st.markdown(
            build_explanation(
                pred,
                conf,
                budget,
                runtime,
                month,
                overview,
                num_companies,
                found,
                star_power,
                state["global_mean"],
            )
        )

        st.divider()

        st.subheader("Top 10 Predictive Features")
        top10 = state["importance_df"].head(10).set_index("Feature")["Importance"]
        st.bar_chart(top10)


with tab_model:
    st.subheader("Model Comparison")

    st.dataframe(state["results_df"], width="stretch")

    st.caption(
        f"Final predictor: **{state['model_name']}** · "
        f"Training rows: {state['n_train']} · Test rows: {state['n_test']} · "
        f"Actors in database: {state['n_actors']:,}"
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

    st.dataframe(cm_df, width="stretch")

    st.divider()

    st.subheader("Feature Importances")
    st.dataframe(state["importance_df"], width="stretch")
