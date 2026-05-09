import re
import ast
import warnings

import numpy as np
import pandas as pd
import streamlit as st

from textblob import TextBlob
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# ── Page config ───────────────────────────────────────────────────────────────
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
    "fight", "haunted", "legend", "discovery", "quest",
]

def keyword_count(text):
    t = clean_text(text)
    return sum(1 for w in AUDIENCE_WORDS if w in t)

def word_count(text):
    return len(clean_text(text).split())


# ── Load data + train models ──────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading data and training models… (~30 seconds)")
def load_and_train():

    # ── Load movies ──────────────────────────────────────────────────────────
    movies = pd.read_csv("tmdb_5000_movies.csv", on_bad_lines="skip")

    # ── Load credits from 3 split files and combine ──────────────────────────
    part1 = pd.read_csv("tmdb_5000_credits_part1.csv", on_bad_lines="skip")
    part2 = pd.read_csv("tmdb_5000_credits_part2.csv", on_bad_lines="skip")
    part3 = pd.read_csv("tmdb_5000_credits_part3.csv", on_bad_lines="skip")
    credits = pd.concat([part1, part2, part3], ignore_index=True)

    # ── Merge ────────────────────────────────────────────────────────────────
    credits = credits.rename(columns={"movie_id": "id"})
    df = movies.merge(credits, on="id", how="left", suffixes=("", "_credits"))

    # ── Feature engineering ──────────────────────────────────────────────────
    df["main_genre"]    = df["genres"].apply(lambda x: get_first_name(x, "Unknown"))
    df["cast_count"]    = df["cast"].apply(get_count)
    df["crew_count"]    = df["crew"].apply(get_count)
    df["release_date"]  = pd.to_datetime(df["release_date"], errors="coerce")
    df["release_year"]  = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month

    for col in ["budget", "revenue", "runtime", "popularity", "vote_average", "vote_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["overview_word_count"]    = df["overview"].apply(word_count)
    df["overview_sentiment"]     = df["overview"].apply(sentiment_score)
    df["overview_subjectivity"]  = df["overview"].apply(subjectivity_score)
    df["audience_keyword_count"] = df["overview"].apply(keyword_count)
    df["budget_log"]             = np.log1p(df["budget"].fillna(0))

    # ── Hard filter BEFORE computing roi so no NaN leaks in ─────────────────
    mdf = df[
        (df["budget"].notna())  & (df["budget"]  > 0) &
        (df["revenue"].notna()) & (df["revenue"] > 0) &
        (df["runtime"].notna()) & (df["runtime"] > 0) &
        (df["popularity"].notna()) &
        (df["vote_average"].notna()) &
        (df["vote_count"].notna())  & (df["vote_count"] >= 25) &
        (df["release_year"].notna()) &
        (df["release_month"].notna()) &
        (df["main_genre"] != "Unknown")
    ].copy()

    # roi is safe to compute now — no zero denominators
    mdf["roi"] = mdf["revenue"] / mdf["budget"]

    # ── Performance score ─────────────────────────────────────────────────────
    mdf["performance_score"] = (
        mdf["vote_average"].rank(pct=True)         * 0.30 +
        np.log1p(mdf["popularity"]).rank(pct=True) * 0.25 +
        np.log1p(mdf["vote_count"]).rank(pct=True) * 0.20 +
        np.log1p(mdf["roi"]).rank(pct=True)        * 0.25
    )

    # Drop any rows where performance_score itself is NaN
    mdf = mdf.dropna(subset=["performance_score"]).copy()

    low_cut  = mdf["performance_score"].quantile(0.33)
    high_cut = mdf["performance_score"].quantile(0.67)

    def label(score):
        if pd.isna(score):           return np.nan
        if score >= high_cut:        return "Likely to Perform Well"
        if score <= low_cut:         return "High Risk of Underperforming"
        return "Average Performance"

    mdf["performance_label"] = mdf["performance_score"].apply(label)

    # Final safety drop — remove any remaining NaN labels
    mdf = mdf.dropna(subset=["performance_label"]).copy()

    # ── Feature matrix ────────────────────────────────────────────────────────
    features = [
        "budget_log", "runtime", "cast_count", "crew_count",
        "release_year", "release_month",
        "overview_word_count", "overview_sentiment",
        "overview_subjectivity", "audience_keyword_count",
        "main_genre",
    ]

    X = pd.get_dummies(mdf[features].copy(), columns=["main_genre"], drop_first=False)
    X = X.fillna(0)
    y = mdf["performance_label"].reset_index(drop=True)
    X = X.reset_index(drop=True)

    # Align just in case
    assert len(X) == len(y), "X and y length mismatch after cleaning"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # ── Train models ──────────────────────────────────────────────────────────
    model_specs = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("model",  LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results, trained = {}, {}
    for name, mdl in model_specs.items():
        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        acc  = accuracy_score(y_test, preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="weighted", zero_division=0
        )
        results[name] = {
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
        }
        trained[name] = mdl

    results_df = pd.DataFrame(results).T.sort_values("F1", ascending=False)
    best_name  = results_df.index[0]
    best_model = trained[best_name]
    best_preds = best_model.predict(X_test)

    # ── Feature importances ───────────────────────────────────────────────────
    if hasattr(best_model, "feature_importances_"):
        fi = best_model.feature_importances_
    else:
        coef = best_model.named_steps["model"].coef_
        fi   = np.mean(np.abs(coef), axis=0)

    importance_df = (
        pd.DataFrame({"Feature": X.columns, "Importance": fi})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )

    genre_list = sorted([
        c.replace("main_genre_", "") for c in X.columns if c.startswith("main_genre_")
    ])

    return {
        "model":        best_model,
        "model_name":   best_name,
        "feature_cols": list(X.columns),
        "genre_list":   genre_list,
        "results_df":   results_df,
        "importance_df":importance_df,
        "report":       classification_report(y_test, best_preds, zero_division=0),
        "cm":           confusion_matrix(y_test, best_preds, labels=sorted(y.unique())),
        "cm_labels":    sorted(y.unique()),
        "n_train":      len(X_train),
        "n_test":       len(X_test),
    }


# ── Prediction ────────────────────────────────────────────────────────────────

def make_prediction(state, genre, budget, runtime, year, month, cast, crew, overview):
    row = pd.DataFrame([{
        "budget_log":             np.log1p(budget),
        "runtime":                runtime,
        "cast_count":             cast,
        "crew_count":             crew,
        "release_year":           year,
        "release_month":          month,
        "overview_word_count":    word_count(overview),
        "overview_sentiment":     sentiment_score(overview),
        "overview_subjectivity":  subjectivity_score(overview),
        "audience_keyword_count": keyword_count(overview),
        "main_genre":             genre,
    }])
    row = pd.get_dummies(row, columns=["main_genre"], drop_first=False)
    for col in state["feature_cols"]:
        if col not in row.columns:
            row[col] = 0
    row = row[state["feature_cols"]].fillna(0)

    pred    = state["model"].predict(row)[0]
    proba   = state["model"].predict_proba(row)[0]
    classes = list(state["model"].classes_)
    conf    = round(float(np.max(proba)) * 100, 1)
    proba_d = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}
    return pred, conf, proba_d


def build_explanation(pred, conf, budget, runtime, overview):
    sent = sentiment_score(overview)
    subj = subjectivity_score(overview)
    keys = keyword_count(overview)
    wc   = word_count(overview)

    reasons = []
    if budget >= 100_000_000:
        reasons.append("a large budget supporting strong production value and marketing")
    elif budget <= 10_000_000:
        reasons.append("a smaller budget, which increases risk unless the concept has breakout potential")
    else:
        reasons.append("a moderate budget compared with most commercial films")

    if 85 <= runtime <= 140:
        reasons.append("a commercially typical runtime")
    else:
        reasons.append("an unusual runtime, which may affect audience accessibility")

    if sent > 0.20:
        reasons.append("a positive, emotionally appealing plot tone")
    elif sent < -0.10:
        reasons.append("a darker or more negative plot tone")
    else:
        reasons.append("a neutral plot tone")

    if keys >= 3:
        reasons.append("multiple audience-appeal keywords in the description")
    elif keys == 0:
        reasons.append("few obvious audience-appeal keywords in the description")
    else:
        reasons.append("some audience-appeal keywords in the description")

    return (
        f"The model predicts **{pred}** with **{conf}% confidence**. "
        f"This is mainly driven by {', '.join(reasons)}. "
        f"The NLP analysis found a sentiment score of **{sent:+.3f}**, "
        f"subjectivity of **{subj:.3f}**, and **{keys}** audience-appeal "
        f"keywords across **{wc}** words."
    )


# ── App ───────────────────────────────────────────────────────────────────────

st.title("🎬 Movie Performance Predictor")
st.caption("Trained on 4,800+ TMDB films · Logistic Regression · Random Forest · Gradient Boosting")

state = load_and_train()

# ── Sidebar ───────────────────────────────────────────────────────────────────
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
    year  = col_a.number_input("Release Year",  min_value=2000, max_value=2035, value=2026, step=1)
    month = col_b.slider("Release Month", 1, 12, 7)

    cast = st.slider("Cast Members",  1,  50,  8)
    crew = st.slider("Crew Members",  5, 200, 30)

    overview = st.text_area(
        "Plot Summary",
        value=(
            "A young woman uncovers a dangerous family secret while racing "
            "across the city during one unforgettable night filled with crime, "
            "revenge, and survival."
        ),
        height=140,
    )

    run_btn = st.button("🎬 Predict Performance", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_pred, tab_model = st.tabs(["📊 Prediction", "🔬 Model Info"])

with tab_pred:
    if not run_btn:
        st.info("Fill in the movie details on the left and click **Predict Performance**.")

    if run_btn:
        pred, conf, proba_d = make_prediction(
            state, genre, budget, runtime, year, month, cast, crew, overview
        )
        sent = sentiment_score(overview)
        subj = subjectivity_score(overview)
        keys = keyword_count(overview)
        wc   = word_count(overview)

        # Result banner
        if "Well" in pred:
            st.success(f"### ✅ {pred}")
        elif "Average" in pred:
            st.warning(f"### ⚠️ {pred}")
        else:
            st.error(f"### 🔴 {pred}")

        st.metric("Model Confidence", f"{conf}%")
        st.divider()

        st.subheader("Class Probabilities")
        for lbl, prob in sorted(proba_d.items(), key=lambda x: -x[1]):
            st.write(f"**{lbl}** — {prob}%")
            st.progress(prob / 100)

        st.divider()

        st.subheader("NLP Analysis of Plot Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sentiment",       f"{sent:+.3f}", help="-1 = very negative · +1 = very positive")
        c2.metric("Subjectivity",    f"{subj:.3f}",  help="0 = objective · 1 = very emotional")
        c3.metric("Appeal Keywords", keys)
        c4.metric("Word Count",      wc)

        st.divider()

        st.subheader("Explanation")
        st.markdown(build_explanation(pred, conf, budget, runtime, overview))

        st.divider()

        st.subheader("Top 10 Predictive Features")
        top10 = state["importance_df"].head(10).set_index("Feature")["Importance"]
        st.bar_chart(top10)

with tab_model:
    st.subheader("Model Comparison")
    st.dataframe(state["results_df"], use_container_width=True)
    st.caption(
        f"Best model: **{state['model_name']}** · "
        f"Training rows: {state['n_train']} · Test rows: {state['n_test']}"
    )

    st.divider()

    st.subheader("Classification Report")
    st.code(state["report"], language=None)

    st.divider()

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(
        state["cm"],
        index   =[f"Actual: {l}"    for l in state["cm_labels"]],
        columns =[f"Predicted: {l}" for l in state["cm_labels"]],
    )
    st.dataframe(cm_df, use_container_width=True)

    st.divider()

    st.subheader("All Feature Importances")
    st.dataframe(state["importance_df"], use_container_width=True)
