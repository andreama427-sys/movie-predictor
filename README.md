# movie-predictor
# 🎬 Movie Performance Predictor

## Project Overview

This project is an AI/ML-powered movie performance prediction tool built using Streamlit and trained on historical TMDB movie data.

The application predicts whether a movie is:

- Likely to Perform Well
- Average Performance
- High Risk of Underperforming

The prediction is based on movie features such as:

- Budget
- Runtime
- Genre
- Release timing
- Cast size
- Crew size
- Production companies
- Actor star power
- Plot summary analysis

The AI component analyzes the movie plot summary using natural language processing (NLP) techniques including sentiment analysis, subjectivity analysis, and audience-appeal keyword detection.

The final output includes:

- Performance prediction
- Confidence score
- Probability breakdown
- AI text analysis
- Actor star-power analysis
- Feature importance explanations

---

# Dataset

This project uses the TMDB 5000 Movies Dataset and TMDB Credits Dataset from Kaggle.

Files used:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits_part1.csv`
- `tmdb_5000_credits_part2.csv`
- `tmdb_5000_credits_part3.csv`

The credits dataset was split into 3 files to meet GitHub upload size limits.

---

# Technologies Used

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- TextBlob

---

# Setup Instructions

## Step 1 — Clone the Repository

Open Terminal or Command Prompt and run:

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
cd movie-predictor
```

Replace `YOUR_GITHUB_REPOSITORY_LINK` with your actual GitHub repository URL.

---

## Step 2 — Install Required Packages

Run:

```bash
pip install -r requirements.txt
```

Then download TextBlob language data:

```bash
python -m textblob.download_corpora
```

---

## Step 3 — Verify Files

Make sure these files are inside the project folder:

```text
app.py
requirements.txt
tmdb_5000_movies.csv
tmdb_5000_credits_part1.csv
tmdb_5000_credits_part2.csv
tmdb_5000_credits_part3.csv
README.md
```

---

## Step 4 — Run the Streamlit App

Run:

```bash
streamlit run app.py
```

---

## Step 5 — Open the App

After running the command above, Streamlit will generate a local URL similar to:

```text
http://localhost:8501
```

Open the link in your browser.

---

# How to Use the App

1. Enter movie details in the sidebar:
   - Genre
   - Budget
   - Runtime
   - Release timing
   - Cast size
   - Crew size
   - Production companies
   - Actor names
   - Plot summary

2. Click:

```text
🎬 Predict Performance
```

3. The app will generate:
   - Movie performance prediction
   - Confidence score
   - AI text analysis
   - Actor star-power analysis
   - Prediction explanation
   - Feature importance analysis

---

# Machine Learning Model

The project uses an ensemble classification model combining:

- Logistic Regression
- Random Forest
- Gradient Boosting

The model predicts one of three categories:

- Likely to Perform Well
- Average Performance
- High Risk of Underperforming

---

# AI Component

The AI/NLP portion analyzes the movie overview and extracts:

- Sentiment score
- Subjectivity score
- Audience-appeal keywords
- Emotional tone
- Word count

This information helps explain WHY the prediction was made.

---

# Actor Star Power Analysis

The application analyzes actor names entered by the user and compares them against historical TMDB movie performance data to estimate overall cast star power.

---

# Evaluation Metrics

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

# Limitations

This model does not account for:

- Real-time marketing campaigns
- Social media virality
- Award buzz
- Franchise loyalty
- Actor controversies
- Streaming platform deals

Predictions are based only on historical TMDB movie data and AI-driven text analysis.

---

# Author

Andrea Morales  
Villanova University  
Finance Major | AI & Machine Learning Minor
