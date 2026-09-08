# Toxic-comment_moderation_pipeline

A content moderation pipeline that classifies toxic comments using XGBoost, with an LLM-assisted routing layer for cases where the classifier is uncertain. Deployed as a FastAPI service on AWS EC2 with automated CI/CD.

**Live API:** `http://<ec2-3.26.72.116>:8000/docs`

## Problem

Online news platforms rely on human moderators to review reader comments, a process that becomes difficult to scale with growing volume.

A pure ML classifier is fast and inexpensive but struggles with semantically ambiguous comments. An LLM can handle ambiguity better but is too slow and expensive to run on every comment.

This project combines both: XGBoost handles confidently classified cases, while only uncertain cases are routed to Gemini.

## Dataset

[Civil Comments (Jigsaw Unintended Bias in Toxicity Classification)](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) - 90,902 real comments from news sites, annotated by human raters.

The dataset includes comment text, crowd-sourced toxicity scores, moderator decisions, identity attributes, and toxicity sub-scores such as insult, threat, obscene, identity attack, and severe toxicity.

## Key EDA Finding: Publication Policy Affects Moderation Decisions

The dataset contains two possible targets: the crowd toxicity score and the moderator's approve/reject decision.

The moderator decision was investigated first.

| Toxicity band | Comments | Rejected | Reject rate |
| ------------- | -------: | -------: | ----------: |
| 0.0–0.2       |   42,134 |    1,837 |        4.4% |
| 0.2–0.5       |    3,317 |      318 |        9.6% |
| 0.5–0.8       |   14,620 |    4,614 |       31.6% |
| 0.8–1.0       |   30,831 |    9,216 |       29.9% |

Rejection increases sharply after the 0.5 toxicity threshold but then flattens. Comments scoring 0.8+ are rejected at almost the same rate as those scoring 0.5–0.8, meaning two-thirds of highly toxic comments remain approved.

Breaking this down by publication explains why:

| Publication | Comments | Avg. toxicity | Reject rate |
| ----------- | -------: | ------------: | ----------: |
| 43          |      608 |         0.496 |       34.2% |
| 54          |   30,130 |         0.406 |       22.7% |
| 102         |   11,808 |         0.467 |       18.3% |
| 55          |    7,252 |         0.466 |       10.2% |
| 53          |    4,525 |         0.232 |        6.5% |

Publications 55 and 102 have almost identical average toxicity scores (0.466 vs 0.467) but substantially different rejection rates.

The conclusion was that toxicity determines whether a comment is a candidate for removal, but publication policy influences the final moderation outcome. Since publication policy is not available in the text itself, predicting approval/rejection using only text would have an information ceiling.

`is_toxic` was therefore used as the prediction target.

The EDA also identified identity bias: comments mentioning certain identity groups received disproportionately high toxicity scores relative to the dataset baseline.

## Method

### Data preparation

Comments were lowercased, cleaned, tokenized, stripped of English stopwords, and stemmed using Porter stemming. Rows emptied during preprocessing were removed, leaving 90,699 comments.

The original dataset was exactly 50/50 balanced. To better reflect a real moderation setting, toxic comments were deliberately downsampled to 15%.

Final dataset: 53,242 comments, split into 42,593 training and 10,649 test observations.

### Features and Models

TF-IDF features with unigrams and bigrams were used, with a 20,000-feature vocabulary and `min_df=3`.

Four models were benchmarked:

| Model               |  F1 @ 0.5 |   Best F1 | Threshold | Precision |    Recall |
| ------------------- | --------: | --------: | --------: | --------: | --------: |
| Logistic Regression |     0.833 |     0.841 |     0.576 |     0.869 |     0.815 |
| Decision Tree       |     0.776 |     0.779 |     0.618 |     0.833 |     0.731 |
| Random Forest       |     0.823 |     0.825 |     0.499 |     0.846 |     0.804 |
| **XGBoost**         | **0.850** | **0.854** | **0.630** | **0.885** | **0.825** |

XGBoost achieved the best performance and was selected as the production classifier.

## Hybrid ML + LLM Routing

Instead of sending every comment to an LLM, XGBoost probability scores were used to identify uncertain cases.

| Probability band |   Cases | % of test set | XGBoost accuracy |
| ---------------- | ------: | ------------: | ---------------: |
| 0.30–0.80        |     777 |          7.3% |            0.725 |
| **0.40–0.80**    | **510** |      **4.8%** |        **0.651** |
| 0.35–0.75        |     557 |          5.2% |            0.679 |
| Overall          |       — |             — |            0.958 |

XGBoost was 95.8% accurate overall but only 65.1% accurate within the 0.4–0.8 probability range. This band was selected because it isolated the hardest cases while routing fewer than 5% of comments to the LLM.

### Routing logic

```text
comment → preprocess → TF-IDF → XGBoost.predict_proba()
                                        │
                      ┌─────────────────┴─────────────────┐
              prob < 0.4 or prob > 0.8             0.4 ≤ prob ≤ 0.8
                       │                                   │
               XGBoost prediction                    Gemini (semantic)
                                                           │
                                                  API error → XGBoost
```

### Results

| System                                |        F1 |
| ------------------------------------- | --------: |
| XGBoost alone                         |     0.854 |
| **Hybrid (XGBoost + Gemini routing)** | **0.878** |

The hybrid system improved F1 by 2.4 percentage points while sending under 5% of traffic to the LLM.

Of the 510 routed cases, 505 were classified by Gemini. Five hit the daily API quota and fell back to the XGBoost prediction.

Gemini uses structured JSON output with schema enforcement, `temperature=0`, `top_k=1`, and exponential backoff for rate-limit errors.

API failures fall back to the classifier prediction rather than defaulting to "not toxic."

## Deployment

The FastAPI service is containerized with Docker and deployed on AWS EC2. GitHub Actions automates builds and deployment on pushes to `main`.

The API key is injected as a runtime environment variable and is not included in the Docker image. NLTK corpora are downloaded during the build process so preprocessing does not require network access at runtime.

### Endpoints

| Method | Path       | Description        |
| ------ | ---------- | ------------------ |
| `GET`  | `/`        | Service status     |
| `GET`  | `/health`  | Health check       |
| `POST` | `/predict` | Classify a comment |

**Request:**

```json
{
  "comment": "That is so pretty dumb lol"
}
```

**Response:**

```json
{
  "comment": "That is so pretty dumb lol",
  "prediction": 1,
  "confidence": 0.9486,
  "routed_to": "xgboost",
  "label": "toxic"
}
```

The `routed_to` field records whether a prediction came from `xgboost`, `gemini`, or `xgboost_fallback`.

## Running Locally

```bash
git clone https://github.com/Hasika4/Toxic-comment_moderation_pipeline.git
cd Toxic-comment_moderation_pipeline

export GEMINI_API_KEY="your_key"
# Windows: setx GEMINI_API_KEY "your_key"

docker build -t toxic-moderation .
docker run -p 8000:8000 -e GEMINI_API_KEY=$GEMINI_API_KEY toxic-moderation
```

Then open `http://localhost:8000/docs`.

To reproduce the analysis, download the dataset from Kaggle and run:

```text
notebooks/Toxic_detection.ipynb
```

## Repository Structure

```text
.
├── notebooks/
│   └── Toxic_detection.ipynb      # EDA, modelling, routing analysis
├── app/
│   └── main.py                    # FastAPI service
├── .github/workflows/
│   └── deploy.yml                 # CI/CD pipeline
├── model.pkl                      # Trained XGBoost + TF-IDF vectorizer
├── Dockerfile
└── requirements.txt
```

## Tech Stack

**ML:** Python, scikit-learn, XGBoost, NLTK, pandas, NumPy
**LLM:** Gemini API
**Serving:** FastAPI, Docker
**Infrastructure:** AWS EC2, GitHub Actions
**Analysis:** pandasql, matplotlib, seaborn

## Limitations and Next Steps

* The LLM's decision boundary was aligned to the dataset labels through prompt design rather than fine-tuning. A small labelled validation set would make prompt iteration more rigorous.
* No drift detection is implemented. Monitoring prediction confidence distributions would help identify changes in comment language over time.
* The routing threshold was selected on the test set. In production, it should be chosen using a separate validation split.
* Identity-group bias in the training data is inherited by the model. Fairness-aware reweighting would be needed before real deployment.
* Adding `publication_id` as a feature would help quantify how much of the moderation decision is driven by policy rather than content.
