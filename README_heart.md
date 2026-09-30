# Heart Screening Console — Heart Disease Risk Prediction

A web app that takes patient data, runs it through a trained heart disease risk model, and then uses an LLM to explain the result in plain language — without ever giving a diagnosis or telling the patient what to do medically. This is Project 2 of the AI Engineering capstone (AnalytixCamp / 365 Boot Camp).

## The problem

A risk score on its own (like "82% probability") doesn't mean much to a patient. The goal here was to pair a real ML model with a Gen AI layer that explains what the result shows, in the voice of a nurse describing a screening result — not a doctor handing out a diagnosis or treatment plan. That line matters a lot for this project specifically, since the brief treats crossing it as an automatic deduction.

## Dataset

**Heart Disease Dataset** from Kaggle, uploaded by David Lapp, based on the classic UCI Cleveland Heart Disease dataset.

- Source: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- License: as listed on the Kaggle page (the dataset is derived from the UCI Machine Learning Repository's Heart Disease dataset, released for research/educational use)
- 1025 rows, 13 patient features (age, sex, chest pain type, resting blood pressure, cholesterol, and others) plus a `target` column (1 = heart disease, 0 = no heart disease)

**Important finding during EDA:** the dataset looked suspiciously easy at first. Checking for duplicate rows showed that 723 of the 1025 rows were exact duplicates — only 302 rows were actually unique. This is because this Kaggle version repeats the original ~303-row UCI dataset several times over. The duplicates were removed before splitting into train/test, otherwise the same patient record could end up in both sets and make the model look far more accurate than it really is.

## What I did

**EDA** — checked the class balance (about 51% disease / 49% no disease, a nicely balanced dataset unlike Project 1's fraud data), confirmed there were no missing values, looked at how age, cholesterol, and blood pressure differ between patients with and without disease, and ran a correlation heatmap. Also caught the duplicate-rows issue described above.

**Preprocessing** — scaled the five continuous columns (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) with a single `StandardScaler` fit across all of them together, then split the data with `stratify=y`.

**Modeling** — trained and compared Logistic Regression and Random Forest, after removing the duplicate rows:

| Metric (disease class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.80 | 0.76 |
| Recall | 0.85 | 0.79 |
| F1-score | 0.82 | 0.78 |
| Accuracy | **0.80** | 0.75 |

Before removing duplicates, Random Forest scored a suspicious 100% on every metric — a clear sign of data leakage, not a genuinely good model. After the fix, Logistic Regression came out ahead, most likely because the cleaned dataset is small (only 241 training rows), and a simpler model generalizes better than a more complex one when there isn't much data to learn from. Logistic Regression is the model that shipped.

**Gen AI layer** — using Groq (`openai/gpt-oss-120b`), wired to the model's actual prediction and probability. The prompt has hard rules built in: never diagnose, never recommend treatment, always frame the result as something a doctor should review. It returns four parts for each patient: a plain-language explanation of the result, a general (not personal) health information note, an explanation of the medical terms involved, and a few questions the patient could raise with their doctor.

**The app** — Flask backend serving a JSON API, plain HTML/CSS/JavaScript frontend (no framework, no templating engine), matching the same approach as Project 1. Features:
- Upload a CSV of patient records and get every row screened
- The 5 patients shown with a full explanation are the 5 **highest-risk** patients in the batch, not just the first 5 rows — this way, a flagged patient is never missed just because they weren't near the top of the file
- A separate tool to paste or upload a medical report/document and get a plain-language summary of it (also with the no-diagnosis rule applied)
- Dashboard with total/higher-risk/lower-risk counts and a donut chart
- A chatbot that answers follow-up questions about whichever batch is currently on screen
- History of past batches, and a dark/light mode toggle

## Tech stack

- Python, Flask
- pandas, scikit-learn
- Groq API for the Gen AI layer
- Vanilla HTML/CSS/JavaScript on the frontend

## Running it locally

```
Heart Disease/
├── app.py
├── EDA_heart_disease.ipynb
├── genai_heart.py
├── heart.csv
├── heart_model.pkl
├── heart_scaler.pkl
├── history.json           (created automatically the first time you run it)
├── .env
└── public/
    ├── index.html
    ├── app.js
    ├── style.css
    └── favicon.png
```

1. Install dependencies:
   ```
   pip install flask pandas scikit-learn groq python-dotenv
   ```
2. Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```
3. Run it:
   ```
   python app.py
   ```
4. Open `http://127.0.0.1:5000` and upload a patient CSV (same columns as training: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`; a `target` column is fine too, it just gets ignored).

## What didn't go smoothly

- Random Forest hit a suspicious 100% accuracy at first. Turned out to be duplicate rows in the dataset leaking between train and test. Fixed by dropping duplicates before splitting, which also flipped the final model choice from Random Forest to Logistic Regression.
- Originally only showed the first 5 rows of a batch with a full explanation. Changed this to sort by risk probability and show the top 5 highest-risk patients instead — in a larger, real batch, the first 5 rows might all be low-risk by chance, and a genuinely flagged patient could go unnoticed.
- The Gen AI layer returns its explanation as JSON so the app can display it in labeled sections instead of one big paragraph. Occasionally the model would wrap that JSON in formatting it wasn't supposed to add, which broke parsing — added clean-up handling so the app still works even when that happens.

## Author

Fatima Bashir
