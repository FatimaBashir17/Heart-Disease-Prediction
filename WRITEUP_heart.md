# Project 2 — Heart Disease Prediction: Write-Up

**1. Which ML problem did you pick within this project, and why that one over the others on the list?**

I picked heart disease prediction over the other options (diabetes, general disease risk, patient risk classification, readmission prediction, patient segmentation). Like fraud detection in Project 1, it's a well-documented public dataset (the classic UCI Cleveland Heart Disease dataset), it's a clear binary classification problem, and it pairs naturally with a Gen AI layer that explains a risk score in plain language — which is exactly what this project needed to demonstrate.

**2. Where did your dataset come from, and what did the EDA tell you before you built anything?**

The dataset is the Heart Disease Dataset on Kaggle (https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), uploaded by David Lapp, based on the UCI Machine Learning Repository's Cleveland Heart Disease data. It has 1025 rows and 13 patient features plus a `target` column.

Before building anything, the EDA showed:
- The classes are fairly balanced — about 51% disease, 49% no disease — unlike Project 1's heavily imbalanced fraud data.
- No missing values.
- Age, cholesterol, and resting blood pressure all showed some visible difference between patients with and without disease.
- Most importantly: checking for duplicate rows revealed that 723 of the 1025 rows were exact duplicates of other rows, leaving only 302 truly unique patients. This turned out to be the single most important thing the EDA caught, since it directly affected how the train/test split needed to be done.

**3. Which algorithm(s) did you try, which one did you ship, and what evaluation metrics justify that choice?**

I trained and compared Logistic Regression and Random Forest. The first time through, Random Forest scored 100% on every metric — precision, recall, and F1, all perfect. That's not a sign of a great model, it's a sign of a broken evaluation: the duplicate rows meant the same patient could appear in both the training set and the test set, so the model wasn't really being tested on unseen data.

After removing the duplicates and re-training on the clean 302-row dataset:

| Metric (disease class) | Logistic Regression | Random Forest |
|---|---|---|
| Precision | 0.80 | 0.76 |
| Recall | 0.85 | 0.79 |
| F1-score | 0.82 | 0.78 |
| Accuracy | 0.80 | 0.75 |

Logistic Regression came out ahead across the board. With only 241 training rows after removing duplicates, a simpler model generalizes more reliably than a more complex one, which is likely why Random Forest lost some of its edge here. I shipped Logistic Regression.

**4. How exactly does the Gen AI layer use the model's output? Show a real example — input, prediction, and the generated explanation it produced.**

The model's prediction and probability are passed directly into the prompt, and the LLM (Groq, `openai/gpt-oss-120b`) is instructed to write four parts: a plain-language explanation of the result, a general health-education note, an explanation of the medical terms involved, and a few questions the patient could ask their doctor. The prompt also hard-codes the safety rule for this project: never diagnose, never recommend treatment, always frame it as a screening result for a doctor to review.

Real example from testing:

- **Input:** `{"age": 58, "chol": 289, "trestbps": 145, "thalach": 120}`
- **Model prediction:** Higher Risk, 82% probability
- **Generated explanation:**
  > The screening result suggests that, based on the information provided, there is a higher probability (about 82%) of an elevated heart-disease risk, which should be reviewed by your doctor. High cholesterol levels are widely recognized as a factor that can increase the risk of heart-related problems over time. Cholesterol refers to the fatty substance in the blood that, when elevated, can contribute to plaque buildup in arteries. You might ask your doctor: "What factors are most influencing my risk score?", "What lifestyle changes could help lower my risk?", and "Should I have any additional tests to better understand my heart health?"

For comparison, a lower-risk patient (age 34, cholesterol 180, 9% probability) produced a completely different explanation, focused on maintaining healthy ranges rather than reviewing an elevated risk — confirming the explanation is genuinely reacting to the model's output rather than repeating a fixed template. Nowhere in either response does it say the patient "has" heart disease or recommend a specific treatment.

**5. What broke during the build, and what did you change because of it?**

- Random Forest hit a suspicious 100% accuracy at first. It turned out to be duplicate rows in the dataset leaking between the train and test sets. Fixed by dropping duplicates before splitting, which also changed the final model choice from Random Forest to Logistic Regression.
- Originally, the app only showed a full explanation for the first 5 rows of an uploaded batch. Changed this to sort by risk probability and show the top 5 highest-risk patients instead, so a genuinely flagged patient never gets missed just because they weren't near the top of the file.
- The Gen AI layer returns its explanation as JSON so the four parts can be displayed as separate labeled sections instead of one paragraph. The model occasionally wrapped that JSON in formatting it wasn't supposed to add, which broke parsing — added clean-up handling so the app still works when that happens.
