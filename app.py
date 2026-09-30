import os
import json
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
app = Flask(__name__, static_folder="public", static_url_path="")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

HISTORY_FILE = "history.json"
GEN_MODEL = "openai/gpt-oss-120b"
NUMERIC_COLS = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

with open("heart_model.pkl", "rb") as f:
    heart_model = pickle.load(f)
with open("heart_scaler.pkl", "rb") as f:
    heart_scaler = pickle.load(f)

# Holds recently scored batch, for the chatbot
LAST_CONTEXT = {"summary": None, "results": [], "chat_log": []}

def preprocess(df):
    """Scale the same numeric columns and ensure exact column order."""
    df = df.copy()
    
    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
            
    df = df[expected_cols]
    df[NUMERIC_COLS] = heart_scaler.transform(df[NUMERIC_COLS])
    return df

def get_heart_explanation(patient_data, prediction, probability):
    """Returns a dict with explanation, general_info, terms, questions.
    Safety rule (from the brief): never diagnose, never prescribe
    treatment, always frame as a screening result for a doctor to review."""
    prompt = f"""
    A heart disease risk screening system analyzed this patient data:

    Patient data: {patient_data}
    Model prediction: {prediction}
    Risk probability: {probability:.2%}

    You are writing for a patient, in the voice of a nurse explaining a
    screening result -- not a doctor. Rules:
    - Never give a diagnosis. Never say the patient "has" heart disease.
      Only describe what the screening result shows.
    - Never recommend treatment, medication, or a specific medical action.
    - Never replace a doctor's judgement. Frame this as a screening
      result a doctor should review, not a final answer.
    - Plain sentences only, no markdown, no asterisks.

    Respond with ONLY a raw JSON object (no code fences, no extra text),
    with exactly these keys:
    "explanation": one short paragraph on what this result shows
    "general_info": one short paragraph of general, widely-known health
      education relevant to the result (not personal advice)
    "terms": one short paragraph explaining any medical terms in the
      patient data in plain language
    "questions": an array of 2 to 3 short questions the patient could
      ask their doctor at their next visit
    """

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    raw = response.choices[0].message.content.strip()

    raw = raw.replace('```json', '').replace('```', '').strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "explanation": raw,
            "general_info": "",
            "terms": "",
            "questions": []
        }


def get_document_summary(document_text):
    prompt = f"""
    Summarize the following medical report or document in plain,
    easy-to-understand language for a patient.

    Rules:
    - Do not add a diagnosis or interpretation beyond what the document
      states.
    - Do not recommend treatment or medical action.
    - Just summarize what the document says, in plain sentences, no
      markdown, no asterisks.
    - End with one line reminding the reader to discuss the report with
      their doctor.

    Document:
    {document_text}
    """
    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    history = history[:20]
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)
    return history


# Page
@app.route("/")
def home():
    return app.send_static_file("index.html")


# API routes
@app.route("/api/history", methods=["GET"])
def api_history():
    return jsonify(load_history())


@app.route("/api/predict", methods=["POST"])
def api_predict():
    file = request.files["file"]
    df = pd.read_csv(file)

    if "target" in df.columns:
        df = df.drop("target", axis=1)

    df_scaled = preprocess(df)

    predictions = heart_model.predict(df_scaled)
    probabilities = heart_model.predict_proba(df_scaled)[:, 1]

    total_rows = int(len(df_scaled))
    total_higher = int(predictions.sum())
    total_lower = int(total_rows - total_higher)
    risk_pct = round((total_higher / total_rows) * 100, 1) if total_rows else 0

    # Sort by highest prob
    df_results = df_scaled.copy()
    df_results['prediction'] = predictions
    df_results['probability'] = probabilities
    df_results['original_row_num'] = df.index + 1
    
    df_sorted = df_results.sort_values(by='probability', ascending=False)

    results = []
    for i in range(min(5, len(df_sorted))):
        row_data = df_sorted.iloc[i]
        orig_row_num = int(row_data['original_row_num'])
        
        # original unscaled vals for readable explanation
        row_dict = df.iloc[orig_row_num - 1].round(2).to_dict()  
        
        label = "Higher Risk" if row_data['prediction'] == 1 else "Lower Risk"
        prob = float(row_data['probability'])
        explanation = get_heart_explanation(row_dict, label, prob)

        results.append({
            "row": orig_row_num,
            "prediction": label,
            "probability": prob,
            "explanation": explanation
        })

    entry = {
        "timestamp": datetime.now().strftime("%d %b, %I:%M %p"),
        "total": total_rows,
        "higher": total_higher,
        "lower": total_lower,
    }
    save_history_entry(entry)

    LAST_CONTEXT["summary"] = (
        f"Batch of {total_rows} patients screened: {total_higher} higher risk, "
        f"{total_lower} lower risk ({risk_pct}% higher-risk rate)."
    )
    LAST_CONTEXT["results"] = results
    LAST_CONTEXT["chat_log"] = []

    return jsonify({
        "total_rows": total_rows,
        "total_higher": total_higher,
        "total_lower": total_lower,
        "risk_pct": risk_pct,
        "results": results
    })



@app.route("/api/summarize", methods=["POST"])
def api_summarize():
    if "file" in request.files and request.files["file"].filename:
        file = request.files["file"]
        document_text = file.read().decode("utf-8", errors="ignore")
    else:
        document_text = request.form.get("text", "").strip()

    if not document_text:
        return jsonify({"summary": "No document text was provided."}), 400

    summary = get_document_summary(document_text[:6000])  # keep prompt size sane
    return jsonify({"summary": summary})


@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"reply": "Type a question about the results above."})

    if not LAST_CONTEXT["summary"]:
        return jsonify({"reply": "Screen a batch first, then ask me about it."})

    rows_context = "\n".join(
        f"- Row {r['row']}: {r['prediction']} ({r['probability']:.1%} probability)"
        for r in LAST_CONTEXT["results"]
    )

    system_context = f"""
    You are a healthcare screening assistant. Answer questions about the
    batch of patients just screened. Be concise (2-4 lines).

    Batch summary: {LAST_CONTEXT['summary']}

    Individual rows scored:
    {rows_context}

    Rules:
    - Never give a diagnosis or recommend treatment.
    - Never invent numbers not shown above.
    - If asked something the data doesn't cover, say so plainly.
    - Reply in the same language/style the question is written in
      (English, Urdu, or a Roman Urdu/English mix).
    - Plain sentences only, no markdown, no asterisks.
    """

    messages = [{"role": "system", "content": system_context}]
    for turn in LAST_CONTEXT["chat_log"][-6:]:
        messages.append(turn)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(model=GEN_MODEL, messages=messages)
    reply = response.choices[0].message.content

    LAST_CONTEXT["chat_log"].append({"role": "user", "content": user_message})
    LAST_CONTEXT["chat_log"].append({"role": "assistant", "content": reply})

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)
