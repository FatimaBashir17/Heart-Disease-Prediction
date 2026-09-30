import os
from dotenv import load_dotenv
from groq import Groq
 
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 
 
def get_heart_explanation(patient_data, prediction, probability):
    """
    patient_data: dict -> e.g. {"age": 58, "chol": 289, "trestbps": 140, ...}
    prediction: str -> "Higher Risk" or "Lower Risk"
    probability: float -> model's confidence score (0 to 1)
    """
    prompt = f"""
    A heart disease risk prediction system analyzed the following patient data:
 
    Patient data: {patient_data}
    Model prediction: {prediction}
    Risk probability: {probability:.2%}
 
    You are writing for a patient, in the voice of a nurse explaining a
    screening result -- not a doctor. Follow these rules strictly:
    - Do NOT give a diagnosis. Do NOT say the patient "has" heart disease.
      Only describe what the screening result shows.
    - Do NOT recommend any treatment, medication, or specific medical action.
    - Do NOT replace or substitute a doctor's judgement.
    - Always frame this as a screening result that a doctor should review,
      not a final answer.
 
    Write four short parts, in plain sentences (no markdown, no asterisks,
    no bullet dashes):
    1. A plain-language explanation of what this result shows.
    2. One general, widely-known piece of health information relevant to
       the result (e.g. general facts about cholesterol or blood pressure),
       phrased as general education, not personal advice.
    3. A one-line explanation of any medical terms used in the patient data
       (e.g. what "cholesterol" or "resting blood pressure" means).
    4. Two or three questions the patient could ask their doctor at their
       next visit, based on this result.
    """
 
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
      #   model="llama3-8b-8192", 
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
 
 
# Test with two example cases
if __name__ == "__main__":
    print("=== Test 1: Higher risk case ===")
    patient_1 = {"age": 58, "chol": 289, "trestbps": 145, "thalach": 120}
    print(get_heart_explanation(patient_1, "Higher Risk", 0.82))
 
    print("\n=== Test 2: Lower risk case ===")
    patient_2 = {"age": 34, "chol": 180, "trestbps": 110, "thalach": 170}
    print(get_heart_explanation(patient_2, "Lower Risk", 0.09))