import pandas as pd
import joblib
import os

# Define paths to models and scalers
MODEL_YOUNG_PATH = "artifacts/model_young.joblib"
MODEL_REST_PATH = "artifacts/model_rest.joblib"
SCALER_YOUNG_PATH = "artifacts/scaler_young.joblib"
SCALER_REST_PATH = "artifacts/scaler_rest.joblib"

# Load models and scalers safely
def load_artifact(path, fallback=None):
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[WARNING] Could not load {path}: {e}")
        return fallback

# Load models/scalers
model_young = load_artifact(MODEL_YOUNG_PATH)
model_rest = load_artifact(MODEL_REST_PATH)
scaler_young = load_artifact(SCALER_YOUNG_PATH, {'cols_to_scale': ['age', 'income_lakhs'], 'scaler': None})
scaler_rest = load_artifact(SCALER_REST_PATH, {'cols_to_scale': ['age', 'income_lakhs'], 'scaler': None})

# Risk normalization
def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    diseases = medical_history.lower().split(" & ")
    total_risk = sum(risk_scores.get(d.strip(), 0) for d in diseases)
    normalized = (total_risk - 0) / (14 - 0)
    return normalized

# Preprocessing
def preprocess_input(input_dict):
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight',
        'smoking_status_Occasional', 'smoking_status_Regular',
        'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    # Initialize dataframe with zeros
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}

    # Fill values
    df['age'] = input_dict.get('Age', 0)
    df['number_of_dependants'] = input_dict.get('Number of Dependants', 0)
    df['income_lakhs'] = input_dict.get('Income in Lakhs', 0)
    df['insurance_plan'] = plan_encoding.get(input_dict.get('Insurance Plan'), 1)
    df['genetical_risk'] = input_dict.get('Genetical Risk', 0)
    df['normalized_risk_score'] = calculate_normalized_risk(input_dict.get('Medical History', 'no disease'))

    # One-hot encoding
    if input_dict.get('Gender') == 'Male':
        df['gender_Male'] = 1
    if input_dict.get('Region') in ['Northwest', 'Southeast', 'Southwest']:
        df[f'region_{input_dict["Region"]}'] = 1
    if input_dict.get('Marital Status') == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    if input_dict.get('BMI Category') in ['Obesity', 'Overweight', 'Underweight']:
        df[f'bmi_category_{input_dict["BMI Category"]}'] = 1
    if input_dict.get('Smoking Status') in ['Occasional', 'Regular']:
        df[f'smoking_status_{input_dict["Smoking Status"]}'] = 1
    if input_dict.get('Employment Status') in ['Salaried', 'Self-Employed']:
        df[f'employment_status_{input_dict["Employment Status"]}'] = 1

    # Scale features safely
    scaler_obj = scaler_young if df['age'].values[0] <= 25 else scaler_rest
    if scaler_obj['scaler'] is not None:
        cols = scaler_obj.get('cols_to_scale', [])
        valid_cols = [col for col in cols if col in df.columns]
        if valid_cols:
            try:
                df[valid_cols] = scaler_obj['scaler'].transform(df[valid_cols])
            except Exception as e:
                print(f"[ERROR] Scaling failed: {e}")
        else:
            print("[WARNING] No valid columns to scale.")

    return df

# Prediction
def predict(input_dict):
    input_df = preprocess_input(input_dict)
    age = input_dict.get('Age', 0)

    if age <= 25:
        if model_young:
            print("[INFO] Using model_young")
            prediction = model_young.predict(input_df)
        else:
            print("[WARNING] model_young is not loaded. Using fallback.")
            prediction = [1000]
    else:
        if model_rest:
            print("[INFO] Using model_rest")
            prediction = model_rest.predict(input_df)
        else:
            print("[WARNING] model_rest is not loaded. Using fallback.")
            prediction = [2000]

    return int(prediction[0])
