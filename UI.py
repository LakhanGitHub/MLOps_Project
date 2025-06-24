import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


# Set MLflow tracking
mlflow.set_tracking_uri('https://dagshub.com/LakhanGitHub/MLOps_Project.mlflow/')
model_name = 'Best_Model'


# Cache model loading to avoid re-downloading on every prediction
@st.cache_resource
def load_model_from_staging():
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=["Staging"])
        
        if versions:
            run_id = versions[0].run_id
            logged_model = f"runs:/{run_id}/Best Model"  # Adjust artifact name if needed
            model = mlflow.pyfunc.load_model(logged_model)
            return model
        else:
            st.error("⚠️ No model found in Staging stage.")
            return None
    except Exception as e:
        st.error(f"Error loading model:\n{e}")
        return None


# Load model at app start
loaded_model = load_model_from_staging()


# Streamlit UI
st.title("💧 Water Potability Prediction App")
st.write("Provide water quality parameters below to predict if the water is **safe to drink**.")


# Input Form
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
        solids = st.number_input("Solids (ppm)", min_value=0.0, value=10000.0)

    with col2:
        chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0)
        sulfate = st.number_input("Sulfate (ppm)", min_value=0.0, value=350.0)
        conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, value=500.0)

    with col3:
        organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=10.0)
        trihalomethanes = st.number_input("Trihalomethanes (µg/L)", min_value=0.0, value=80.0)
        turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0)

    submit = st.form_submit_button("Predict")


# Prediction Logic
if submit:
    if loaded_model is not None:
        input_data = pd.DataFrame({
            'ph': [ph],
            'Hardness': [hardness],
            'Solids': [solids],
            'Chloramines': [chloramines],
            'Sulfate': [sulfate],
            'Conductivity': [conductivity],
            'Organic_carbon': [organic_carbon],
            'Trihalomethanes': [trihalomethanes],
            'Turbidity': [turbidity]
        })

        try:
            prediction = loaded_model.predict(input_data)[0]
            if prediction == 1:
                st.success("✅ The water is **Safe to Drink**.")
            else:
                st.error("The water is **Not Safe to Drink**.")
        except Exception as e:
            st.error(f"Error during prediction:\n{e}")
