import streamlit as st
import numpy as np
import joblib

# Cargar los modelos y el nuevo escalador
logistic_model = joblib.load('logistic_regression_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
linear_model = joblib.load('linear_regression_model.pkl')
scaler_selected = joblib.load('scaler_selected.pkl')

st.title("Predicción de Fraude en Transacciones")

# Elegir el modelo
model_choice = st.selectbox(
    "Seleccione el modelo para la predicción:",
    ["Logistic Regression", "Random Forest", "Linear Regression"]
)

# Crear campos para ingresar valores de las características
st.header("Ingrese los valores de las características")
feature_1 = st.number_input("Feature 1 (ejemplo: Tiempo desde última transacción)", value=0.0)
feature_2 = st.number_input("Feature 2 (ejemplo: Distancia al comercio)", value=0.0)
feature_3 = st.number_input("Feature 3 (ejemplo: Edad del cliente)", value=0.0)
feature_4 = st.number_input("Feature 4 (ejemplo: Monto de la transacción)", value=0.0)
feature_5 = st.number_input("Feature 5 (ejemplo: Tasa de fraude del comercio)", value=0.0)

# Organizar los valores ingresados en un array para la predicción
features = np.array([[feature_1, feature_2, feature_3, feature_4, feature_5]])

# Botón para predecir
if st.button("Predecir"):
    # Escalar las características ingresadas
    features_scaled = scaler_selected.transform(features)
    
    # Inicializar predicciones
    logistic_pred = None
    rf_pred = None
    linear_pred = None

    # Seleccionar el modelo según la elección
    if model_choice == "Logistic Regression":
        logistic_pred = logistic_model.predict(features_scaled)[0]
        probability = logistic_model.predict_proba(features_scaled)[0][1]  # Probabilidad de fraude
        
    elif model_choice == "Random Forest":
        rf_pred = rf_model.predict(features_scaled)[0]
        
    elif model_choice == "Linear Regression":
        linear_pred_continuous = linear_model.predict(features_scaled)[0]
        linear_pred = int(linear_pred_continuous >= 0.5)  # Convertir a clasificación binaria

    # Mostrar resultados
    st.write("### Resultados:")
    
    if logistic_pred is not None:
        st.write(f"- **Regresión Logística**: {'Fraude' if logistic_pred == 1 else 'No Fraude'}")
        st.write(f"  - Probabilidad de fraude: {probability:.2f}")
        
    if rf_pred is not None:
        st.write(f"- **Random Forest**: {'Fraude' if rf_pred == 1 else 'No Fraude'}")
        
    if linear_pred is not None:
        st.write(f"- **Regresión Lineal**: {'Fraude' if linear_pred == 1 else 'No Fraude'}")
