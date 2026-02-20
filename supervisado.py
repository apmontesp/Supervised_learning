import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="AutoML Classifier", layout="wide")

st.title("🚀 Pipeline de Clasificación Inteligente")
st.write("Carga datos de scikit-learn, entrena y despliega modelos en tiempo real.")

# 1. Carga de Datos
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

st.sidebar.header("Configuración")
test_size = st.sidebar.slider("Tamaño de prueba (Test set)", 0.1, 0.5, 0.2)

if st.checkbox("Mostrar datos brutos"):
    st.write(df.head())

# 2. Preprocesamiento (Escalado)
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Feature Extraction (PCA opcional)
st.sidebar.subheader("Feature Extraction")
use_pca = st.sidebar.checkbox("¿Usar PCA?")
if use_pca:
    n_comp = st.sidebar.slider("Componentes PCA", 2, 10, 2)
    pca = PCA(n_components=n_comp)
    X_train_scaled = pca.fit_transform(X_train_scaled)
    X_test_scaled = pca.transform(X_test_scaled)
    st.write(f"Varianza explicada por PCA: {np.sum(pca.explained_variance_ratio_):.2f}")

# 4. Selección de Modelos
st.sidebar.subheader("Elegir Modelo")
algo = st.sidebar.selectbox("Algoritmo", ("LDA", "Naive Bayes", "KNN", "Decision Tree"))

if algo == "LDA":
    model = LDA()
elif algo == "Naive Bayes":
    model = GaussianNB()
elif algo == "KNN":
    k = st.sidebar.slider("K (Vecinos)", 1, 15, 5)
    model = KNeighborsClassifier(n_neighbors=k)
else:
    depth = st.sidebar.slider("Profundidad máxima", 1, 20, 5)
    model = DecisionTreeClassifier(max_depth=depth)

# 5. Entrenamiento y Validación
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

# 6. Despliegue Condicional
st.subheader(f"Resultado del Modelo: {algo}")
col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{acc:.2%}")
    if acc > 0.85:
        st.success("✅ ¡Modelo con desempeño alto detectado! Listo para producción.")
    else:
        st.warning("⚠️ Desempeño medio/bajo. Intenta ajustar parámetros o usar PCA.")

# 7. Gráficas de Desempeño
with col2:
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicción", y="Real"),
                       x=['Maligno', 'Benigno'], y=['Maligno', 'Benigno'],
                       title="Matriz de Confusión")
    st.plotly_chart(fig_cm)

# --- USO REAL (Inferencia) ---
if acc > 0.85:
    st.divider()
    st.subheader("🎮 Probar Modelo en Tiempo Real")
    st.write("Modifica los valores promedio para predecir el tipo de tumor:")
    
    # Creamos inputs dinámicos basados en las columnas originales
    input_data = []
    cols = st.columns(3)
    for i, feature in enumerate(data.feature_names[:6]): # Limitamos a 6 para el ejemplo
        with cols[i % 3]:
            val = st.number_input(f"{feature}", float(df[feature].mean()))
            input_data.append(val)
    
    # Padding con ceros para las features restantes si no se muestran todas
    full_input = input_data + [0] * (len(data.feature_names) - len(input_data))
    
    if st.button("Predecir"):
        # Aplicar el mismo escalado y PCA que el entrenamiento
        obs = np.array(full_input).reshape(1, -1)
        obs_scaled = scaler.transform(obs)
        if use_pca:
            obs_scaled = pca.transform(obs_scaled)
            
        prediction = model.predict(obs_scaled)
        clase = data.target_names[prediction[0]]
        st.info(f"Resultado de la predicción: **{clase.upper()}**")
