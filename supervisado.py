import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_curve, auc)

# --- Configuración UI ---
st.set_page_config(page_title="Data Scientist Sandbox", layout="wide")
st.title("🔬 Laboratorio de Clasificación Avanzado")

# --- 1. SELECCIÓN DE DATASET ---
st.sidebar.header("1. Configuración de Datos")
ds_name = st.sidebar.selectbox("Selecciona Dataset", ("Iris", "Breast Cancer", "Wine"))

def get_dataset(name):
    if name == "Iris":
        data = datasets.load_iris()
    elif name == "Wine":
        data = datasets.load_wine()
    else:
        data = datasets.load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    return X, y, data.target_names

X, y, target_names = get_dataset(ds_name)
st.write(f"### Dataset: {ds_name} ({X.shape[0]} muestras, {X.shape[1]} características)")

# --- 2. PREPROCESO Y PCA POR VARIANZA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.sidebar.header("2. Feature Extraction")
use_pca = st.sidebar.toggle("Activar PCA")
if use_pca:
    var_target = st.sidebar.slider("Varianza explicada deseada", 0.50, 0.99, 0.95)
    pca_temp = PCA().fit(X_scaled)
    # Calcular cuántos componentes necesitamos para esa varianza
    cum_var = np.cumsum(pca_temp.explained_variance_ratio_)
    n_components = np.argmax(cum_var >= var_target) + 1
    
    pca = PCA(n_components=n_components)
    X_final = pca.fit_transform(X_scaled)
    st.info(f"PCA redujo las dimensiones de {X.shape[1]} a **{n_components}** para mantener el {var_target*100:.0f}% de varianza.")
else:
    X_final = X_scaled

# --- 3. MODELO Y VALIDACIÓN ---
st.sidebar.header("3. Algoritmo y Validación")
algo = st.sidebar.selectbox("Modelo", ("KNN", "LDA", "Naive Bayes", "Decision Tree"))
cv_type = st.sidebar.selectbox("Método de Validación", ("Simple Split", "K-Fold", "Stratified K-Fold"))

# Instanciar Modelo
if algo == "KNN":
    model = KNeighborsClassifier(n_neighbors=st.sidebar.slider("K", 1, 15, 5))
elif algo == "LDA":
    model = LDA()
elif algo == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=st.sidebar.slider("Profundidad", 1, 10, 3))
else:
    model = GaussianNB()

# Ejecución de Validación
if cv_type == "Simple Split":
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    main_acc = accuracy_score(y_test, y_pred)
else:
    folds = st.sidebar.number_input("Número de Folds", 2, 10, 5)
    cv_strat = KFold(n_splits=folds) if cv_type == "K-Fold" else StratifiedKFold(n_splits=folds)
    scores = cross_val_score(model, X_final, y, cv=cv_strat)
    main_acc = scores.mean()
    # Para métricas de test, hacemos un split rápido para visualización
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# --- 4. MÉTRICAS Y VISUALIZACIÓN ---
st.subheader("📊 Resultados de Desempeño")
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
m_col1.metric("Accuracy (Promedio)", f"{main_acc:.2%}")
m_col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2%}")
m_col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2%}")
m_col4.metric("F1-Score", f"{f1_score(y_test, y_pred, average='weighted'):.2%}")

viz_opt = st.tabs(["Matriz de Confusión", "Curva ROC", "Distribución de Clases", "PCA Biplot"])

with viz_opt[0]:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig)

with viz_opt[1]:
    if len(np.unique(y)) == 2: # Solo para binaria (Breast Cancer)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        st.plotly_chart(px.area(x=fpr, y=tpr, title=f'Curva ROC (AUC={auc(fpr, tpr):.2f})', 
                                labels={'x':'Falsos Positivos', 'y':'Verdaderos Positivos'}))
    else:
        st.warning("La curva ROC estándar solo está disponible para clasificación binaria.")

with viz_opt[2]:
    fig_dist = px.histogram(y, x="target", color=y.astype(str), title="Balance de Clases Original")
    st.plotly_chart(fig_dist)

with viz_opt[3]:
    if use_pca and n_components >= 2:
        df_pca = pd.DataFrame(X_final[:, :2], columns=['PC1', 'PC2'])
        df_pca['Clase'] = y.values
        st.plotly_chart(px.scatter(df_pca, x='PC1', y='PC2', color='Clase', title="Visualización 2D (PCA)"))
    else:
        st.info("Activa PCA con al menos 2 componentes para ver esta gráfica.")

# --- 5. DESPLIEGUE ---
if main_acc > 0.80:
    st.success(f"🚀 Modelo desplegado. Desempeño superior al umbral (80%).")
    with st.expander("Probar con nuevos datos"):
        st.write("Introduce valores (se aplicará el escalado automáticamente):")
        # Simulación de inputs para las primeras 4 columnas
        user_vals = [st.number_input(f"Valor {col}", value=float(X[col].mean())) for col in X.columns[:4]]
        if st.button("Clasificar"):
            # Ajuste de dimensiones para el modelo
            input_array = np.array(user_vals + [0]*(X.shape[1]-4)).reshape(1, -1)
            input_proc = scaler.transform(input_array)
            if use_pca: input_proc = pca.transform(input_proc)
            pred = model.predict(input_proc)
            st.balloons()
            st.write(f"### Resultado: **{target_names[pred[0]]}**")
