import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plataforma de Analisis de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS Personalizado para tarjetas ejecutivas
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #0066cc;
    }
    .stApp {
        background-color: #fcfcfc;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ Plataforma de Analisis de Datos")
st.caption(
    "Sistema de Analítica Avanzada, Diagnóstico Prescriptivo y Gestión de Riesgos"
)

# ---------------------------------------------------------
# 1. SIDEBAR: CARGA Y FILTROS GLOBALES (MULTI-HOJA Y MULTI-FORMATO)
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración & Datos")
    file = st.file_uploader("Cargar Base de Datos", type=["csv", "xlsx"])

    if file is not None:
        hoja_seleccionada = None

        # Si es un archivo Excel con múltiples hojas
        if file.name.endswith(".xlsx") or file.name.endswith(".xls"):
            try:
                excel_file = pd.ExcelFile(file)
                hojas = excel_file.sheet_names

                # Menú desplegable para elegir la hoja
                hoja_seleccionada = st.selectbox(
                    "📄 Selecciona la hoja de Excel:", options=hojas
                )
            except Exception as e:
                st.error(f"Error al leer las hojas del Excel: {e}")

        # Control de carga para actualizar cuando cambie el archivo o la hoja seleccionada
        cache_key = f"{file.name}_{hoja_seleccionada}"

        if st.session_state.get("last_loaded_key") != cache_key:
            try:
                if file.name.endswith(".csv"):
                    df_cargado = pd.read_csv(file)
                else:
                    df_cargado = pd.read_excel(
                        file, sheet_name=hoja_seleccionada
                    )

                df_cargado = df_cargado.drop_duplicates()

                # --- PROCESAMIENTO Y LECTURA DE VARIABLES CUALITATIVAS Y CUANTITATIVAS ---
                for col in df_cargado.columns:
                    # Detectar y tratar variables cualitativas (texto/categorías)
                    if df_cargado[col].dtype == "object" or isinstance(df_cargado[col].dtype, pd.CategoricalDtype):
                        df_cargado[col] = (
                            df_cargado[col]
                            .fillna("Desconocido")
                            .astype(str)
                            .str.strip()  # Elimina espacios en blanco accidentales
                        )
                    # Tratamiento de variables cuantitativas
                    elif np.issubdtype(df_cargado[col].dtype, np.number):
                        df_cargado[col] = df_cargado[col].fillna(df_cargado[col].median())

                # Guardar en session state
                st.session_state["df_raw"] = df_cargado
                st.session_state["last_loaded_key"] = cache_key
                st.success("✔ Datos cargados y procesados correctamente")

            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")

    # Selección de Módulo Principal
    if "df_raw" in st.session_state:
        modulo = st.radio(
            "Selecciona el Módulo",
            [
                "📊 Dashboard",
                "🔍 Analisis",
                "🤖 Modelo Predictivo",
                "🚨 Riesgos",
                "🧪 Prueba de Hipótesis",
            ],
        )
        st.markdown("---")

# ---------------------------------------------------------
# VALIDACIÓN GLOBAL DE DATOS
# ---------------------------------------------------------
if "df_raw" not in st.session_state:
    st.warning(
        "👈 Inicia cargando una base de datos en el panel izquierdo (CSV o Excel)."
    )
    st.stop()

df = st.session_state["df_raw"].copy()

# Clasificación precisa de variables cualitativas y cuantitativas
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns

# ---------------------------------------------------------
# KPI HEADER GLOBAL (RESUMEN EJECUTIVO PERMANENTE)
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Total Registros", f"{len(df):,}")

with kpi2:
    st.metric("Variables Cuantitativas", len(num_cols))

with kpi3:
    st.metric("Variables Cualitativas", len(cat_cols))

with kpi4:
    completes_score = round(
        (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 1
    )
    st.metric(
        "Completitud de Datos",
        f"{completes_score}%",
        delta="Excelente" if completes_score > 90 else "Atención",
    )

st.markdown("---")

# ---------------------------------------------------------
# MÓDULO 1: EXECUTIVE DASHBOARD
# ---------------------------------------------------------
if modulo == "📊 Dashboard":
    st.subheader("📈 Resumen Ejecutivo y Tendencias Globales")

    tab1, tab2 = st.tabs(["📌 Vista General", "🔍 Explorador Multivariable"])

    with tab1:
        col_left, col_right = st.columns([2, 1])

        with col_left:
            if len(num_cols) > 0:
                var_sel = st.selectbox(
                    "Métrica Principal de Negocio", num_cols, key="dash_main"
                )

                fig_hist = px.histogram(
                    df,
                    x=var_sel,
                    title=f"Distribución General de {var_sel}",
                    marginal="box",
                    color_discrete_sequence=["#45c7e7"],
                )
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No hay variables numéricas para graficar en esta sección.")

        with col_right:
            st.markdown("#### 💡 Insights Automáticos")
            if len(num_cols) > 0:
                mean_val = df[var_sel].mean()
                median_val = df[var_sel].median()
                std_val = df[var_sel].std()

                st.write(f"* **Promedio:** `{mean_val:.2f}`")
                st.write(f"* **Mediana:** `{median_val:.2f}`")
                st.write(f"* **Desviación Estándar:** `{std_val:.2f}`")

                if abs(mean_val - median_val) > (std_val * 0.2):
                    st.warning(
                        "⚠️ La distribución presenta sesgo significativo. Se recomienda usar la mediana como medida principal."
                    )
                else:
                    st.success(
                        "✅ La distribución es relativamente simétrica."
                    )

    with tab2:
        if len(num_cols) >= 2:
            col_x, col_y, col_c = st.columns(3)
            var_x = col_x.selectbox("Eje X", num_cols, index=0)
            var_y = col_y.selectbox(
                "Eje Y", num_cols, index=min(1, len(num_cols) - 1)
            )
            var_c = col_c.selectbox(
                "Color por Variable Cualitativa (Opcional)", ["Ninguna"] + list(cat_cols)
            )

            color_arg = None if var_c == "Ninguna" else var_c
            fig_scatter = px.scatter(
                df,
                x=var_x,
                y=var_y,
                color=color_arg,
                title=f"Relación entre {var_x} y {var_y}",
                template="plotly_white",
                trendline="ols" if var_c == "Ninguna" else None,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Se requieren al menos 2 variables numéricas para el gráfico de dispersión.")

# ---------------------------------------------------------
# MÓDULO 2: DIAGNOSTICS & PROFILING
# ---------------------------------------------------------
elif modulo == "🔍 Analisis":
    st.subheader("🔍 Diagnóstico Estadístico y Calidad de la Base de Datos")

    tab_prof1, tab_prof2, tab_prof3 = st.tabs(
        ["🔗 Correlaciones", "📊 Perfil Cuantitativo", "🏷️ Perfil Cualitativo"]
    )

    with tab_prof1:
        if len(num_cols) >= 2:
            method = st.radio(
                "Método de Correlación",
                ["pearson", "spearman"],
                horizontal=True,
            )
            corr = df[num_cols].corr(method=method)

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                title=f"Matriz de Correlaciones ({method.capitalize()})",
                color_continuous_scale="Blues",
                aspect="auto",
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Se requieren más columnas numéricas.")

    with tab_prof2:
        st.write("#### Tabla de Descriptores Estadísticos Avanzados (Numéricos)")
        if len(num_cols) > 0:
            stats_df = df[num_cols].describe().T
            stats_df["skewness"] = df[num_cols].skew()
            stats_df["kurtosis"] = df[num_cols].kurt()
            st.dataframe(
                stats_df.style.background_gradient(
                    cmap="Blues", subset=["mean", "std"]
                )
            )

    with tab_prof3:
        st.write("#### Frecuencia de Variables Cualitativas / Categóricas")
        if len(cat_cols) > 0:
            var_cat_sel = st.selectbox("Selecciona la Variable Cualitativa:", cat_cols)
            freq_df = df[var_cat_sel].value_counts().reset_index()
            freq_df.columns = [var_cat_sel, "Frecuencia"]
            freq_df["Porcentaje (%)"] = round((freq_df["Frecuencia"] / len(df)) * 100, 2)

            col_cat_tbl, col_cat_fig = st.columns([1, 1])
            with col_cat_tbl:
                st.dataframe(freq_df)
            with col_cat_fig:
                fig_cat = px.bar(
                    freq_df.head(15),
                    x=var_cat_sel,
                    y="Frecuencia",
                    title=f"Top Categorías de {var_cat_sel}",
                    template="plotly_white",
                    text_auto=True,
                )
                st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No se encontraron variables cualitativas en esta base de datos.")

# ---------------------------------------------------------
# MÓDULO 3: PREDICTIVE ENGINE (IA CON SOPORTE CUALITATIVO)
# ---------------------------------------------------------
elif modulo == "🤖 Modelo Predictivo":
    st.subheader("🤖 Modelado Predictivo Basado en Machine Learning")

    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.markdown("#### Configuración del Modelo")
        target_var = st.selectbox("Variable Objetivo (Target)", df.columns)
        test_size = st.slider("Proporción de Test (%)", 10, 40, 20) / 100

        btn_train = st.button(
            "🚀 Entrenar y Evaluar Modelo", use_container_width=True
        )

    with col_results:
        if btn_train:
            df_ml = df.copy()

            # Detectar si la variable objetivo es cualitativa o categórica discreta
            is_class = (
                df_ml[target_var].dtype == "object"
                or isinstance(df_ml[target_var].dtype, pd.CategoricalDtype)
                or df_ml[target_var].nunique() < 10
            )

            if is_class:
                le_target = LabelEncoder()
                df_ml[target_var] = le_target.fit_transform(
                    df_ml[target_var].astype(str)
                )
                model_type = "Clasificación (Random Forest Classifier)"
            else:
                model_type = "Regresión (Random Forest Regressor)"

            X = df_ml.drop(columns=[target_var])
            y = df_ml[target_var]

            # Codificar automáticamente todas las variables cualitativas de entrada (Features)
            for c in X.columns:
                if X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if is_class:
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)

                st.success(f"**Tipo de Modelo:** {model_type}")
                st.metric("Precisión Global (Accuracy)", f"{acc*100:.2f}%")
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)

                st.success(f"**Tipo de Modelo:** {model_type}")
                st.metric("Coeficiente de Determinación (R²)", f"{r2:.3f}")

            # Importancia de Variables
            importances = pd.Series(
                model.feature_importances_, index=X.columns
            ).sort_values(ascending=True)
            fig_imp = px.bar(
                importances,
                orientation="h",
                title="Importancia Relativa de las Variables (Feature Importance)",
                labels={"value": "Importancia", "index": "Variable"},
                template="plotly_white",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

# ---------------------------------------------------------
# MÓDULO 4: RISK & ANOMALIES
# ---------------------------------------------------------
elif modulo == "🚨 Riesgos":
    st.subheader("🚨 Detección de Riesgos, Anomalías y Segmentación")

    tab_risk1, tab_risk2 = st.tabs(
        ["⚠️ Isolation Forest (Anomalías)", "🎯 K-Means Clustering"]
    )

    with tab_risk1:
        if len(num_cols) > 0:
            contam = st.slider(
                "Sensibilidad de Detección de Anomalías", 0.01, 0.15, 0.05
            )

            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(df[num_cols])

            iso = IsolationForest(contamination=contam, random_state=42)
            df["Anomaly"] = iso.fit_predict(scaled_num)

            anomalies_cnt = sum(df["Anomaly"] == -1)
            st.error(
                f"Se han identificado **{anomalies_cnt}** registros anómalos o atípicos de un total de {len(df)}."
            )

            st.write("#### Registros Atípicos Detectados")
            st.dataframe(df[df["Anomaly"] == -1].head(10))
        else:
            st.warning("Se requieren variables numéricas para la detección de anomalías.")

    with tab_risk2:
        if len(num_cols) >= 2:
            k_val = st.slider("Número de Segmentos (Clusters K)", 2, 8, 3)

            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(df[num_cols])

            km = KMeans(n_clusters=k_val, n_init=10, random_state=42)
            df_cluster = df.copy()
            df_cluster["Cluster"] = km.fit_predict(scaled_num).astype(str)

            fig_cls = px.scatter(
                df_cluster,
                x=num_cols[0],
                y=num_cols[1],
                color="Cluster",
                title=f"Segmentación K-Means ({num_cols[0]} vs {num_cols[1]})",
                template="plotly_white",
            )
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.warning("Se requieren al menos 2 variables numéricas para la segmentación K-Means.")

# ---------------------------------------------------------
# MÓDULO 5: HYPOTHESIS TESTING
# ---------------------------------------------------------
elif modulo == "🧪 Prueba de Hipótesis":
    st.subheader("🧪 Inferencia Estadística y Pruebas de Hipótesis")

    if len(cat_cols) > 0 and len(num_cols) > 0:
        col_g, col_m = st.columns(2)
        group_var = col_g.selectbox("Variable Cualitativa (Factor / Grupo)", cat_cols)
        metric_var = col_m.selectbox("Variable Cuantitativa (Respuesta)", num_cols)

        groups = [
            g[metric_var].values
            for _, g in df.groupby(group_var)
            if len(g) > 1
        ]

        if len(groups) >= 2:
            f_stat, p_val = stats.f_oneway(*groups)

            st.markdown("### Resultados de la Prueba ANOVA")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Estadístico F", f"{f_stat:.3f}")
            res_col2.metric("P-Valor (p-value)", f"{p_val:.5f}")

            if p_val < 0.05:
                st.success(
                    "🟢 **Rechazo de Hipótesis Nula ($H_0$):** Existe diferencia estadísticamente significativa entre las categorías cualitativas analizadas ($p < 0.05$)."
                )
            else:
                st.warning(
                    "🟡 **No se rechaza $H_0$:** No hay suficiente evidencia empírica para afirmar que existe diferencia entre las categorías ($p \ge 0.05$)."
                )

            fig_box = px.box(
                df,
                x=group_var,
                y=metric_var,
                title=f"Comparativa de {metric_var} por la variable cualitativa {group_var}",
                template="plotly_white",
            )
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("La variable cualitativa seleccionada debe tener al menos 2 categorías con datos suficientes.")
    else:
        st.warning(
            "Se requiere al menos una variable cualitativa y una cuantitativa."
        )
        
        
