import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# ---------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA Y ESTILO
# ---------------------------------------------------------
st.set_page_config(
    page_title="Plataforma Avanzada de Análisis de Datos",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    .explanation-box {
        background-color: #eef6ff;
        border-left: 4px solid #0284c7;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🛡️ Plataforma Avanzada de Análisis de Datos")
st.caption(
    "Sistema Integral de Analítica, Diagnóstico Prescriptivo, Riesgos e Inferencia Estadística"
)

# ---------------------------------------------------------
# FUNCIONES DE CARGA CON CACHÉ
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_y_limpiar_datos(file, hoja=None):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, sheet_name=hoja)

    df = df.drop_duplicates()

    # Procesamiento estandarizado de tipos de datos
    for col in df.columns:
        if df[col].dtype == "object" or isinstance(df[col].dtype, pd.CategoricalDtype):
            df[col] = df[col].fillna("Desconocido").astype(str).str.strip()
        elif np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].fillna(df[col].median())

    return df

# ---------------------------------------------------------
# 1. SIDEBAR: CARGA DE ARCHIVOS Y FILTROS GLOBALES
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración & Datos")
    file = st.file_uploader("Cargar Base de Datos", type=["csv", "xlsx"])

    if file is not None:
        hoja_seleccionada = None
        if file.name.endswith(".xlsx") or file.name.endswith(".xls"):
            try:
                excel_file = pd.ExcelFile(file)
                hojas = excel_file.sheet_names
                hoja_seleccionada = st.selectbox("📄 Selecciona la hoja:", options=hojas)
            except Exception as e:
                st.error(f"Error al abrir Excel: {e}")

        try:
            df_raw = cargar_y_limpiar_datos(file, hoja_seleccionada)
            st.session_state["df_raw"] = df_raw
            st.success("✔ Datos cargados correctamente")
        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")

    # MÓDULO DE FILTROS GLOBALES DINÁMICOS
    df_filtrado = None
    if "df_raw" in st.session_state:
        st.header("🌪️ Filtros Globales")
        df_base = st.session_state["df_raw"].copy()
        
        cat_cols_global = df_base.select_dtypes(include=["object", "category"]).columns
        num_cols_global = df_base.select_dtypes(include=np.number).columns

        # Filtro por variable cualitativa
        if len(cat_cols_global) > 0:
            var_filtro_cat = st.selectbox("Filtrar por Categoría:", ["Todas"] + list(cat_cols_global))
            if var_filtro_cat != "Todas":
                opciones = list(df_base[var_filtro_cat].unique())
                seleccion = st.multiselect(f"Valores de {var_filtro_cat}:", opciones, default=opciones)
                if seleccion:
                    df_base = df_base[df_base[var_filtro_cat].isin(seleccion)]

        # Filtro por rango numérico
        if len(num_cols_global) > 0:
            var_filtro_num = st.selectbox("Filtrar por Rango Numérico (Opcional):", ["Ninguno"] + list(num_cols_global))
            if var_filtro_num != "Ninguno":
                min_val = float(df_base[var_filtro_num].min())
                max_val = float(df_base[var_filtro_num].max())
                rango = st.slider(f"Rango de {var_filtro_num}:", min_val, max_val, (min_val, max_val))
                df_base = df_base[(df_base[var_filtro_num] >= rango[0]) & (df_base[var_filtro_num] <= rango[1])]

        df_filtrado = df_base

        st.markdown("---")
        modulo = st.radio(
            "Selecciona el Módulo",
            [
                "📊 Dashboard",
                "🔍 Análisis & Perfil",
                "🤖 Modelo Predictivo",
                "🚨 Riesgos & Anomalías",
                "🧪 Pruebas de Hipótesis",
            ],
        )

# ---------------------------------------------------------
# VALIDACIÓN GLOBAL DE DATOS
# ---------------------------------------------------------
if df_filtrado is None:
    st.warning("👈 Inicia cargando una base de datos en el panel izquierdo (CSV o Excel).")
    st.stop()

df = df_filtrado.copy()
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns

# ---------------------------------------------------------
# KPI HEADER GLOBAL PERMANENTE
# ---------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Registros Filtrados", f"{len(df):,} / {len(st.session_state['df_raw']):,}")
with kpi2:
    st.metric("Var. Cuantitativas", len(num_cols))
with kpi3:
    st.metric("Var. Cualitativas", len(cat_cols))
with kpi4:
    completitud = round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 1) if df.size > 0 else 0
    st.metric("Calidad de Datos", f"{completitud}%", delta="Excelente" if completitud > 90 else "Atención")

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
                var_sel = st.selectbox("Métrica Principal de Negocio", num_cols, key="dash_main")
                fig_hist = px.histogram(
                    df,
                    x=var_sel,
                    title=f"Distribución General de {var_sel}",
                    marginal="box",
                    color_discrete_sequence=["#0066cc"],
                )
                fig_hist.update_layout(template="plotly_white")
                st.plotly_chart(fig_hist, use_container_width=True)

                st.markdown(
                    f"""<div class="explanation-box">
                    💡 <b>¿Qué significa este gráfico?</b><br>
                    Este gráfico muestra la frecuencia con la que ocurren distintos valores de <b>{var_sel}</b>. 
                    El centro de la caja superior marca el valor típico (mediana), mientras que los puntos extremos representan valores inusualmente altos o bajos.
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("No hay variables numéricas en el dataset filtrado.")

        with col_right:
            st.markdown("#### 💡 Insights Automáticos")
            if len(num_cols) > 0:
                mean_val = df[var_sel].mean()
                median_val = df[var_sel].median()
                std_val = df[var_sel].std()

                st.write(f"* **Promedio:** `{mean_val:.2f}`")
                st.write(f"* **Mediana:** `{median_val:.2f}`")
                st.write(f"* **Desviación Estándar:** `{std_val:.2f}`")

                if abs(mean_val - median_val) > (std_val * 0.2 if std_val > 0 else 0):
                    st.warning("⚠️ La distribución presenta sesgo. Se sugiere la mediana como indicador estable.")
                    st.markdown(
                        """<div class="explanation-box">
                        <b>Explicación:</b> El promedio está siendo arrastrado por valores extremos (muy altos o muy bajos). La <b>mediana</b> representa mejor el comportamiento real del grupo.
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.success("✅ La distribución es simétrica.")
                    st.markdown(
                        """<div class="explanation-box">
                        <b>Explicación:</b> Los datos están equilibrados alrededor del centro. Tanto el promedio como la mediana son confiables para tomar decisiones.
                        </div>""",
                        unsafe_allow_html=True,
                    )

    with tab2:
        if len(num_cols) >= 2:
            col_x, col_y, col_c = st.columns(3)
            var_x = col_x.selectbox("Eje X", num_cols, index=0)
            var_y = col_y.selectbox("Eje Y", num_cols, index=min(1, len(num_cols) - 1))
            var_c = col_c.selectbox("Color por Var. Cualitativa", ["Ninguna"] + list(cat_cols))

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

            st.markdown(
                f"""<div class="explanation-box">
                💡 <b>¿Cómo interpretar este gráfico de dispersión?</b><br>
                Cada punto representa un registro de tu base de datos.<br>
                • <b>Si los puntos suben hacia la derecha:</b> Cuando aumenta <b>{var_x}</b>, también tiende a aumentar <b>{var_y}</b>.<br>
                • <b>Si los puntos bajan hacia la derecha:</b> Cuando aumenta <b>{var_x}</b>, disminuye <b>{var_y}</b>.<br>
                • <b>Si están dispersos sin forma:</b> No hay una relación directa evidente entre ambas variables.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Se requieren al menos 2 variables numéricas.")

# ---------------------------------------------------------
# MÓDULO 2: DIAGNÓSTICO Y PERFILADO
# ---------------------------------------------------------
elif modulo == "🔍 Análisis & Perfil":
    st.subheader("🔍 Diagnóstico Estadístico y Perfil de Datos")

    tab_prof1, tab_prof2, tab_prof3 = st.tabs(
        ["🔗 Correlaciones", "📊 Perfil Cuantitativo", "🏷️ Perfil Cualitativo"]
    )

    with tab_prof1:
        if len(num_cols) >= 2:
            method = st.radio("Método de Correlación", ["pearson", "spearman"], horizontal=True)
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

            st.markdown(
                """<div class="explanation-box">
                💡 <b>¿Cómo leer la Matriz de Correlaciones?</b><br>
                Mide qué tan conectadas están dos variables en una escala de <b>-1.0 a +1.0</b>:<br>
                • <b>Cercano a +1.0 (Azul Oscuro):</b> Relación fuerte positiva (ambas crecen juntas).<br>
                • <b>Cercano a 0.0 (Claro):</b> Independientes (lo que le pase a una no afecta a la otra).<br>
                • <b>Cercano a -1.0:</b> Relación inversa (cuando una sube, la otra baja).
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Se necesitan más columnas numéricas.")

    with tab_prof2:
        st.write("#### Descriptores Estadísticos Avanzados (Cuantitativos)")
        if len(num_cols) > 0:
            stats_df = df[num_cols].describe().T
            stats_df["skewness"] = df[num_cols].skew()
            stats_df["kurtosis"] = df[num_cols].kurt()
            st.dataframe(stats_df.style.background_gradient(cmap="Blues", subset=["mean", "std"]))

            st.markdown(
                """<div class="explanation-box">
                💡 <b>Guía rápida de columnas:</b><br>
                • <b>mean / std:</b> Promedio y nivel de variabilidad (desviación).<br>
                • <b>min / max:</b> Los valores límite registrados.<br>
                • <b>skewness (sesgo):</b> Si es mayor a 1 o menor a -1, la mayoría de los datos se concentran en un extremo.
                </div>""",
                unsafe_allow_html=True,
            )

    with tab_prof3:
        st.write("#### Distribución de Variables Cualitativas")
        if len(cat_cols) > 0:
            var_cat_sel = st.selectbox("Selecciona Variable Cualitativa:", cat_cols)
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

            st.markdown(
                f"""<div class="explanation-box">
                💡 <b>¿Qué muestra este desglose?</b><br>
                Indica la participación de cada grupo dentro de <b>{var_cat_sel}</b>. Permite identificar de un vistazo las categorías dominantes y las que tienen baja participación.
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("No se encontraron variables cualitativas.")

# ---------------------------------------------------------
# MÓDULO 3: PREDICTIVE ENGINE CON EXPLICACIÓN
# ---------------------------------------------------------
elif modulo == "🤖 Modelo Predictivo":
    st.subheader("🤖 Modelado Predictivo de Machine Learning")

    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.markdown("#### Configuración del Modelo")
        
        posibles_targets = [
            c for c in df.columns if df[c].nunique() < (len(df) * 0.8) or np.issubdtype(df[c].dtype, np.number)
        ]
        
        if not posibles_targets:
            posibles_targets = list(df.columns)

        target_var = st.selectbox("Variable Objetivo (Target)", posibles_targets)
        test_size = st.slider("Proporción de Test (%)", 10, 40, 20) / 100

        btn_train = st.button("🚀 Entrenar Modelo", use_container_width=True)

    with col_results:
        if btn_train:
            df_ml = df.copy()

            if df_ml[target_var].nunique() > 50 and (df_ml[target_var].dtype == "object" or isinstance(df_ml[target_var].dtype, pd.CategoricalDtype)):
                st.error("⚠️ La variable seleccionada tiene demasiadas categorías únicas. Selecciona una variable agrupadora (ej. Segmento, Estado, Tipo de Cliente).")
                st.stop()

            is_class = (
                df_ml[target_var].dtype == "object"
                or isinstance(df_ml[target_var].dtype, pd.CategoricalDtype)
                or df_ml[target_var].nunique() < 10
            )

            le_target = None
            if is_class:
                le_target = LabelEncoder()
                df_ml[target_var] = le_target.fit_transform(df_ml[target_var].astype(str))
                model_type = "Clasificación (Random Forest Classifier)"
            else:
                model_type = "Regresión (Random Forest Regressor)"

            X = df_ml.drop(columns=[target_var])
            y = df_ml[target_var]

            for c in X.columns:
                if X[c].dtype == "object" or isinstance(X[c].dtype, pd.CategoricalDtype):
                    X[c] = LabelEncoder().fit_transform(X[c].astype(str))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            if is_class:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                acc = accuracy_score(y_test, preds)
                st.success(f"**Modelo de {model_type}**")
                st.metric("Precisión (Accuracy)", f"{acc*100:.2f}%")

                st.markdown(
                    f"""<div class="explanation-box">
                    💡 <b>¿Qué significa esta Precisión del {acc*100:.1f}%?</b><br>
                    Indica el porcentaje de veces que el modelo acertó la categoría correcta en datos totalmente nuevos. 
                    Un valor superior al <b>75%</b> se considera confiable para toma de decisiones.
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.write("#### 🎯 Matriz de Confusión")
                cm = confusion_matrix(y_test, preds)
                
                unique_labels_encoded = np.unique(np.concatenate((y_test, preds)))
                if le_target is not None:
                    labels_str = le_target.inverse_transform(unique_labels_encoded)
                else:
                    labels_str = [str(l) for l in unique_labels_encoded]

                fig_cm = px.imshow(
                    cm,
                    x=labels_str,
                    y=labels_str,
                    text_auto=True,
                    labels=dict(x="Predicción del Modelo", y="Valor Real"),
                    color_continuous_scale="Blues",
                    template="plotly_white"
                )
                st.plotly_chart(fig_cm, use_container_width=True)

                st.markdown(
                    """<div class="explanation-box">
                    💡 <b>¿Cómo entender la Matriz de Confusión?</b><br>
                    • Los números en la <b>diagonal principal (de arriba-izquierda a abajo-derecha)</b> representan los aciertos exactos del modelo.<br>
                    • Las casillas fuera de la diagonal muestran en qué categorías el modelo se confundió.
                    </div>""",
                    unsafe_allow_html=True,
                )

            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                st.success(f"**Modelo de {model_type}**")
                st.metric("Coeficiente R²", f"{r2:.3f}")

                st.markdown(
                    f"""<div class="explanation-box">
                    💡 <b>¿Qué significa un R² de {r2:.3f}?</b><br>
                    Mide la capacidad del modelo para explicar los cambios en <b>{target_var}</b>:<br>
                    • <b>Cercano a 1.0 (100%):</b> El modelo predice casi perfectamente el valor numérico.<br>
                    • <b>Mayor a 0.70:</b> Excelente capacidad predictiva.<br>
                    • <b>Menor a 0.40:</b> Se necesitan más o mejores datos para predecir esta variable.
                    </div>""",
                    unsafe_allow_html=True,
                )

            importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
            fig_imp = px.bar(
                importances,
                orientation="h",
                title="Importancia Relativa de las Variables",
                template="plotly_white",
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            st.markdown(
                f"""<div class="explanation-box">
                💡 <b>¿Qué revela este gráfico de importancia?</b><br>
                La barra más larga superior (<b>{importances.index[-1]}</b>) es la variable que más influye para determinar el resultado final de <b>{target_var}</b>. Las variables con barras muy cortas casi no influyen en la predicción.
                </div>""",
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# MÓDULO 4: RIESGOS Y ANOMALÍAS CON EXPLICACIÓN
# ---------------------------------------------------------
elif modulo == "🚨 Riesgos & Anomalías":
    st.subheader("🚨 Detección de Riesgos, Anomalías y Segmentación")

    tab_risk1, tab_risk2 = st.tabs(["⚠️ Detección de Anomalías", "🎯 Segmentación (K-Means)"])

    with tab_risk1:
        if len(num_cols) > 0:
            contam = st.slider("Sensibilidad de Detección (%)", 1, 15, 5) / 100

            scaler = StandardScaler()
            scaled_num = scaler.fit_transform(df[num_cols])

            iso = IsolationForest(contamination=contam, random_state=42)
            df["Es_Anomalia"] = iso.fit_predict(scaled_num)
            df["Es_Anomalia"] = df["Es_Anomalia"].map({1: "Normal", -1: "Atípico"})

            num_anom = (df["Es_Anomalia"] == "Atípico").sum()
            st.error(f"Se identificaron **{num_anom}** registros anómalos o atípicos de **{len(df)}** datos.")

            df_anomalias = df[df["Es_Anomalia"] == "Atípico"]
            st.dataframe(df_anomalias.head(10))

            st.markdown(
                f"""<div class="explanation-box">
                💡 <b>¿Qué es un registro atípico / anómalo?</b><br>
                Son filas en tu base de datos cuyo comportamiento combinado se aleja drásticamente del patrón del grupo (pueden ser posibles errores de digitación, fraudes, o casos extraordinarios de negocio que requieren auditoría).
                </div>""",
                unsafe_allow_html=True,
            )

            csv_data = df_anomalias.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Descargar Lista de Anomalías (CSV)",
                data=csv_data,
                file_name="reporte_anomalias.csv",
                mime="text/csv",
            )
        else:
            st.warning("Se requieren variables numéricas.")

    with tab_risk2:
        if len(num_cols) >= 2:
            k_val = st.slider("Número de Segmentos (Clusters)", 2, 8, 3)

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

            st.markdown(
                f"""<div class="explanation-box">
                💡 <b>¿Qué significa esta segmentación?</b><br>
                El algoritmo ha agrupado automáticamente tus registros en <b>{k_val} grupos (clusters)</b> con características similares. Puedes utilizar estos grupos para estrategias diferenciadas (ej. Clientes Premium vs Estándar).
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("Se requieren al menos 2 variables numéricas.")

# ---------------------------------------------------------
# MÓDULO 5: PRUEBAS DE HIPÓTESIS CON EXPLICACIÓN
# ---------------------------------------------------------
elif modulo == "🧪 Pruebas de Hipótesis":
    st.subheader("🧪 Inferencia Estadística y Pruebas de Hipótesis")

    tipo_prueba = st.radio(
        "Selecciona el tipo de evaluación:",
        ["Cualitativa vs Cuantitativa (ANOVA)", "Cualitativa vs Cualitativa (Chi-Cuadrado $\\chi^2$)"],
        horizontal=True,
    )

    # ANOVA
    if tipo_prueba == "Cualitativa vs Cuantitativa (ANOVA)":
        if len(cat_cols) > 0 and len(num_cols) > 0:
            col_g, col_m = st.columns(2)
            group_var = col_g.selectbox("Variable Cualitativa (Grupo)", cat_cols)
            metric_var = col_m.selectbox("Variable Cuantitativa (Respuesta)", num_cols)

            groups = [g[metric_var].values for _, g in df.groupby(group_var) if len(g) > 1]

            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)

                st.markdown("### Resultados ANOVA")
                c1, c2 = st.columns(2)
                c1.metric("Estadístico F", f"{f_stat:.3f}")
                c2.metric("P-Valor", f"{p_val:.5f}")

                if p_val < 0.05:
                    st.success("🟢 **Diferencia Significativa Encontrada**")
                    st.markdown(
                        f"""<div class="explanation-box">
                        💡 <b>Conclusión Práctica (p-valor = {p_val:.5f} < 0.05):</b><br>
                        Existe una diferencia real y estadísticamente comprobada en el promedio de <b>{metric_var}</b> entre las distintas categorías de <b>{group_var}</b>. La diferencia no es casualidad ni azar.
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("🟡 **Sin Diferencia Significativa**")
                    st.markdown(
                        f"""<div class="explanation-box">
                        💡 <b>Conclusión Práctica (p-valor = {p_val:.5f} ≥ 0.05):</b><br>
                        No hay suficiente evidencia estadística para afirmar que las categorías de <b>{group_var}</b> influyan en el resultado promedio de <b>{metric_var}</b>. Todos los grupos se comportan de forma similar.
                        </div>""",
                        unsafe_allow_html=True,
                    )

                fig_box = px.box(df, x=group_var, y=metric_var, template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("Se requiere al menos 1 variable cualitativa y 1 cuantitativa.")

    # CHI-CUADRADO
    else:
        if len(cat_cols) >= 2:
            col_c1, col_c2 = st.columns(2)
            cat_var1 = col_c1.selectbox("Primera Variable Cualitativa:", cat_cols, index=0)
            cat_var2 = col_c2.selectbox("Segunda Variable Cualitativa:", cat_cols, index=min(1, len(cat_cols) - 1))

            if cat_var1 != cat_var2:
                contingency_table = pd.crosstab(df[cat_var1], df[cat_var2])
                chi2, p_val, dof, _ = stats.chi2_contingency(contingency_table)

                st.markdown("### Resultados Chi-Cuadrado ($\chi^2$)")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Estadístico $\chi^2$", f"{chi2:.3f}")
                rc2.metric("P-Valor", f"{p_val:.5f}")
                rc3.metric("Grados de Libertad", dof)

                if p_val < 0.05:
                    st.success("🟢 **Existe Asociación Significativa**")
                    st.markdown(
                        f"""<div class="explanation-box">
                        💡 <b>Conclusión Práctica (p-valor = {p_val:.5f} < 0.05):</b><br>
                        Las variables <b>{cat_var1}</b> y <b>{cat_var2}</b> están estrechamente relacionadas. Pertenecer a una categoría específica sí afecta la probabilidad de pertenecer a la otra.
                        </div>""",
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning("🟡 **Variables Independientes**")
                    st.markdown(
                        f"""<div class="explanation-box">
                        💡 <b>Conclusión Práctica (p-valor = {p_val:.5f} ≥ 0.05):</b><br>
                        <b>{cat_var1}</b> y <b>{cat_var2}</b> son independientes. Saber la categoría de una no te ayuda a predecir la otra.
                        </div>""",
                        unsafe_allow_html=True,
                    )

                fig_chi = px.imshow(
                    contingency_table,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=f"Asociación entre {cat_var1} y {cat_var2}",
                )
                st.plotly_chart(fig_chi, use_container_width=True)
            else:
                st.warning("Selecciona dos variables cualitativas diferentes.")
        else:
            st.warning("Se necesitan al menos 2 variables cualitativas en la base de datos.")
