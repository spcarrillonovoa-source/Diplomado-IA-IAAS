"""
Frontend Streamlit del Modelo Predictivo de IAAS — v8
Modos: predicción individual y predicción en batch.
Los casos individuales se guardan automáticamente en un Excel dentro
del repositorio de GitHub configurado.
"""
import os
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import streamlit as st
import gdown

from preprocesamiento import (
    COLS_CRITERIO, COLUMNAS_CASO, CRITERIOS_VALIDOS,
    calcular_features_criterios_fila, es_criterio_valido,
    nivel_riesgo, preparar_fila,
)
from github_store import GitHubExcelStore
from listas import cargar_listas

# ───────────────────────── Configuración general ─────────────────────────
st.set_page_config(page_title="Predicción IAAS", page_icon="🏥", layout="wide")

FILE_ID_PKL = "1ydemljmhEHc6WVQJWst9ekDgUiMbxyeA"
MODELO_LOCAL = "modelo_iaas3.pkl"

GITHUB_REPO = "spcarrillonovoa-source/Diplomado-IA-IAAS"
GITHUB_PATH_CASOS = "data/casos_ingresados.xlsx"
GITHUB_BRANCH = "main"


# ───────────────────────── Carga de modelo (cacheada) ─────────────────────────
@st.cache_resource(show_spinner="Cargando modelo desde Google Drive…")
def cargar_modelo():
    if not os.path.exists(MODELO_LOCAL):
        url = f"https://drive.google.com/uc?id={FILE_ID_PKL}"
        gdown.download(url, MODELO_LOCAL, quiet=True)
    with open(MODELO_LOCAL, "rb") as f:
        paquete = pickle.load(f)
    return paquete


paquete = cargar_modelo()
modelo = paquete["modelo"]
UMBRAL_OPTIMO_MODELO = paquete.get("umbral_optimo", 0.5)
metricas = paquete.get("metricas", {})


@st.cache_data(show_spinner=False)
def obtener_listas():
    return cargar_listas()


LISTAS = obtener_listas()


# ───────────────────────── Sidebar: umbral y modo ─────────────────────────
st.sidebar.title("🏥 Modelo IAAS")
st.sidebar.markdown("Random Forest balanceado · Criterios IAAS validados")

st.sidebar.subheader("🎛️ Umbral de decisión")
usar_optimo = st.sidebar.checkbox(
    "Usar umbral óptimo del modelo", value=True,
    help=f"Umbral óptimo guardado en el modelo: {UMBRAL_OPTIMO_MODELO:.4f}",
)
if usar_optimo:
    UMBRAL = UMBRAL_OPTIMO_MODELO
else:
    UMBRAL = st.sidebar.slider("Umbral manual", 0.0, 1.0, float(UMBRAL_OPTIMO_MODELO), 0.01)

st.sidebar.caption(f"Umbral en uso: **{UMBRAL:.2f}**")

with st.sidebar.expander("📋 Métricas del modelo (entrenamiento)"):
    for k, v in metricas.items():
        if k not in ["n_train", "n_test", "n_total", "tn", "fp", "fn", "tp"]:
            st.write(f"**{k}**: {v}")

st.sidebar.divider()
modo = st.sidebar.radio("Modo de ingreso de datos", ["🔍 Individual", "📊 Batch (Excel)"])


# ───────────────────────── Conexión a GitHub (lazy) ─────────────────────────
def get_github_store():
    token = st.secrets.get("GITHUB_TOKEN", None)
    if not token:
        return None
    return GitHubExcelStore(
        repo=GITHUB_REPO, path=GITHUB_PATH_CASOS, token=token, branch=GITHUB_BRANCH
    )


# ───────────────────────── Utilidades de UI ─────────────────────────
OPCIONES_CRITERIO = ["Sin hallazgos IAAS"] + CRITERIOS_VALIDOS


def mostrar_resultado(prob, pred, nivel, color, caso):
    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric("Probabilidad de IAAS", f"{prob*100:.1f}%")
        st.metric("Predicción", pred)
        st.markdown(f"**Nivel de riesgo:** {nivel}")
        st.caption(f"Umbral aplicado: {UMBRAL:.2f}")
    with col2:
        n_crit, n_cat, cat_p = calcular_features_criterios_fila(caso)
        st.write(f"**Criterios válidos:** {n_crit}/5")
        st.write(f"**Categorías IAAS distintas:** {n_cat} ({cat_p})")
        crit_detectados = [
            caso.get(c) for c in COLS_CRITERIO if es_criterio_valido(caso.get(c, ""))
        ]
        if crit_detectados:
            st.write("**Criterios activos:**")
            for c in crit_detectados:
                st.write(f"- {c}")
        else:
            st.write("Sin criterios IAAS válidos detectados.")

    st.progress(min(max(prob, 0.0), 1.0))


# ───────────────────────── MODO INDIVIDUAL ─────────────────────────
if modo == "🔍 Individual":
    st.title("🔍 Predicción individual")
    st.caption("Ingresa los datos del caso. Al predecir, el caso queda guardado automáticamente en GitHub.")

    with st.form("form_individual"):
        c1, c2, c3 = st.columns(3)
        with c1:
            servicio = st.selectbox("Servicio", LISTAS["Servicio"])
            procedencia = st.selectbox("Procedencia", LISTAS["Procedencia"])
        with c2:
            destino = st.selectbox("Destino", LISTAS["Destino"])
            fecha_ingreso = st.date_input("Fecha de ingreso", value=dt.date.today())
        with c3:
            araisp = st.selectbox("ARAISP", ["No", "Sí"])
            tipo_invasivo = st.selectbox("Tipo de invasivo", LISTAS["Tipo de invasivo"])

        st.markdown("**Criterios IAAS** (selecciona hasta 5; deja 'Sin hallazgos IAAS' si no aplica)")
        criterios = []
        for i in range(5):
            criterios.append(
                st.selectbox(f"Criterio {i+1}", OPCIONES_CRITERIO, key=f"criterio_{i}")
            )

        enviado = st.form_submit_button("Predecir y guardar")

    if enviado:
        caso = {
            "Servicio": servicio,
            "Procedencia": procedencia,
            "Destino": destino,
            "Fecha de ingreso": fecha_ingreso.isoformat(),
            "ARAISP": None if araisp == "No" else "Sí",
            "Tipo de invasivo": tipo_invasivo,
        }
        for i, c in enumerate(criterios):
            caso[f"Criterio {i+1}"] = c

        X_caso = preparar_fila(caso.copy(), paquete)
        prob = modelo.predict_proba(X_caso)[0][1]
        pred = "IAAS" if prob >= UMBRAL else "NO IAAS"
        nivel, color = nivel_riesgo(prob)

        st.success("Predicción calculada.")
        mostrar_resultado(prob, pred, nivel, color, caso)

        store = get_github_store()
        if store is None:
            st.warning(
                "No se encontró `GITHUB_TOKEN` en los secrets de la app. "
                "El caso NO se guardó en GitHub. Configura el secret para habilitar el guardado automático."
            )
        else:
            fila_guardar = {col: caso.get(col) for col in COLUMNAS_CASO}
            fila_guardar["Prob_IAAS (%)"] = round(prob * 100, 1)
            fila_guardar[f"Prediccion (umbral {UMBRAL:.2f})"] = pred
            fila_guardar["Fecha_registro"] = dt.datetime.now().isoformat(timespec="seconds")
            try:
                with st.spinner("Guardando caso en GitHub…"):
                    store.agregar_fila(
                        fila_guardar,
                        mensaje_commit=f"Nuevo caso individual ({dt.datetime.now().isoformat(timespec='seconds')})",
                    )
                st.success(f"Caso guardado en `{GITHUB_REPO}/{GITHUB_PATH_CASOS}`.")
            except Exception as e:
                st.error(f"No se pudo guardar el caso en GitHub: {e}")


# ───────────────────────── MODO BATCH ─────────────────────────
else:
    st.title("📊 Predicción en batch")
    st.caption(
        "Sube un Excel con las columnas: " + ", ".join(COLUMNAS_CASO) +
        " (opcionalmente con columna `IAAS` para evaluar el modelo)."
    )

    archivo = st.file_uploader("Archivo Excel (.xlsx)", type=["xlsx"])

    if archivo is not None:
        df_batch = pd.read_excel(archivo)
        faltantes = [c for c in COLUMNAS_CASO if c not in df_batch.columns]
        if faltantes:
            st.error(f"El archivo no tiene las columnas requeridas: {faltantes}")
        else:
            tiene_iaas_real = "IAAS" in df_batch.columns
            st.write(f"**Filas cargadas:** {len(df_batch)} · **Tiene columna IAAS:** {tiene_iaas_real}")

            if st.button("Ejecutar predicción en batch"):
                cols_excluir = [
                    "Prob_IAAS (%)", "Prediccion (umbral 0.55)",
                    "Prediccion (umbral 0.63)", "Prediccion (umbral 0.68)",
                    "Prediccion", "Acierto",
                ]
                df_trabajo = df_batch.drop(columns=[c for c in cols_excluir if c in df_batch.columns], errors="ignore")

                probs, preds = [], []
                with st.spinner("Calculando predicciones…"):
                    for _, row in df_trabajo.iterrows():
                        caso_dict = row.to_dict()
                        X_row = preparar_fila(caso_dict, paquete)
                        prob = modelo.predict_proba(X_row)[0][1]
                        probs.append(prob)
                        preds.append("IAAS" if prob >= UMBRAL else "NO IAAS")

                df_trabajo = df_trabajo.copy()
                df_trabajo["Prob_IAAS (%)"] = [round(p * 100, 1) for p in probs]
                df_trabajo[f"Prediccion (umbral {UMBRAL:.2f})"] = preds

                st.success(f"Predicciones completadas: {len(df_trabajo)} casos")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("IAAS", preds.count("IAAS"))
                with c2:
                    st.metric("NO IAAS", preds.count("NO IAAS"))

                st.dataframe(df_trabajo, use_container_width=True)

                buffer = pd.ExcelWriter("salida_temp.xlsx", engine="openpyxl")
                df_trabajo.to_excel(buffer, index=False)
                buffer.close()
                with open("salida_temp.xlsx", "rb") as f:
                    st.download_button(
                        "⬇️ Descargar predicciones (.xlsx)", f,
                        file_name="predicciones_iaas.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )

                # ── Evaluación si hay columna IAAS real ──
                if tiene_iaas_real:
                    st.divider()
                    st.subheader("📈 Evaluación con etiquetas reales")

                    def limpiar_iaas(val):
                        v = str(val).strip().upper()
                        return 1 if v in ["SI", "SÍ", "1", "IAAS", "TRUE", "YES", "S"] else 0

                    from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
                    import matplotlib.pyplot as plt

                    y_true = df_batch["IAAS"].apply(limpiar_iaas).values
                    y_prob_arr = np.array(probs)
                    y_pred_bin = (y_prob_arr >= UMBRAL).astype(int)

                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    espec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    auc = roc_auc_score(y_true, y_prob_arr)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sensibilidad", f"{sens:.3f}")
                    c2.metric("Especificidad", f"{espec:.3f}")
                    c3.metric("AUC-ROC", f"{auc:.3f}")

                    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                    ax = axes[0]
                    cm = np.array([[tn, fp], [fn, tp]])
                    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
                    plt.colorbar(im, ax=ax)
                    etiq = ["No IAAS (0)", "IAAS (1)"]
                    ax.set_xticks([0, 1]); ax.set_xticklabels(etiq, rotation=20)
                    ax.set_yticks([0, 1]); ax.set_yticklabels(etiq)
                    total = cm.sum()
                    labels_cm = [["VN", "FP"], ["FN", "VP"]]
                    for i in range(2):
                        for j in range(2):
                            c = cm[i, j]
                            ax.text(j, i, f"{labels_cm[i][j]}\n{c}\n({c/total*100:.1f}%)",
                                    ha="center", va="center", fontsize=11,
                                    color="white" if c > cm.max() / 2 else "black")
                    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
                    ax.set_title(f"Matriz de Confusión | Umbral {UMBRAL:.2f}", fontweight="bold")

                    ax2 = axes[1]
                    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob_arr)
                    ax2.plot(fpr_arr, tpr_arr, color="#3498db", lw=2, label=f"AUC = {auc:.4f}")
                    ax2.plot([0, 1], [0, 1], "--", color="gray", lw=1)
                    ax2.scatter([1 - espec], [sens], color="#e74c3c", s=100, zorder=5,
                                label=f"Umbral {UMBRAL:.2f}")
                    ax2.set_xlabel("1 - Especificidad (FPR)"); ax2.set_ylabel("Sensibilidad (TPR)")
                    ax2.set_title("Curva ROC", fontweight="bold")
                    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    st.text(classification_report(y_true, y_pred_bin, target_names=["No IAAS", "IAAS"]))
