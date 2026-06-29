import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gdown
import os
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# ══════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="Predictor IAAS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# ESTILOS
# ══════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background-color: #f8fafc; }

    .stApp > header { background: transparent; }

    /* Header principal */
    .header-box {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    .header-box h1 { margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.5px; }
    .header-box p  { margin: 0.3rem 0 0; font-size: 0.95rem; opacity: 0.85; }

    /* Tarjetas de resultado */
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    .result-iaas    { background: #fee2e2; color: #991b1b; border: 2px solid #fca5a5; }
    .result-no-iaas { background: #dcfce7; color: #166534; border: 2px solid #86efac; }

    /* Métrica de probabilidad */
    .prob-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #64748b;
        margin-bottom: 0.2rem;
    }
    .prob-value {
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
    }

    /* Sección */
    .section-title {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #94a3b8;
        margin-bottom: 0.8rem;
        padding-bottom: 0.4rem;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Badges de riesgo */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    /* Ocultar footer de Streamlit */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# CONSTANTES Y FUNCIONES DEL MODELO
# ══════════════════════════════════════════════
CRITERIOS_VALIDOS = [
    'A.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'A.I.a.e2.Hipotermia igual o menor a 36 °C axilar',
    'A.I.a.e3.Hipotensión',
    'A.I.a.e4. Taquicardia o bradicardia',
    'A.I.a.e5.Apnea en pacientes menores de un año',
    'A.I.a.e6.Eritema y exudado en sitio de inserción del CVC',
    'A.I.b.e1.Detección en uno o más set de hemocultivos periféricos de un microorganismo patógeno no relacionado con otra infección activa en otra localización por el mismo agente',
    'A.I.b.e2.Detección de microorganismo comensal en al menos dos sets de hemocultivos periféricos tomados en sitios anatómicos diferentes no relacionado con otra infección activa en otra localización por el mismo agente',
    'A.I.b.e3.Detección de microorganismo comensal en al menos un set de hemocultivos periféricos y en cultivo de punta de catéter retirado por sospecha clínica de infección, no relacionado con otra infección activa en otra localización por el mismo agente',
    'B.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'B.I.a.e2.Síntomas irritativos vesicales (tenesmo vesical, urgencia miccional, polaquiuria, disuria, dolor suprapúbico)',
    'B.I.a.e3.Dolor costo vertebral a la palpación o espontáneo',
    'B.I.a.e4.Alteración nueva del estado de conciencia en pacientes de 65 o más años',
    'B.I.b.e1.Leucocituria de acuerdo con los valores de referencia del laboratorio que procesó la muestra tomada',
    'B.I.b.e2.Presencia de placas de pus',
    'B.I.b.e3.Presencia de piocitos',
    'B.I.c.e1.Cultivo de orina con no más de dos microorganismos, en el que al menos uno de ellos tiene recuento de más de 100.000 UFC/ml',
    'C.I.a.e1.Presencia de pus (exudado purulento) en el sitio de incisión quirúrgica, incluido el sitio de la salida de drenaje por contrabertura, con o sin cultivos positivos',
    'C.II.a.e1.Fiebre igual o mayor a 38 °C no atribuible a otra causa',
    'C.II.a.e2.Sensibilidad o dolor en la zona de la incisión quirúrgica',
    'C.II.a.e3.Aumento de volumen localizado en la zona de la incisión quirúrgica',
    'C.II.a.e4.Eritema o calor local en la zona de la incisión quirúrgica',
    'C.II.a.e5.La incisión es deliberadamente abierta por un integrante del equipo de salud1 con presencia de exudado que, sin tener aspecto de pus, se describe como turbio, serohemático o seropurulento',
    'C.II.a.e6.Aislamiento de microorganismo en cultivo obtenido con técnica aséptica de la incisión o tejido subcutáneo',
    'D.I.a.e1.Paciente tiene dos o más deposiciones líquidas dentro de 12 horas con o sin otra sintomatología, no atribuible a causas no infecciosas',
    'D.I.b.e1.Si se cuenta con agente etiológico identificado, no hay evidencias que el microorganismo se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    'D.II.a.e1.Paciente presenta un episodio de deposiciones líquidas o disgregadas',
    'D.II.b.e1.Crecimiento de microorganismo patógeno entérico en cultivo de deposiciones o en muestra de hisopado rectal',
    'D.II.b.e2.Microorganismo patógeno entérico detectado por cualquier medio que no sea cultivo',
    'D.II.c.e1.No hay evidencias que el microorganismo se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    'E.I.a.e1.Presencia de más de una deposición líquida en 12 horas',
    'E.I.a.e2.Presencia de 3 o más deposiciones disgregadas o líquidas en 24 horas',
    'E.II.a.e1.Muestra de deposición positiva a toxina de C. difficile por cualquier técnica de laboratorio, o aislamiento de cepa productora de toxina detectada en deposición por cultivo u otro medio incluida biología molecular',
    'F.I.a1.e1.Infiltrado', 'F.I.a1.e2.Condensación', 'F.I.a1.e3.Cavitación',
    'F.I.a2.e1.Infiltrado nuevo o progresión de uno existente', 'F.I.a2.e2.Condensación', 'F.I.a2.e3.Cavitación',
    'F.I.b.e1.Fiebre mayor o igual a 38 °C axilar',
    'F.I.b.e2.Leucopenia (<4.000 leucocitos/mm3) o leucocitosis (>12.000 leucocitos/mm3)',
    'F.I.b.e3.Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.I.b.e4.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC/ml o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    'F.II.a.e1.Infiltrado nuevo o progresión de uno existente', 'F.II.a.e2.Condensación', 'F.II.a.e3.Cavitación', 'F.II.a.e4.Neumatoceles',
    'F.II.b.e1.Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.II.c.e1.Temperatura corporal inestable',
    'F.II.c.e2.Leucopenia (11.000 leucocitos/mm3) con desviación a izquierda (Mayor o igual a 10% de baciliformes o formas más inmaduras)',
    'F.II.c.e3.Aparición de expectoración purulenta, o cambios en las características, o aumento de la cantidad, o aumento en los requerimientos de aspiración de secreciones',
    'F.II.c.e4.Sibilancias, estertores o roncus', 'F.II.c.e5.Inestabilidad hemodinámica',
    'F.II.c.e6.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC/ml1010 o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    'F.III.a.e1.Presenta Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.III.b.e1.Aparición de expectoración, aumento o cambio en las características, o aumento de los requerimientos de aspiración o succión de secreciones',
    'F.III.b.e2.Hemoptisis',
    'F.III.b.e3.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC3/ml o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    'G.I.a.e1.Fiebre igual o mayor a 38 °C axilar o hipotermia sin otra causa reconocible',
    'G.I.a.e2.Leucopenia (<4.000 leucocitos/mm3) o leucocitosis (>11.000 leucocitos/mm3)',
    'G.I.a.e3.Tos', 'G.I.a.e4.Aparición o incremento de producción de expectoración',
    'G.I.a.e5.Roncus', 'G.I.a.e6.Sibilancias',
    'G.I.a.e7.Distrés respiratorio o síndrome de dificultad respiratoria',
    'G.I.a.e8.Apnea', 'G.I.a.e9.Bradicardia',
    'G.I.a.e10.Imagen pulmonar no presente al ingreso compatible con infección viral',
    'G.I.b.e1.Detección de agente viral respiratorio por cualquier técnica de laboratorio',
    'G.I.c.e1.No hay evidencias que el agente viral respiratorio se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    'H.I.a.e1.Fiebre igual o mayor a 37,8 °C axilar',
    'H.I.a.e2.Perdida brusca y completa del olfato (anosmia)',
    'H.I.a.e3.Perdida brusca y completa del gusto (ageusia)',
    'H.I.a.e4.Tos o estornudos', 'H.I.a.e5.Congestión nasal',
    'H.I.a.e6.Disnea o dificultad respiratoria', 'H.I.a.e7.Taquipnea',
    'H.I.a.e8.Odinofagia', 'H.I.a.e9.Mialgia', 'H.I.a.e10.Debilidad general o fatiga',
    'H.I.a.e11.Dolor torácico', 'H.I.a.e12.Calofríos', 'H.I.a.e13.Diarrea',
    'H.I.a.e14.Anorexia o nauseas o vómitos', 'H.I.a.e15.Cefalea',
    'H.I.b.e1.Prueba PCR para SARS-CoV-2 positiva', 'H.I.b.e2.Prueba de antígenos para SARS-CoV-2 positiva',
    'H.I.c.e1.Tomografía de tórax con opacidades bilaterales múltiples en vidrio esmerilado, con distribución pulmonar periférica y baja sin otra causa conocida',
    'H.II.a.e1.Fiebre igual o mayor a 37,8 °C axilar',
    'H.II.a.e2.Perdida brusca y completa del olfato (anosmia)',
    'H.II.a.e3. Perdida brusca y completa del gusto (ageusia)',
    'H.II.a.e4.Tos o estornudos', 'H.II.a.e5.Congestión nasal',
    'H.II.a.e6.Disnea o dificultad respiratoria', 'H.II.a.e7.Taquipnea',
    'H.II.a.e8.Odinofagia', 'H.II.a.e9.Mialgia', 'H.II.a.e10.Debilidad general o fatiga',
    'H.II.a.e11.Dolor torácico', 'H.II.a.e12.Calofríos', 'H.II.a.e13.Diarrea',
    'H.II.a.e14.Anorexia o nauseas o vómitos', 'H.II.a.e15.Cefalea',
    'H.II.b.e1.Prueba PCR para SARS-CoV-2 positiva', 'H.II.b.e2.Prueba de antígenos para SARS-CoV-2 positiva',
    'I.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'I.I.a.e2.Sensibilidad uterina o subinvolución uterina',
    'I.I.a.e3.Loquios de aspecto purulento o cambio en la evolución de su aspecto o aumento de mal olor',
    'I.II.a.e1.La paciente tiene un cultivo de fluido o tejido endometrial positivo obtenidos intraoperatoriamente, por punción uterina o por aspirado uterino con técnica aséptica hasta 10 días posterior al parto',
    'J.I.a.e1.Detección de microorganismos (cultivo, test molecular) en líquido cefalorraquídeo (LCR) recolectado con técnica aséptica para fines diagnósticos o terapéuticos',
    'J.II.a.e1.Fiebre igual o mayor a 38 °C axilar', 'J.II.a.e2.Dolor de cabeza',
    'J.II.a.e3.Signos meníngeos', 'J.II.a.e4.Signos de nervios craneales',
    'J.II.a.e5.Modificación cualitativa o cuantitativa de conciencia.',
    'J.II.a.e6.Apnea (en menores de un año)', 'J.II.a.e7.Bradicardia (en menores de un año)',
    'J.II.b.e1.LCR con aumento de glóbulos blancos o en los niveles de proteínas o con descenso de nivel de glucosa según rangos reportados por laboratorio local',
    'J.II.b.e2.Microorganismo observados en tinción de Gram del LCR',
    'J.II.b.e3.Identificación en uno o más set de hemocultivos periféricos de un microorganismo no relacionado con otra infección activa en otra localización por el mismo agente',
    'K.I.a.e1.Paciente presenta identificación de un microorganismo en muestra tomada con técnica aséptica de cámara anterior, posterior o humor vítreo',
    'K.II.a.e1.Dolor ocular', 'K.II.a.e2.Visión borrosa', 'K.II.a.e3.Hipopion',
    'K.II.b.e1.Como consecuencia de los signos y síntomas, el médico inicia terapia antibiótica de 2 o más días de duración',
]

CRITERIOS_SET = {c.strip().lower() for c in CRITERIOS_VALIDOS}
CATEGORIA_CRITERIO = {
    'a': 'bacteriemia_cvc', 'b': 'itu_sonda', 'c': 'isq',
    'd': 'diarrea_infecciosa', 'e': 'c_difficile', 'f': 'neumonia',
    'g': 'ira_viral', 'h': 'covid19', 'i': 'endometritis',
    'j': 'meningitis', 'k': 'endoftalmitis',
}
COLS_CRITERIO   = ['Criterio 1', 'Criterio 2', 'Criterio 3', 'Criterio 4', 'Criterio 5']
VALORES_NO_CRIT = {'sin hallazgos iaas', 'no aplica', 'nan', 'sin_dato', ''}

OPCIONES_SIN_DATO = ['sin_dato']

def es_criterio_valido(valor):
    v = str(valor).strip().lower()
    return v not in VALORES_NO_CRIT and v in CRITERIOS_SET

def calcular_features_criterios_fila(fila):
    conteo, categorias = 0, {}
    for col in COLS_CRITERIO:
        val = fila.get(col, '')
        if es_criterio_valido(val):
            conteo += 1
            pref = str(val).strip()[0].lower() if str(val).strip() else ''
            if pref in CATEGORIA_CRITERIO:
                cat = CATEGORIA_CRITERIO[pref]
                categorias[cat] = categorias.get(cat, 0) + 1
    n_cat = len(categorias)
    cat_p = max(categorias, key=categorias.get) if categorias else 'ninguna'
    return conteo, n_cat, cat_p

def limpiar_texto(val):
    v = str(val).strip().lower()
    return 'sin_dato' if v in ['nan', ''] else v

def categorizar_dispositivo(tipo):
    tipo = str(tipo).lower()
    if 'catéter venoso central' in tipo or 'cvc' in tipo: return 'cvc'
    elif 'sonda vesical' in tipo:                          return 'sonda_vesical'
    elif 'ventilación' in tipo:                            return 'ventilacion_mecanica'
    elif 'epicutáneo' in tipo:                             return 'cateter_epicutaneo'
    elif 'hemodiálisis' in tipo or 'hemodialisis' in tipo: return 'cateter_hemodialisis'
    elif 'umbilical' in tipo:                              return 'cateter_umbilical'
    elif 'derivativa' in tipo or 'dve' in tipo or 'dvp' in tipo: return 'derivativa_ventricular'
    else:                                                  return 'otro'

def tiene_araisp_fn(x):
    return 0 if pd.isna(x) or str(x).strip().lower() in ['nan', '', 'no', 'no aplica'] else 1

def preparar_fila(caso, paquete):
    enc        = paquete['encoders']
    feats_cat  = paquete['features_categoricas']
    feats_num  = paquete['features_numericas']
    feats_all  = paquete['features']

    for col in ['Servicio', 'Procedencia', 'Destino', 'Tipo de invasivo']:
        caso[col] = limpiar_texto(caso.get(col, 'sin_dato'))

    _mapa = {'epicutáneo': 'catéter epicutáneo',
             'dve': 'derivativa ventricular externa',
             'dvp': 'derivativa ventricular peritoneal'}
    caso['Tipo de invasivo'] = _mapa.get(caso['Tipo de invasivo'], caso['Tipo de invasivo'])

    n_crit, n_cat, cat_p = calcular_features_criterios_fila(caso)
    caso['n_criterios']          = n_crit
    caso['n_cat_criterios']      = n_cat
    caso['categoria_principal']  = cat_p
    caso['tiene_araisp']         = tiene_araisp_fn(caso.get('ARAISP', ''))
    caso['tipo_dispositivo_cat'] = categorizar_dispositivo(caso.get('Tipo de invasivo', ''))

    fi = caso.get('Fecha de ingreso', None)
    try:
        caso['anio'] = pd.to_datetime(fi, errors='raise').year if fi is not None and not pd.isna(fi) else 2025
    except Exception:
        caso['anio'] = 2025

    row = {}
    for col in feats_cat:
        val     = caso.get(col, 'sin_dato')
        val_str = str(val)
        if val_str not in [str(c) for c in enc[col].classes_]:
            val_str = str(enc[col].classes_[0])
        row[col + '_enc'] = enc[col].transform([val_str])[0]
    for col in feats_num:
        row[col] = caso.get(col, 0)

    return pd.DataFrame([row])[feats_all]

def nivel_riesgo(prob):
    if prob < 0.25: return '🟢 BAJO',      '#16a34a'
    if prob < 0.50: return '🟡 MODERADO',  '#d97706'
    if prob < 0.75: return '🔴 ALTO',      '#dc2626'
    return              '🚨 MUY ALTO',     '#7f1d1d'

# ══════════════════════════════════════════════
# CARGA DEL MODELO (cached)
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelo desde Google Drive…")
def cargar_modelo():
    FILE_ID  = '1OO3IOYFwtA-Xm3FZ03SYGlWAoS4BtiIU'
    destino  = '/tmp/modelo_iaas.pkl'
    if not os.path.exists(destino):
        gdown.download(f'https://drive.google.com/uc?id={FILE_ID}', destino, quiet=True)
    with open(destino, 'rb') as f:
        return pickle.load(f)

# ══════════════════════════════════════════════
# SIDEBAR — navegación y métricas del modelo
# ══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 Predictor IAAS")
    st.markdown("---")
    modo = st.radio(
        "Modo de ingreso",
        ["📋 Caso manual", "📂 Batch (Excel)"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Cargar modelo
    try:
        paquete = cargar_modelo()
        modelo  = paquete['modelo']
        UMBRAL  = paquete.get('umbral_optimo', 0.5)
        met     = paquete.get('metricas', {})

        st.markdown("**Modelo cargado ✅**")
        st.caption(f"Umbral óptimo: `{UMBRAL:.3f}`")
        st.markdown("---")
        st.markdown("**Métricas (set de prueba)**")
        col1, col2 = st.columns(2)
        col1.metric("AUC-ROC",    f"{met.get('auc_roc', 0):.3f}")
        col2.metric("Accuracy",   f"{met.get('accuracy', 0):.3f}")
        col1.metric("Sensibilidad", f"{met.get('recall', 0):.3f}")
        col2.metric("Precisión",   f"{met.get('precision', 0):.3f}")
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.stop()

# ══════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════
st.markdown("""
<div class="header-box">
    <h1>🏥 Predictor de Infecciones Asociadas a la Atención en Salud</h1>
    <p>Random Forest Balanceado · Predicción individual o por lote · Umbral óptimo aplicado automáticamente</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# OBTENER OPCIONES VÁLIDAS DEL MODELO
# ══════════════════════════════════════════════
cats      = paquete.get('categorias_validas', {})
enc       = paquete['encoders']

def opciones(col):
    return sorted(enc[col].classes_.tolist()) if col in enc else ['sin_dato']

# ══════════════════════════════════════════════
# MODO 1: CASO MANUAL
# ══════════════════════════════════════════════
if modo == "📋 Caso manual":

    st.markdown("#### Datos del caso")

    with st.form("form_caso"):
        c1, c2, c3 = st.columns(3)
        with c1:
            servicio   = st.selectbox("Servicio",          opciones('Servicio'))
            procedencia = st.selectbox("Procedencia",      opciones('Procedencia'))
        with c2:
            destino    = st.selectbox("Destino",           opciones('Destino'))
            tipo_inv   = st.selectbox("Tipo de invasivo",  opciones('Tipo de invasivo'))
        with c3:
            araisp     = st.selectbox("ARAISP",            ['No', 'Sí', 'No aplica'])
            fecha_ing  = st.date_input("Fecha de ingreso", value=None)

        st.markdown("**Criterios IAAS**")
        opciones_criterio = ['Sin hallazgos IAAS'] + CRITERIOS_VALIDOS
        cc = st.columns(5)
        criterios = []
        for i, col in enumerate(cc):
            criterios.append(col.selectbox(f"Criterio {i+1}", opciones_criterio, key=f"crit_{i}"))

        submitted = st.form_submit_button("🔍 Predecir", use_container_width=True, type="primary")

    if submitted:
        caso = {
            'Servicio':        servicio,
            'Procedencia':     procedencia,
            'Destino':         destino,
            'Tipo de invasivo': tipo_inv,
            'ARAISP':          araisp,
            'Fecha de ingreso': str(fecha_ing) if fecha_ing else None,
            'Criterio 1':      criterios[0],
            'Criterio 2':      criterios[1],
            'Criterio 3':      criterios[2],
            'Criterio 4':      criterios[3],
            'Criterio 5':      criterios[4],
        }

        X    = preparar_fila(caso.copy(), paquete)
        prob = modelo.predict_proba(X)[0][1]
        pred = 'IAAS' if prob >= UMBRAL else 'NO IAAS'
        nivel, color = nivel_riesgo(prob)

        st.markdown("---")
        r1, r2, r3 = st.columns([2, 2, 3])

        with r1:
            css_class = "result-iaas" if pred == "IAAS" else "result-no-iaas"
            st.markdown(f'<div class="result-card {css_class}">{"⚠️ " if pred=="IAAS" else "✅ "}{pred}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="prob-label">Probabilidad IAAS</div><div class="prob-value" style="color:{color}">{prob*100:.1f}%</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="prob-label">Nivel de riesgo</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:1.4rem;font-weight:700;color:{color};margin-top:0.2rem">{nivel}</div>', unsafe_allow_html=True)
            n_crit, _, _ = calcular_features_criterios_fila(caso)
            st.markdown(f'<div class="prob-label" style="margin-top:1rem">Criterios IAAS activos</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:2rem;font-weight:700">{n_crit}/5</div>', unsafe_allow_html=True)

        with r3:
            # Barra de probabilidad
            fig, ax = plt.subplots(figsize=(5, 1.2))
            cmap = LinearSegmentedColormap.from_list('r', ['#16a34a', '#d97706', '#dc2626'], N=100)
            ax.barh(0, prob,       height=0.5, color=cmap(prob), edgecolor='#333')
            ax.barh(0, 1 - prob, left=prob, height=0.5, color='#e2e8f0', edgecolor='#ccc')
            ax.axvline(UMBRAL, color='#1e3a5f', lw=2, ls='--', label=f'Umbral {UMBRAL:.2f}')
            ax.set_xlim(0, 1); ax.set_yticks([])
            ax.set_xlabel('Probabilidad de IAAS', fontsize=9)
            ax.legend(fontsize=8, loc='upper right')
            ax.spines[['top', 'right', 'left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════
# MODO 2: BATCH
# ══════════════════════════════════════════════
else:
    st.markdown("#### Carga de archivo Excel")

    fuente = st.radio("Fuente del archivo", ["📁 Subir archivo", "🔗 Desde Google Drive (set de prueba)"], horizontal=True)

    df_batch = None

    if fuente == "📁 Subir archivo":
        archivo = st.file_uploader("Sube el archivo Excel con los casos", type=["xlsx"])
        if archivo:
            df_batch = pd.read_excel(archivo)

    else:
        if st.button("⬇️ Cargar set de prueba desde Drive"):
            with st.spinner("Descargando desde Google Drive…"):
                try:
                    FILE_ID = '1wR6OKLboJxZkKeyy8p946Zyd0gwlNAvN'
                    destino = '/tmp/set_prueba.xlsx'
                    gdown.download(
                        f'https://drive.google.com/uc?id={FILE_ID}&export=download',
                        destino, quiet=True, fuzzy=True
                    )
                    df_batch = pd.read_excel(destino)
                    st.success(f"✅ Archivo cargado: {len(df_batch)} filas")
                except Exception as e:
                    st.error(f"Error al descargar: {e}")

    if df_batch is not None:
        st.markdown(f"**{len(df_batch)} casos cargados** · Columnas: {list(df_batch.columns)}")
        TIENE_IAAS = 'IAAS' in df_batch.columns

        COLS_EXCLUIR = ['Prob_IAAS (%)', 'Prediccion', 'Acierto'] + \
                       [c for c in df_batch.columns if 'Prediccion (umbral' in c]
        df_trabajo = df_batch.drop(columns=[c for c in COLS_EXCLUIR if c in df_batch.columns], errors='ignore')

        if st.button("▶️ Ejecutar predicciones", type="primary"):
            probs, preds = [], []
            bar = st.progress(0, text="Procesando casos…")

            for i, (_, row) in enumerate(df_trabajo.iterrows()):
                try:
                    X    = preparar_fila(row.to_dict(), paquete)
                    prob = modelo.predict_proba(X)[0][1]
                except Exception:
                    prob = 0.0
                pred = 'IAAS' if prob >= UMBRAL else 'NO IAAS'
                probs.append(round(prob * 100, 1))
                preds.append(pred)
                bar.progress((i + 1) / len(df_trabajo), text=f"Caso {i+1}/{len(df_trabajo)}")

            bar.empty()
            df_resultado = df_trabajo.copy()
            df_resultado['Prob_IAAS (%)']             = probs
            df_resultado[f'Prediccion (umbral {UMBRAL:.2f})'] = preds
            if TIENE_IAAS:
                def limpiar_iaas(v):
                    v = str(v).strip().lower()
                    if v in ('si', 'sí'): return 1
                    if v == 'no':         return 0
                    return None
                iaas_real = df_resultado['IAAS'].apply(limpiar_iaas)
                iaas_pred = [1 if p == 'IAAS' else 0 for p in preds]
                df_resultado['Acierto'] = ['✓ Correcto' if r == p else '✗ Error'
                                           for r, p in zip(iaas_real, iaas_pred)]

            # Resumen
            from collections import Counter
            cnt = Counter(preds)
            m1, m2, m3 = st.columns(3)
            m1.metric("Total casos",  len(preds))
            m2.metric("IAAS",         cnt['IAAS'],    delta=None)
            m3.metric("NO IAAS",      cnt['NO IAAS'], delta=None)

            # Gráficos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5))
            prob_vals = [p / 100 for p in probs]
            cmap = LinearSegmentedColormap.from_list('r', ['#16a34a', '#d97706', '#dc2626'], N=100)
            ax1.hist(probs, bins=20, color='#2d6a9f', edgecolor='white', alpha=0.85)
            ax1.axvline(UMBRAL * 100, color='#dc2626', lw=2, ls='--', label=f'Umbral {UMBRAL*100:.0f}%')
            ax1.set_xlabel('Probabilidad IAAS (%)'); ax1.set_ylabel('N° casos')
            ax1.set_title('Distribución de probabilidades'); ax1.legend()
            ax1.spines[['top', 'right']].set_visible(False)

            if TIENE_IAAS and 'Acierto' in df_resultado.columns:
                from sklearn.metrics import confusion_matrix
                iaas_real_clean = df_resultado['IAAS'].apply(limpiar_iaas).dropna()
                iaas_pred_clean = [1 if p == 'IAAS' else 0
                                   for p, v in zip(preds, df_resultado['IAAS'].apply(limpiar_iaas)) if v is not None]
                if len(iaas_real_clean) > 0:
                    cm = confusion_matrix(iaas_real_clean, iaas_pred_clean)
                    im = ax2.imshow(cm, cmap='Blues')
                    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
                    ax2.set_xticklabels(['NO IAAS', 'IAAS']); ax2.set_yticklabels(['NO IAAS', 'IAAS'])
                    ax2.set_xlabel('Predicción'); ax2.set_ylabel('Real')
                    ax2.set_title('Matriz de Confusión')
                    for i in range(2):
                        for j in range(2):
                            ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                                     fontsize=14, fontweight='bold',
                                     color='white' if cm[i, j] > cm.max() / 2 else 'black')
            else:
                counts = [cnt['IAAS'], cnt['NO IAAS']]
                ax2.bar(['IAAS', 'NO IAAS'], counts, color=['#dc2626', '#16a34a'], edgecolor='white')
                ax2.set_title('Distribución de predicciones')
                ax2.spines[['top', 'right']].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            # Tabla
            st.markdown("**Resultados detallados**")
            st.dataframe(df_resultado, use_container_width=True, height=400)

            # Descarga
            buf = io.BytesIO()
            df_resultado.to_excel(buf, index=False, engine='openpyxl')
            st.download_button(
                "⬇️ Descargar resultados Excel",
                data=buf.getvalue(),
                file_name="predicciones_iaas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
