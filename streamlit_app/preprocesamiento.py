"""
Funciones de preprocesamiento del modelo IAAS.
Extraídas y mantenidas coherentes con frontend_modelo_iaas_v8.ipynb.
"""
import pandas as pd

CRITERIOS_VALIDOS = [
    # A - Bacteriemia/Fungemia asociada a CVC
    'A.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'A.I.a.e2.Hipotermia igual o menor a 36 °C axilar',
    'A.I.a.e3.Hipotensión',
    'A.I.a.e4. Taquicardia o bradicardia',
    'A.I.a.e5.Apnea en pacientes menores de un año',
    'A.I.a.e6.Eritema y exudado en sitio de inserción del CVC',
    'A.I.b.e1.Detección en uno o más set de hemocultivos periféricos de un microorganismo patógeno no relacionado con otra infección activa en otra localización por el mismo agente',
    'A.I.b.e2.Detección de microorganismo comensal en al menos dos sets de hemocultivos periféricos tomados en sitios anatómicos diferentes no relacionado con otra infección activa en otra localización por el mismo agente',
    'A.I.b.e3.Detección de microorganismo comensal en al menos un set de hemocultivos periféricos y en cultivo de punta de catéter retirado por sospecha clínica de infección, no relacionado con otra infección activa en otra localización por el mismo agente',
    # B - ITU asociada a sonda vesical
    'B.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'B.I.a.e2.Síntomas irritativos vesicales (tenesmo vesical, urgencia miccional, polaquiuria, disuria, dolor suprapúbico)',
    'B.I.a.e3.Dolor costo vertebral a la palpación o espontáneo',
    'B.I.a.e4.Alteración nueva del estado de conciencia en pacientes de 65 o más años',
    'B.I.b.e1.Leucocituria de acuerdo con los valores de referencia del laboratorio que procesó la muestra tomada',
    'B.I.b.e2.Presencia de placas de pus',
    'B.I.b.e3.Presencia de piocitos',
    'B.I.c.e1.Cultivo de orina con no más de dos microorganismos, en el que al menos uno de ellos tiene recuento de más de 100.000 UFC/ml',
    # C - ISQ
    'C.I.a.e1.Presencia de pus (exudado purulento) en el sitio de incisión quirúrgica, incluido el sitio de la salida de drenaje por contrabertura, con o sin cultivos positivos',
    'C.II.a.e1.Fiebre igual o mayor a 38 °C no atribuible a otra causa',
    'C.II.a.e2.Sensibilidad o dolor en la zona de la incisión quirúrgica',
    'C.II.a.e3.Aumento de volumen localizado en la zona de la incisión quirúrgica',
    'C.II.a.e4.Eritema o calor local en la zona de la incisión quirúrgica',
    'C.II.a.e5.La incisión es deliberadamente abierta por un integrante del equipo de salud1 con presencia de exudado que, sin tener aspecto de pus, se describe como turbio, serohemático o seropurulento',
    'C.II.a.e6.Aislamiento de microorganismo en cultivo obtenido con técnica aséptica de la incisión o tejido subcutáneo',
    # D - Diarrea infecciosa
    'D.I.a.e1.Paciente tiene dos o más deposiciones líquidas dentro de 12 horas con o sin otra sintomatología, no atribuible a causas no infecciosas',
    'D.I.b.e1.Si se cuenta con agente etiológico identificado, no hay evidencias que el microorganismo se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    'D.II.a.e1.Paciente presenta un episodio de deposiciones líquidas o disgregadas',
    'D.II.b.e1.Crecimiento de microorganismo patógeno entérico en cultivo de deposiciones o en muestra de hisopado rectal',
    'D.II.b.e2.Microorganismo patógeno entérico detectado por cualquier medio que no sea cultivo',
    'D.II.c.e1.No hay evidencias que el microorganismo se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    # E - C. difficile
    'E.I.a.e1.Presencia de más de una deposición líquida en 12 horas',
    'E.I.a.e2.Presencia de 3 o más deposiciones disgregadas o líquidas en 24 horas',
    'E.II.a.e1.Muestra de deposición positiva a toxina de C. difficile por cualquier técnica de laboratorio, o aislamiento de cepa productora de toxina detectada en deposición por cultivo u otro medio incluida biología molecular',
    # F - Neumonía
    'F.I.a1.e1.Infiltrado',
    'F.I.a1.e2.Condensación',
    'F.I.a1.e3.Cavitación',
    'F.I.a2.e1.Infiltrado nuevo o progresión de uno existente',
    'F.I.a2.e2.Condensación',
    'F.I.a2.e3.Cavitación',
    'F.I.b.e1.Fiebre mayor o igual a 38 °C axilar',
    'F.I.b.e2.Leucopenia (<4.000 leucocitos/mm3) o leucocitosis (>12.000 leucocitos/mm3)',
    'F.I.b.e3.Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.I.b.e4.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC/ml o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    'F.II.a.e1.Infiltrado nuevo o progresión de uno existente',
    'F.II.a.e2.Condensación',
    'F.II.a.e3.Cavitación',
    'F.II.a.e4.Neumatoceles',
    'F.II.b.e1.Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.II.c.e1.Temperatura corporal inestable',
    'F.II.c.e2.Leucopenia (11.000 leucocitos/mm3) con desviación a izquierda (Mayor o igual a 10% de baciliformes o formas más inmaduras)',
    'F.II.c.e3.Aparición de expectoración purulenta, o cambios en las características, o aumento de la cantidad, o aumento en los requerimientos de aspiración de secreciones',
    'F.II.c.e4.Sibilancias, estertores o roncus',
    'F.II.c.e5.Inestabilidad hemodinámica',
    'F.II.c.e6.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC/ml1010 o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    'F.III.a.e1.Presenta Deterioro en el intercambio gaseoso no explicable por otra causa',
    'F.III.b.e1.Aparición de expectoración, aumento o cambio en las características, o aumento de los requerimientos de aspiración o succión de secreciones',
    'F.III.b.e2.Hemoptisis',
    'F.III.b.e3.Aspirado endotraqueal con aislamiento de microorganismo patógeno > 100.000 UFC3/ml o lavado bronco alveolar o cepillo protegido con recuento significativo (104 o 103 ufc/ml respectivamente) o panel molecular con recuento significativo para neumonía de acuerdo con laboratorio local',
    # G - IRA viral
    'G.I.a.e1.Fiebre igual o mayor a 38 °C axilar o hipotermia sin otra causa reconocible',
    'G.I.a.e2.Leucopenia (<4.000 leucocitos/mm3) o leucocitosis (>11.000 leucocitos/mm3)',
    'G.I.a.e3.Tos',
    'G.I.a.e4.Aparición o incremento de producción de expectoración',
    'G.I.a.e5.Roncus',
    'G.I.a.e6.Sibilancias',
    'G.I.a.e7.Distrés respiratorio o síndrome de dificultad respiratoria',
    'G.I.a.e8.Apnea',
    'G.I.a.e9.Bradicardia',
    'G.I.a.e10.Imagen pulmonar no presente al ingreso compatible con infección viral',
    'G.I.b.e1.Detección de agente viral respiratorio por cualquier técnica de laboratorio',
    'G.I.c.e1.No hay evidencias que el agente viral respiratorio se haya encontrado presente o en periodo incubación al momento del ingreso hospitalario',
    # H - COVID-19
    'H.I.a.e1.Fiebre igual o mayor a 37,8 °C axilar',
    'H.I.a.e2.Perdida brusca y completa del olfato (anosmia)',
    'H.I.a.e3.Perdida brusca y completa del gusto (ageusia)',
    'H.I.a.e4.Tos o estornudos',
    'H.I.a.e5.Congestión nasal',
    'H.I.a.e6.Disnea o dificultad respiratoria',
    'H.I.a.e7.Taquipnea',
    'H.I.a.e8.Odinofagia',
    'H.I.a.e9.Mialgia',
    'H.I.a.e10.Debilidad general o fatiga',
    'H.I.a.e11.Dolor torácico',
    'H.I.a.e12.Calofríos',
    'H.I.a.e13.Diarrea',
    'H.I.a.e14.Anorexia o nauseas o vómitos',
    'H.I.a.e15.Cefalea',
    'H.I.b.e1.Prueba PCR para SARS-CoV-2 positiva',
    'H.I.b.e2.Prueba de antígenos para SARS-CoV-2 positiva',
    'H.I.c.e1.Tomografía de tórax con opacidades bilaterales múltiples en vidrio esmerilado, con distribución pulmonar periférica y baja sin otra causa conocida',
    'H.II.a.e1.Fiebre igual o mayor a 37,8 °C axilar',
    'H.II.a.e2.Perdida brusca y completa del olfato (anosmia)',
    'H.II.a.e3. Perdida brusca y completa del gusto (ageusia)',
    'H.II.a.e4.Tos o estornudos',
    'H.II.a.e5.Congestión nasal',
    'H.II.a.e6.Disnea o dificultad respiratoria',
    'H.II.a.e7.Taquipnea',
    'H.II.a.e8.Odinofagia',
    'H.II.a.e9.Mialgia',
    'H.II.a.e10.Debilidad general o fatiga',
    'H.II.a.e11.Dolor torácico',
    'H.II.a.e12.Calofríos',
    'H.II.a.e13.Diarrea',
    'H.II.a.e14.Anorexia o nauseas o vómitos',
    'H.II.a.e15.Cefalea',
    'H.II.b.e1.Prueba PCR para SARS-CoV-2 positiva',
    'H.II.b.e2.Prueba de antígenos para SARS-CoV-2 positiva',
    # I - Endometritis
    'I.I.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'I.I.a.e2.Sensibilidad uterina o subinvolución uterina',
    'I.I.a.e3.Loquios de aspecto purulento o cambio en la evolución de su aspecto o aumento de mal olor',
    'I.II.a.e1.La paciente tiene un cultivo de fluido o tejido endometrial positivo obtenidos intraoperatoriamente, por punción uterina o por aspirado uterino con técnica aséptica hasta 10 días posterior al parto',
    # J - Meningitis/Ventriculitis
    'J.I.a.e1.Detección de microorganismos (cultivo, test molecular) en líquido cefalorraquídeo (LCR) recolectado con técnica aséptica para fines diagnósticos o terapéuticos',
    'J.II.a.e1.Fiebre igual o mayor a 38 °C axilar',
    'J.II.a.e2.Dolor de cabeza',
    'J.II.a.e3.Signos meníngeos',
    'J.II.a.e4.Signos de nervios craneales',
    'J.II.a.e5.Modificación cualitativa o cuantitativa de conciencia.',
    'J.II.a.e6.Apnea (en menores de un año)',
    'J.II.a.e7.Bradicardia (en menores de un año)',
    'J.II.b.e1.LCR con aumento de glóbulos blancos o en los niveles de proteínas o con descenso de nivel de glucosa según rangos reportados por laboratorio local',
    'J.II.b.e2.Microorganismo observados en tinción de Gram del LCR',
    'J.II.b.e3.Identificación en uno o más set de hemocultivos periféricos de un microorganismo no relacionado con otra infección activa en otra localización por el mismo agente',
    # K - Endoftalmitis
    'K.I.a.e1.Paciente presenta identificación de un microorganismo en muestra tomada con técnica aséptica de cámara anterior, posterior o humor vítreo',
    'K.II.a.e1.Dolor ocular',
    'K.II.a.e2.Visión borrosa',
    'K.II.a.e3.Hipopion',
    'K.II.b.e1.Como consecuencia de los signos y síntomas, el médico inicia terapia antibiótica de 2 o más días de duración',
]
CRITERIOS_SET = {c.strip().lower() for c in CRITERIOS_VALIDOS}
CATEGORIA_CRITERIO = {
    'a': 'bacteriemia_cvc', 'b': 'itu_sonda', 'c': 'isq',
    'd': 'diarrea_infecciosa', 'e': 'c_difficile', 'f': 'neumonia',
    'g': 'ira_viral', 'h': 'covid19', 'i': 'endometritis',
    'j': 'meningitis', 'k': 'endoftalmitis',
}
COLS_CRITERIO = ['Criterio 1', 'Criterio 2', 'Criterio 3', 'Criterio 4', 'Criterio 5']
VALORES_NO_CRITERIO = {'sin hallazgos iaas', 'no aplica', 'nan', 'sin_dato', ''}

# Columnas exactas requeridas para un caso, en el orden del set de prueba original
COLUMNAS_CASO = [
    'Servicio', 'Procedencia', 'Destino', 'Fecha de ingreso', 'ARAISP',
    'Tipo de invasivo',
    'Criterio 1', 'Criterio 2', 'Criterio 3', 'Criterio 4', 'Criterio 5',
]


def es_criterio_valido(valor):
    v = str(valor).strip().lower()
    return v not in VALORES_NO_CRITERIO and v in CRITERIOS_SET


def calcular_features_criterios_fila(fila):
    conteo = 0
    categorias = {}
    for col in COLS_CRITERIO:
        val = fila.get(col, '')
        if es_criterio_valido(val):
            conteo += 1
            pref = str(val).strip()[0].lower() if str(val).strip() else ''
            if pref in CATEGORIA_CRITERIO:
                cat = CATEGORIA_CRITERIO[pref]
                categorias[cat] = categorias.get(cat, 0) + 1
    n_cat = len(categorias)
    cat_principal = max(categorias, key=categorias.get) if categorias else 'ninguna'
    return conteo, n_cat, cat_principal


def limpiar_texto(val, col=None):
    v = str(val).strip().lower()
    if v in ['nan', '']:
        v = 'sin_dato'
    return v


def categorizar_dispositivo(tipo):
    tipo = str(tipo).lower()
    if 'catéter venoso central' in tipo or 'cvc' in tipo:
        return 'cvc'
    elif 'sonda vesical' in tipo:
        return 'sonda_vesical'
    elif 'ventilación' in tipo:
        return 'ventilacion_mecanica'
    elif 'epicutáneo' in tipo:
        return 'cateter_epicutaneo'
    elif 'hemodiálisis' in tipo or 'hemodialisis' in tipo:
        return 'cateter_hemodialisis'
    elif 'umbilical' in tipo:
        return 'cateter_umbilical'
    elif 'derivativa' in tipo or 'dve' in tipo or 'dvp' in tipo:
        return 'derivativa_ventricular'
    else:
        return 'otro'


def tiene_araisp_fn(x):
    return 0 if pd.isna(x) or str(x).strip().lower() in ['nan', '', 'no', 'no aplica'] else 1


def preparar_fila(caso, paquete):
    """Convierte un dict de caso en vector de features para el modelo."""
    enc = paquete['encoders']
    feats_cat = paquete['features_categoricas']
    feats_num = paquete['features_numericas']
    feats_all = paquete['features']

    caso = dict(caso)
    for col in ['Servicio', 'Procedencia', 'Destino', 'Tipo de invasivo']:
        caso[col] = limpiar_texto(caso.get(col, 'sin_dato'))
    _mapa_tipo = {
        'epicutáneo': 'catéter epicutáneo',
        'dve': 'derivativa ventricular externa',
        'dvp': 'derivativa ventricular peritoneal',
    }
    caso['Tipo de invasivo'] = _mapa_tipo.get(caso['Tipo de invasivo'], caso['Tipo de invasivo'])

    n_crit, n_cat, cat_p = calcular_features_criterios_fila(caso)
    caso['n_criterios'] = n_crit
    caso['n_cat_criterios'] = n_cat
    caso['categoria_principal'] = cat_p
    caso['tiene_araisp'] = tiene_araisp_fn(caso.get('ARAISP', ''))
    caso['tipo_dispositivo_cat'] = categorizar_dispositivo(caso.get('Tipo de invasivo', ''))

    fi = caso.get('Fecha de ingreso', None)
    try:
        caso['anio'] = pd.to_datetime(fi, errors='raise').year if fi is not None and not pd.isna(fi) else 2025
    except Exception:
        caso['anio'] = 2025

    row = {}
    for col in feats_cat:
        val = caso.get(col, 'sin_dato')
        val_str = str(val)
        if val_str not in [str(c) for c in enc[col].classes_]:
            val_str = str(enc[col].classes_[0])
        row[col + '_enc'] = enc[col].transform([val_str])[0]
    for col in feats_num:
        row[col] = caso.get(col, 0)

    return pd.DataFrame([row])[feats_all]


def nivel_riesgo(prob):
    if prob < 0.25:
        return '🟢 BAJO', '#2ecc71'
    if prob < 0.50:
        return '🟡 MODERADO', '#f39c12'
    if prob < 0.75:
        return '🔴 ALTO', '#e74c3c'
    return '🚨 MUY ALTO', '#c0392b'
