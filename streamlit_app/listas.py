"""
Carga las listas de valores válidos (Servicio, Procedencia, Destino,
Tipo de invasivo) desde la hoja 'Listas' de Data_2_24_v3.xlsx.
El archivo vive dentro del propio repo, en data/Data_2_24_v3.xlsx.
"""
import os
import pandas as pd

RUTA_LISTAS = os.path.join(os.path.dirname(__file__), "data", "Data_2_24_v3.xlsx")

# Fallback hardcodeado, usado solo si el archivo no se puede leer
# (por ejemplo si la ruta cambia o el archivo no se subió al repo).
FALLBACK = {
    "Servicio": [
        "UCI infanto juvenil", "UCI neonatal", "UCI", "Neurología", "UCO", "UPA",
        "Sala de cuidados intermedios", "Medicina", "Pensionado", "Diálisis",
        "Unidad emergencia", "Pediatría", "Traumatología",
        "Hospitalización domiciliaria", "Urología", "Ambulatorio", "Hemodinamia",
        "Gine-obstetricia", "Área Dental",
    ],
    "Tipo de invasivo": [
        "Independiente del invasivo", "Catéter epicutáneo", "Catéter hemodiálisis",
        "Catéter implantofix", "Catéter Midline NP periférica",
        "Catéter umbilical arterial", "Catéter umbilical venoso",
        "Catéter venoso central", "Sonda vesical foley",
        "Sonda vesical foley tres lúmenes", "Ventilación mecánica", "DVE", "DVP",
        "MAC", "Traqueotomía", "Cirugía prótesis cadera",
        "Cirugía tumor SNC adultos", "Cirugía tumor SNC niños",
        "Cirugía hernia inguinal adulto c/s malla",
        "Cirugía colecistectomía laparatómica",
        "Cirugía colecistectomía laparoscópica",
        "Cirugía de cesárea c/s trabajo de parto",
        "Endoftalmítis en cirugía de catarata c/s LIO", "Acceso arterial",
        "Acceso venoso periférico",
    ],
    "Procedencia": [
        "Alta", "Anatomìa patológica", "Cirugía", "Emergencia", "Extrasistema",
        "Gine-obstetricia", "H. Penco-Lirquén", "H. Tome", "Hemodinamia",
        "Hospitalización domiciliaria", "Medicina", "Neurológía", "Pabellón",
        "Paciente sigue hospitalizado", "Pensionado", "Traslado en red",
        "Traumatología", "UCI infanto juvenil", "UCI neonatal", "UCI",
        "Unidad emergencia", "Sala de cuidados intermedios", "UCO", "UPA",
        "Urología", "Ambulatorio", "Pediatría", "Fallece", "Oncología",
        "UTI neonatal", "NEO básico",
    ],
    "Destino": [
        "Alta", "Anatomìa patológica", "Cirugía", "Emergencia", "Extrasistema",
        "Gine-obstetricia", "H. Penco-Lirquén", "H. Tome", "Hemodinamia",
        "Hospitalización domiciliaria", "Medicina", "Neurología", "Pabellón",
        "Paciente sigue hospitalizado", "Pensionado", "Traslado en red",
        "Traumatología", "UCI infanto juvenil", "UCI neonatal", "UCI",
        "Unidad emergencia", "Sala de cuidados intermedios", "UCO", "UPA",
        "Urología", "Ambulatorio", "Pediatría", "Fallece", "Oncología",
        "UTI neonatal", "NEO básico",
    ],
}

# Mapeo: nombre de campo usado en la app -> columna en la hoja 'Listas'
COLUMNAS_ORIGEN = {
    "Servicio": "Unidad",
    "Tipo de invasivo": "Tipo de invasivo",
    "Procedencia": "Procedencia",
    "Destino": "Destino",
}


def _extraer_valores_unicos(serie: pd.Series) -> list[str]:
    valores = serie.dropna().astype(str).str.strip()
    valores = valores[valores != ""]
    vistos = set()
    resultado = []
    for v in valores:
        if v not in vistos:
            vistos.add(v)
            resultado.append(v)
    return resultado


def cargar_listas() -> dict[str, list[str]]:
    """Devuelve un dict {campo: [valores]} leído desde la hoja 'Listas'."""
    try:
        df = pd.read_excel(RUTA_LISTAS, sheet_name="Listas")
    except Exception:
        return FALLBACK

    listas = {}
    for campo, columna in COLUMNAS_ORIGEN.items():
        if columna in df.columns:
            valores = _extraer_valores_unicos(df[columna])
            listas[campo] = valores if valores else FALLBACK[campo]
        else:
            listas[campo] = FALLBACK[campo]
    return listas
