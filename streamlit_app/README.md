# Predicción IAAS — App Streamlit (v8)

App web basada en `frontend_modelo_iaas_v8.ipynb` para predecir riesgo de IAAS,
en dos modos: **predicción individual** y **predicción en batch (Excel)**.

- El modelo (`modelo_iaas3.pkl`) se descarga automáticamente desde Google Drive
  la primera vez que se ejecuta la app.
- El umbral de decisión es configurable desde la barra lateral (óptimo del
  modelo o manual).
- **Cada caso individual ingresado se guarda automáticamente** en un archivo
  Excel (`data/casos_ingresados.xlsx`) dentro de este mismo repositorio de
  GitHub, vía la API de GitHub (commits automáticos).

## Estructura de archivos

```
streamlit_app/
├── app.py                   # App principal de Streamlit
├── preprocesamiento.py      # Lógica de features, coherente con el notebook v8
├── github_store.py          # Lectura/escritura del Excel en GitHub vía API
├── listas.py                # Carga las listas desplegables desde data/Data_2_24_v3.xlsx
├── data/
│   └── Data_2_24_v3.xlsx    # Fuente de las listas (hoja 'Listas'): Servicio, Procedencia, Destino, Tipo de invasivo
├── requirements.txt
├── secrets.toml.example     # Plantilla de secrets (NO subir el real con el token)
└── .gitignore
```

## 1. Subir estos archivos al repositorio

Repositorio: `https://github.com/spcarrillonovoa-source/Diplomado-IA-IAAS`

Puedes subir esta carpeta completa (`streamlit_app/`) a la raíz del repo, o
ajustar las rutas si prefieres otra estructura. Si cambias la ubicación de
`app.py`, recuerda actualizar el "Main file path" al desplegar en Streamlit
Cloud.

Pasos sugeridos (vía interfaz web de GitHub):
1. Entra al repo → botón **Add file → Upload files**.
2. Arrastra todos los archivos de `streamlit_app/` (mantén la estructura de
   carpetas si subes `data/` también, aunque esa carpeta se crea sola al
   guardar el primer caso).
3. Confirma el commit en la rama `main`.

## 2. Crear el Personal Access Token de GitHub

1. En GitHub: **Settings → Developer settings → Personal access tokens →
   Fine-grained tokens → Generate new token**.
2. Configura:
   - **Repository access**: Solo el repositorio `Diplomado-IA-IAAS`.
   - **Permissions → Contents**: Read and write.
3. Genera el token y cópialo (solo se muestra una vez).

## 3. Desplegar en Streamlit Community Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) y conecta tu cuenta
   de GitHub.
2. Crea una nueva app:
   - **Repository**: `spcarrillonovoa-source/Diplomado-IA-IAAS`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app/app.py` (ajusta si subiste los
     archivos en otra ruta)
3. Antes de lanzar (o después, en **Settings → Secrets**), pega lo siguiente,
   reemplazando con tu token real:

```toml
GITHUB_TOKEN = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

4. Guarda y despliega. La app instalará automáticamente lo de
   `requirements.txt`.

## 4. Uso

- **Modo Individual**: completa el formulario y pulsa "Predecir y guardar".
  Los campos `Servicio`, `Procedencia`, `Destino` y `Tipo de invasivo` se
  despliegan como listas, cargadas automáticamente desde la hoja `Listas`
  del archivo `data/Data_2_24_v3.xlsx` incluido en el repo (si ese archivo
  no está disponible, la app usa una lista de respaldo fija). El resultado
  se muestra en pantalla y el caso (datos + predicción) se agrega como
  nueva fila en `data/casos_ingresados.xlsx` dentro del repo, mediante un
  commit automático.
- **Modo Batch**: sube un Excel con las columnas requeridas (las mismas del
  set de prueba: `Servicio`, `Procedencia`, `Destino`, `Fecha de ingreso`,
  `ARAISP`, `Tipo de invasivo`, `Criterio 1`...`Criterio 5`). Si además
  incluye la columna `IAAS`, la app muestra matriz de confusión, curva ROC
  y métricas de evaluación.

## Notas importantes

- El archivo `data/casos_ingresados.xlsx` se crea automáticamente en el
  primer caso guardado; no es necesario crearlo a mano.
- Si `GITHUB_TOKEN` no está configurado en los secrets, la app sigue
  funcionando para predecir, pero mostrará una advertencia y **no** guardará
  el caso en GitHub.
- Cada guardado genera un commit nuevo en el repositorio (quedará un
  historial completo de cambios al archivo).
