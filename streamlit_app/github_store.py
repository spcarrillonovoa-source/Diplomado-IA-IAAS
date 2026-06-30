"""
Integración con GitHub para persistir los casos ingresados manualmente.
Usa la API REST de GitHub (contents API) para leer y actualizar
un único archivo Excel dentro del repositorio.
"""
import base64
import io
import requests
import pandas as pd

API_BASE = "https://api.github.com"


class GitHubExcelStore:
    def __init__(self, repo: str, path: str, token: str, branch: str = "main"):
        """
        repo: 'usuario/repositorio'
        path: ruta del archivo dentro del repo, ej. 'data/casos_ingresados.xlsx'
        token: Personal Access Token con permiso de escritura sobre el repo
        branch: rama donde se guardará el archivo
        """
        self.repo = repo
        self.path = path
        self.token = token
        self.branch = branch
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }

    def _contents_url(self):
        return f"{API_BASE}/repos/{self.repo}/contents/{self.path}"

    def leer_excel(self) -> tuple[pd.DataFrame, str | None]:
        """Devuelve (DataFrame, sha). Si el archivo no existe, DataFrame vacío y sha=None."""
        resp = requests.get(
            self._contents_url(), headers=self.headers, params={"ref": self.branch}, timeout=20
        )
        if resp.status_code == 404:
            return pd.DataFrame(), None
        resp.raise_for_status()
        data = resp.json()
        contenido = base64.b64decode(data["content"])
        df = pd.read_excel(io.BytesIO(contenido))
        return df, data["sha"]

    def guardar_excel(self, df: pd.DataFrame, sha: str | None, mensaje_commit: str):
        """Sube el DataFrame como xlsx al repo, creando o actualizando el archivo."""
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        contenido_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        payload = {
            "message": mensaje_commit,
            "content": contenido_b64,
            "branch": self.branch,
        }
        if sha:
            payload["sha"] = sha

        resp = requests.put(self._contents_url(), headers=self.headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

    def agregar_fila(self, fila: dict, mensaje_commit: str):
        """Lee el Excel actual, agrega una fila nueva y vuelve a subirlo."""
        df, sha = self.leer_excel()
        nueva = pd.DataFrame([fila])
        df_actualizado = pd.concat([df, nueva], ignore_index=True)
        self.guardar_excel(df_actualizado, sha, mensaje_commit)
        return df_actualizado
