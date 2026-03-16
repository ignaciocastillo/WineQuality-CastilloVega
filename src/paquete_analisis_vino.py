from pathlib import Path
from functools import wraps
from typing import Union

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import umap.umap_ as umap


# ---------------------------------------------------------------------------
# Decoradores de validación
# ---------------------------------------------------------------------------

def _requiere_escalado(metodo):
    """Valida que escalar_variables() haya sido ejecutado antes del método."""
    @wraps(metodo)
    def wrapper(self, *args, **kwargs):
        if self.X_scaled is None:
            raise ValueError(
                f"{metodo.__name__} requiere ejecutar escalar_variables() primero."
            )
        return metodo(self, *args, **kwargs)
    return wrapper


def _requiere_pca(metodo):
    """Valida que ejecutar_pca() haya sido ejecutado antes del método."""
    @wraps(metodo)
    def wrapper(self, *args, **kwargs):
        if self.pca_model is None:
            raise ValueError(
                f"{metodo.__name__} requiere ejecutar ejecutar_pca() primero."
            )
        return metodo(self, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class AnalisisVino:
    """
    Pipeline de análisis no supervisado sobre el dataset de calidad de vino.

    Flujo esperado:
        1. separar_quality()
        2. escalar_variables()
        3. Métodos de PCA, clustering, t-SNE o UMAP según necesidad.
    """

    def __init__(self, ruta_csv: str) -> None:
        """
        Inicializa el analizador cargando el CSV y preparando carpetas de salida.

        Args:
            ruta_csv: Ruta relativa al CSV desde la raíz del proyecto.

        Raises:
            FileNotFoundError: Si el archivo CSV no existe en la ruta indicada.
        """
        self.project_root = Path(__file__).resolve().parents[1]
        self.csv_path = (self.project_root / ruta_csv).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en: {self.csv_path}")

        self.fig_path = self.project_root / "outputs" / "figures"
        self.tab_path = self.project_root / "outputs" / "tables"
        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.tab_path.mkdir(parents=True, exist_ok=True)

        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()

        # Estado interno — se puebla durante el pipeline
        self.X: pd.DataFrame | None = None
        self.y: pd.Series | None = None
        self.X_scaled: pd.DataFrame | None = None
        self.scaler: StandardScaler | None = None
        self.pca_model: PCA | None = None
        self.pca_scores: pd.DataFrame | None = None
        self.kmeans_model: KMeans | None = None
        self.hac_model: AgglomerativeClustering | None = None

    # -----------------------------------------------------------------------
    # Helpers privados
    # -----------------------------------------------------------------------

    def _guardar_figura(self, fig: plt.Figure, nombre: str) -> None:
        """Guarda una figura en la carpeta de figuras."""
        fig.savefig(self.fig_path / nombre, dpi=300)

    def _guardar_tabla(self, df: pd.DataFrame, nombre: str, **kwargs) -> None:
        """Guarda un DataFrame como CSV en la carpeta de tablas."""
        df.to_csv(self.tab_path / nombre, **kwargs)

    def _scatter_2d(
        self,
        coords: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str | None,
        titulo: str,
        nombre_archivo: str,
        guardar: bool,
    ) -> tuple[plt.Figure, pd.DataFrame]:
        """
        Genera un scatter 2D.

        Returns:
            Tupla (figura, dataframe con coordenadas y color).
        """
        df_plot = coords[[x_col, y_col]].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        if color_col is not None and color_col in coords.columns:
            df_plot[color_col] = coords[color_col].values
            scatter = ax.scatter(df_plot[x_col], df_plot[y_col], c=df_plot[color_col])
            fig.colorbar(scatter)
        else:
            ax.scatter(df_plot[x_col], df_plot[y_col])

        ax.set(title=titulo, xlabel=x_col, ylabel=y_col)
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, nombre_archivo)

        plt.close(fig)
        return fig, df_plot

    def _evaluar_kmeans_rango(self, k_min: int, k_max: int, n_init: int) -> pd.DataFrame:
        """Calcula inercia y silhouette para un rango de valores de k."""
        resultados = [
            {
                "k": k,
                "inertia": (modelo := KMeans(n_clusters=k, random_state=42, n_init=n_init).fit(self.X_scaled)).inertia_,
                "silhouette_score": silhouette_score(self.X_scaled, modelo.labels_),
            }
            for k in range(k_min, k_max + 1)
        ]
        return pd.DataFrame(resultados)

    # -----------------------------------------------------------------------
    # Exploración del dataset
    # -----------------------------------------------------------------------

    def resumen_datos(self) -> pd.DataFrame:
        """Retorna y guarda un resumen básico del dataset (filas, columnas, duplicados)."""
        resumen = pd.DataFrame({
            "métrica": ["n_filas", "n_columnas", "duplicados"],
            "valor": [self.df.shape[0], self.df.shape[1], int(self.df.duplicated().sum())],
        })
        self._guardar_tabla(resumen, "resumen_datos.csv", index=False)
        return resumen

    def columnas(self) -> pd.DataFrame:
        """Retorna y guarda la lista de columnas del dataset."""
        tabla = pd.DataFrame({"columnas": self.df.columns})
        self._guardar_tabla(tabla, "columnas.csv", index=False)
        return tabla

    def nulos_por_columna(self) -> pd.DataFrame:
        """Retorna y guarda el conteo de valores nulos por columna."""
        tabla = self.df.isnull().sum().rename_axis("columna").reset_index(name="nulos")
        self._guardar_tabla(tabla, "nulos_por_columna.csv", index=False)
        return tabla

    def estadisticas(self) -> pd.DataFrame:
        """Retorna y guarda las estadísticas descriptivas del dataset."""
        tabla = self.df.describe().T
        self._guardar_tabla(tabla, "estadisticas_descriptivas.csv")
        return tabla

    # -----------------------------------------------------------------------
    # Preprocesamiento
    # -----------------------------------------------------------------------

    def separar_quality(self) -> None:
        """Separa la columna 'quality' como variable objetivo (y) del resto (X)."""
        self.y = self.df["quality"].copy()
        self.X = self.df.drop(columns=["quality"]).copy()
        print(f"separar_quality — X: {self.X.shape} | y: {self.y.shape}")

    def escalar_variables(self) -> pd.DataFrame:
        """
        Estandariza X con StandardScaler.

        Raises:
            ValueError: Si separar_quality() no fue ejecutado antes.
        """
        if self.X is None:
            raise ValueError("Primero debe ejecutar separar_quality().")

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        tabla = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        self._guardar_tabla(tabla, "variables_escaladas.csv", index=False)
        return tabla

    # -----------------------------------------------------------------------
    # Visualizaciones exploratorias
    # -----------------------------------------------------------------------

    def distribucion_quality(self, guardar: bool = True) -> plt.Figure:
        """Grafica la distribución de frecuencias de la variable quality."""
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=self.df["quality"], ax=ax)
        ax.set(title="Distribución de la variable quality", xlabel="Quality", ylabel="Frecuencia")
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, "distribucion_quality.png")

        plt.close(fig)
        return fig

    def matriz_correlacion(self, guardar: bool = True) -> pd.DataFrame:
        """Calcula y visualiza la matriz de correlación entre variables numéricas."""
        corr = self.df.corr(numeric_only=True)
        self._guardar_tabla(corr, "matriz_correlacion.csv")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Matriz de correlación")
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, "matriz_correlacion.png")

        plt.close(fig)
        return corr

    # -----------------------------------------------------------------------
    # PCA
    # -----------------------------------------------------------------------

    @_requiere_escalado
    def ejecutar_pca(
        self,
        n_components: int | float | None = None,
        svd_solver: str = "auto",
        random_state: int = 42,
    ) -> pd.DataFrame:
        """
        Ajusta PCA sobre las variables escaladas y guarda los scores.

        Args:
            n_components: Número de componentes, fracción de varianza o None (todos).
            svd_solver: Solver de sklearn ('auto', 'full', 'randomized', 'arpack').
            random_state: Semilla de aleatoriedad.
        """
        self.pca_model = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state,
        )
        self.pca_scores = self.pca_model.fit_transform(self.X_scaled)

        columnas = [f"PC{i + 1}" for i in range(self.pca_scores.shape[1])]
        df_pca = pd.DataFrame(self.pca_scores, columns=columnas)
        self._guardar_tabla(df_pca, "pca_scores.csv", index=False)
        return df_pca

    @_requiere_escalado
    def comparar_pca_configuraciones(self) -> pd.DataFrame:
        """Compara distintas configuraciones de PCA y retorna métricas de varianza explicada."""
        configuraciones = [
            {"modelo": "pca_standard",      "n_components": None, "svd_solver": "auto"},
            {"modelo": "pca_2_componentes", "n_components": 2,    "svd_solver": "auto"},
            {"modelo": "pca_90_varianza",   "n_components": 0.90, "svd_solver": "full"},
            {"modelo": "pca_randomized_2",  "n_components": 2,    "svd_solver": "randomized"},
        ]

        resultados = [
            {
                "modelo": cfg["modelo"],
                "n_componentes": len(
                    (m := PCA(n_components=cfg["n_components"], svd_solver=cfg["svd_solver"], random_state=42).fit(self.X_scaled)).explained_variance_ratio_
                ),
                "varianza_explicada_total": m.explained_variance_ratio_.sum(),
                "svd_solver": cfg["svd_solver"],
            }
            for cfg in configuraciones
        ]

        tabla = pd.DataFrame(resultados)
        self._guardar_tabla(tabla, "comparacion_pca.csv", index=False)
        return tabla

    @_requiere_pca
    def scree_plot(self, guardar: bool = True) -> plt.Figure:
        """Grafica la varianza explicada por cada componente principal (Scree Plot)."""
        varianza = self.pca_model.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(varianza) + 1), varianza, marker="o")
        ax.set(
            title="Scree Plot",
            xlabel="Componente principal",
            ylabel="Varianza explicada",
            xticks=range(1, len(varianza) + 1),
        )
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, "scree_plot.png")

        plt.close(fig)
        return fig

    @_requiere_pca
    def pca_loadings(self) -> pd.DataFrame:
        """Retorna y guarda la contribución (loading) de cada variable a cada componente."""
        columnas = [f"PC{i + 1}" for i in range(self.pca_model.components_.shape[0])]
        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            index=self.X.columns,
            columns=columnas,
        )
        self._guardar_tabla(loadings, "pca_loadings.csv")
        return loadings

    @_requiere_escalado
    def scatter_pca(self, color_by: str | None = None, guardar: bool = True) -> pd.DataFrame:
        """
        Proyecta los datos en 2 componentes principales y genera un scatter.

        Args:
            color_by: Columna a usar como color ('quality', otra columna o None).
        """
        coords_array = PCA(n_components=2, random_state=42).fit_transform(self.X_scaled)
        coords = pd.DataFrame(coords_array, columns=["PC1", "PC2"])

        color_col = color_by if color_by in (list(self.X.columns) + ["quality"]) else None
        if color_col == "quality":
            coords["quality"] = self.y.values
        elif color_col:
            coords[color_col] = self.df[color_col].values

        nombre = f"scatter_pca_{color_by or 'simple'}.png"
        _, df_plot = self._scatter_2d(coords, "PC1", "PC2", color_col, "Proyección PCA 2D", nombre, guardar)
        self._guardar_tabla(df_plot, "scatter_pca_2d.csv", index=False)
        return df_plot

    # -----------------------------------------------------------------------
    # KMeans
    # -----------------------------------------------------------------------

    @_requiere_escalado
    def evaluar_kmeans(self, k_min: int = 2, k_max: int = 6, n_init: int = 10) -> pd.DataFrame:
        """Evalúa KMeans para un rango de k y retorna inercia y silhouette por k."""
        tabla = self._evaluar_kmeans_rango(k_min, k_max, n_init)
        self._guardar_tabla(tabla, "evaluacion_kmeans.csv", index=False)
        return tabla

    @_requiere_escalado
    def grafico_elbow(self, k_min: int = 2, k_max: int = 6, guardar: bool = True) -> plt.Figure:
        """Grafica la inercia vs. k para identificar el codo óptimo."""
        tabla = self._evaluar_kmeans_rango(k_min, k_max, n_init=10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tabla["k"], tabla["inertia"], marker="o")
        ax.set(title="Método del codo", xlabel="Número de clusters (k)", ylabel="Inercia")
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, "kmeans_elbow.png")

        plt.close(fig)
        return fig

    @_requiere_escalado
    def grafico_silhouette_kmeans(self, k_min: int = 2, k_max: int = 6, guardar: bool = True) -> plt.Figure:
        """Grafica el silhouette score vs. k para comparar configuraciones de KMeans."""
        tabla = self._evaluar_kmeans_rango(k_min, k_max, n_init=10)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tabla["k"], tabla["silhouette_score"], marker="o")
        ax.set(title="Silhouette Score por k", xlabel="Número de clusters (k)", ylabel="Silhouette score")
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, "kmeans_silhouette.png")

        plt.close(fig)
        return fig

    @_requiere_escalado
    def ejecutar_kmeans(self, k: int = 3, n_init: int = 10) -> pd.DataFrame:
        """
        Ejecuta KMeans con el k indicado y agrega las etiquetas al DataFrame original.

        Returns:
            DataFrame con columna 'cluster_kmeans' añadida.
        """
        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = self.kmeans_model.fit_predict(self.X_scaled)

        resultado = self.df.assign(cluster_kmeans=labels)
        self._guardar_tabla(resultado, f"kmeans_clusters_k_{k}.csv", index=False)
        return resultado

    @_requiere_escalado
    def comparar_kmeans_configuraciones(self) -> pd.DataFrame:
        """Compara KMeans para k ∈ {2, 3, 4, 5} y retorna inercia y silhouette."""
        tabla = self._evaluar_kmeans_rango(k_min=2, k_max=5, n_init=10)
        tabla.insert(0, "modelo", tabla["k"].apply(lambda k: f"kmeans_k{k}"))
        self._guardar_tabla(tabla, "comparacion_kmeans.csv", index=False)
        return tabla

    @_requiere_escalado
    def scatter_clusters_pca(self, cluster_col: str, guardar: bool = True) -> pd.DataFrame:
        """
        Visualiza los clusters en el espacio de las 2 primeras componentes principales.

        Args:
            cluster_col: Nombre de la columna de clusters en self.df.

        Raises:
            ValueError: Si la columna no existe en self.df.
        """
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna '{cluster_col}' no existe en self.df.")

        coords_array = PCA(n_components=2, random_state=42).fit_transform(self.X_scaled)
        coords = pd.DataFrame(coords_array, columns=["PC1", "PC2"])
        coords[cluster_col] = self.df[cluster_col].values

        nombre = f"scatter_pca_{cluster_col}.png"
        titulo = f"Clusters en espacio PCA — {cluster_col}"
        _, df_plot = self._scatter_2d(coords, "PC1", "PC2", cluster_col, titulo, nombre, guardar)
        self._guardar_tabla(df_plot, f"scatter_pca_{cluster_col}.csv", index=False)
        return df_plot

    def resumen_clusters(self, cluster_col: str) -> pd.DataFrame:
        """
        Calcula el promedio de cada variable por cluster.

        Args:
            cluster_col: Columna de etiquetas de cluster en self.df.
        """
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna '{cluster_col}' no existe en self.df.")

        resumen = self.df.groupby(cluster_col).mean(numeric_only=True)
        self._guardar_tabla(resumen, f"resumen_{cluster_col}.csv")
        return resumen


    # -----------------------------------------------------------------------
    # HAC (Clustering Jerárquico Aglomerativo)
    # -----------------------------------------------------------------------

    @_requiere_escalado
    def ejecutar_hac(self, n_clusters: int = 3, linkage_method: str = "average") -> pd.DataFrame:
        """
        Ejecuta clustering jerárquico aglomerativo y agrega las etiquetas al DataFrame.

        Args:
            n_clusters: Número de clusters.
            linkage_method: Método de enlace ('ward', 'complete', 'average', 'single').
        """
        self.hac_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method,
        )
        labels = self.hac_model.fit_predict(self.X_scaled)

        resultado = self.df.assign(cluster_hac=labels)
        self._guardar_tabla(
            resultado,
            f"hac_clusters_{linkage_method}_k_{n_clusters}.csv",
            index=False,
        )
        return resultado

    @_requiere_escalado
    def comparar_hac_configuraciones(self, n_clusters: int = 3) -> pd.DataFrame:
        """Compara HAC con métodos de enlace 'ward', 'complete' y 'average'."""
        metodos = ["ward", "complete", "average"]

        resultados = [
            {
                "modelo": f"hac_{metodo}",
                "linkage": metodo,
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_score(
                    self.X_scaled,
                    AgglomerativeClustering(n_clusters=n_clusters, linkage=metodo).fit_predict(self.X_scaled),
                ),
            }
            for metodo in metodos
        ]

        tabla = pd.DataFrame(resultados)
        self._guardar_tabla(tabla, "comparacion_hac.csv", index=False)
        return tabla

    @_requiere_escalado
    def graficar_dendrograma(self, linkage_method: str = "ward", guardar: bool = True) -> plt.Figure:
        """Grafica el dendrograma del clustering jerárquico."""
        z = linkage(self.X_scaled, method=linkage_method)

        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(z, truncate_mode="level", p=5, ax=ax)
        ax.set(
            title=f"Dendrograma ({linkage_method})",
            xlabel="Observaciones",
            ylabel="Distancia",
        )
        fig.tight_layout()

        if guardar:
            self._guardar_figura(fig, f"dendrograma_{linkage_method}.png")

        plt.close(fig)
        return fig

    @_requiere_escalado
    def graficar_pairplot_hac(
        self,
        n_clusters: int = 3,
        linkage_method: str = "average",
        guardar: bool = True,
    ) -> sns.PairGrid:
        """Visualiza los clusters HAC en todas las combinaciones de variables (pairplot)."""
        labels = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage_method
        ).fit_predict(self.X_scaled)

        resultado = self.df.assign(cluster_hac=labels)
        grid = sns.pairplot(resultado, hue="cluster_hac", palette="viridis", diag_kind="kde")
        grid.fig.suptitle(f"Pairplot HAC ({linkage_method}) — k={n_clusters}", y=1.01)

        if guardar:
            self._guardar_figura(grid.fig, f"pairplot_hac_{linkage_method}_k_{n_clusters}.png")

        plt.close()
        return grid

    # -----------------------------------------------------------------------
    # t-SNE
    # -----------------------------------------------------------------------

    @_requiere_escalado
    def ejecutar_tsne(
        self,
        n_components: int = 2,
        perplexity: int = 30,
        learning_rate: Union[str, float] = "auto",
        max_iter: int = 1000,
    ) -> pd.DataFrame:
        """
        Ejecuta t-SNE y retorna las coordenadas reducidas junto con quality.

        Args:
            perplexity: Parámetro de vecindad (5–50 recomendado).
            learning_rate: Tasa de aprendizaje o 'auto'.
        """
        modelo = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42,
        )
        coords = modelo.fit_transform(self.X_scaled)

        df_tsne = pd.DataFrame(coords, columns=["TSNE1", "TSNE2"]).assign(quality=self.y.values)
        self._guardar_tabla(df_tsne, f"tsne_perplexity_{perplexity}.csv", index=False)
        return df_tsne

    @_requiere_escalado
    def comparar_tsne_configuraciones(self) -> pd.DataFrame:
        """Compara t-SNE con perplexity {5, 30, 50} usando silhouette sobre KMeans(k=3)."""
        resultados = [
            {
                "modelo": f"tsne_perplexity_{p}",
                "perplexity": p,
                "silhouette_score_2d": silhouette_score(
                    df_temp := pd.DataFrame(
                        TSNE(n_components=2, perplexity=p, learning_rate="auto", max_iter=1000, random_state=42)
                        .fit_transform(self.X_scaled),
                        columns=["TSNE1", "TSNE2"],
                    ),
                    KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(df_temp),
                ),
            }
            for p in [5, 30, 50]
        ]

        tabla = pd.DataFrame(resultados)
        self._guardar_tabla(tabla, "comparacion_tsne.csv", index=False)
        return tabla

    def graficar_tsne(
        self,
        df_tsne: pd.DataFrame,
        color_by: str = "quality",
        nombre_archivo: str = "tsne_plot.png",
        guardar: bool = True,
    ) -> plt.Figure:
        """Visualiza la proyección t-SNE coloreada por una columna del DataFrame."""
        fig, _ = self._scatter_2d(
            df_tsne, "TSNE1", "TSNE2",
            color_by if color_by in df_tsne.columns else None,
            "Proyección t-SNE", nombre_archivo, guardar,
        )
        return fig

    # -----------------------------------------------------------------------
    # UMAP
    # -----------------------------------------------------------------------

    @_requiere_escalado
    def ejecutar_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
    ) -> pd.DataFrame:
        """
        Ejecuta UMAP y retorna las coordenadas reducidas junto con quality.

        Args:
            n_neighbors: Tamaño del vecindario local (mayor = estructura más global).
            min_dist: Distancia mínima entre puntos en la proyección.
        """
        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        coords = modelo.fit_transform(self.X_scaled)

        df_umap = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"]).assign(quality=self.y.values)
        self._guardar_tabla(
            df_umap,
            f"umap_neighbors_{n_neighbors}_mindist_{min_dist}.csv",
            index=False,
        )
        return df_umap

    @_requiere_escalado
    def comparar_umap_configuraciones(self) -> pd.DataFrame:
        """Compara UMAP con 3 combinaciones de n_neighbors y min_dist usando silhouette."""
        configuraciones = [
            {"n_neighbors": 10, "min_dist": 0.1},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.3},
        ]

        resultados = [
            {
                "modelo": f"umap_n{cfg['n_neighbors']}_d{cfg['min_dist']}",
                "n_neighbors": cfg["n_neighbors"],
                "min_dist": cfg["min_dist"],
                "silhouette_score_2d": silhouette_score(
                    df_temp := pd.DataFrame(
                        umap.UMAP(n_components=2, random_state=42, **cfg).fit_transform(self.X_scaled),
                        columns=["UMAP1", "UMAP2"],
                    ),
                    KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(df_temp),
                ),
            }
            for cfg in configuraciones
        ]

        tabla = pd.DataFrame(resultados)
        self._guardar_tabla(tabla, "comparacion_umap.csv", index=False)
        return tabla

    def graficar_umap(
        self,
        df_umap: pd.DataFrame,
        color_by: str = "quality",
        nombre_archivo: str = "umap_plot.png",
        guardar: bool = True,
    ) -> plt.Figure:
        """Visualiza la proyección UMAP coloreada por una columna del DataFrame."""
        fig, _ = self._scatter_2d(
            df_umap, "UMAP1", "UMAP2",
            color_by if color_by in df_umap.columns else None,
            "Proyección UMAP", nombre_archivo, guardar,
        )
        return fig