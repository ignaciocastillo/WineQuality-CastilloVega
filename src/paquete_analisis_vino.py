from pathlib import Path

# Manejo de datos y gráficos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocesamiento y reducción de dimensionalidad
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Algoritmos de clustering
from sklearn.cluster import KMeans, AgglomerativeClustering

# Métodos no lineales y métricas
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

import umap.umap_ as umap


class AnalisisVino:
    def __init__(self, ruta_csv: str):
        # Construir rutas del proyecto y del dataset
        self.ruta_csv = ruta_csv
        self.project_root = Path(__file__).resolve().parents[1]
        self.csv_path = (self.project_root / ruta_csv).resolve()

        # Validar que el archivo exista
        if not self.csv_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en: {self.csv_path}")

        # Definir carpetas de salida
        self.fig_path = self.project_root / "outputs" / "figures"
        self.tab_path = self.project_root / "outputs" / "tables"

        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.tab_path.mkdir(parents=True, exist_ok=True)

        # Cargar dataset y limpiar nombres de columnas
        self.df = pd.read_csv(self.csv_path)
        self.df.columns = self.df.columns.str.strip()

        # Atributos que se llenan durante el análisis
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None

        self.pca_model = None
        self.pca_scores = None
        self.kmeans_model = None
        self.hac_model = None

    def resumen_datos(self) -> pd.DataFrame:
        # Crear resumen básico del dataset
        resumen = pd.DataFrame({
            "métrica": ["n_filas", "n_columnas", "duplicados"],
            "valor": [
                self.df.shape[0],
                self.df.shape[1],
                int(self.df.duplicated().sum())
            ]
        })

        resumen.to_csv(self.tab_path / "resumen_datos.csv", index=False)
        return resumen

    def columnas(self) -> pd.DataFrame:
        # Listar columnas del dataset
        tabla = pd.DataFrame({"columnas": self.df.columns})
        tabla.to_csv(self.tab_path / "columnas.csv", index=False)
        return tabla

    def nulos_por_columna(self) -> pd.DataFrame:
        # Contar valores nulos por variable
        tabla = self.df.isnull().sum().reset_index()
        tabla.columns = ["columna", "nulos"]
        tabla.to_csv(self.tab_path / "nulos_por_columna.csv", index=False)
        return tabla

    def estadisticas(self) -> pd.DataFrame:
        # Calcular estadísticas descriptivas
        tabla = self.df.describe().T
        tabla.to_csv(self.tab_path / "estadisticas_descriptivas.csv")
        return tabla

    def separar_quality(self) -> None:
        # Separar quality para los métodos no supervisados
        self.y = self.df["quality"].copy()
        self.X = self.df.drop(columns=["quality"]).copy()
        print("separar_quality ejecutado correctamente — X:", self.X.shape, "| y:", self.y.shape)

    def escalar_variables(self) -> pd.DataFrame:
        # Verificar que las variables ya estén separadas
        if self.X is None:
            raise ValueError("Primero debe ejecutar separar_quality().")

        # Estandarizar variables para PCA y clustering
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        tabla = pd.DataFrame(self.X_scaled, columns=self.X.columns)
        tabla.to_csv(self.tab_path / "variables_escaladas.csv", index=False)
        return tabla

    def distribucion_quality(self, guardar: bool = True):
        # Graficar la distribución de la variable quality
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=self.df["quality"], ax=ax)
        ax.set_title("Distribución de la variable quality")
        ax.set_xlabel("Quality")
        ax.set_ylabel("Frecuencia")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / "distribucion_quality.png", dpi=300)

        plt.close(fig)
        return fig

    def matriz_correlacion(self, guardar: bool = True) -> pd.DataFrame:
        # Calcular y visualizar la matriz de correlación
        corr = self.df.corr(numeric_only=True)
        corr.to_csv(self.tab_path / "matriz_correlacion.csv")

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        ax.set_title("Matriz de correlación")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / "matriz_correlacion.png", dpi=300)

        self._ultima_figura = fig
        return corr

    def obtener_ultima_figura(self):
        # Recuperar la última figura generada
        return getattr(self, "_ultima_figura", None)

    def ejecutar_pca(self, n_components=None, svd_solver="auto", random_state: int = 42) -> pd.DataFrame:
        # Ejecutar PCA sobre las variables escaladas
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        self.pca_model = PCA(
            n_components=n_components,
            svd_solver=svd_solver,
            random_state=random_state
        )
        self.pca_scores = self.pca_model.fit_transform(self.X_scaled)

        columnas = [f"PC{i + 1}" for i in range(self.pca_scores.shape[1])]
        df_pca = pd.DataFrame(self.pca_scores, columns=columnas)
        df_pca.to_csv(self.tab_path / "pca_scores.csv", index=False)

        return df_pca

    def comparar_pca_configuraciones(self) -> pd.DataFrame:
        # Comparar distintas configuraciones de PCA
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        configuraciones = [
            {"modelo": "pca_standard", "n_components": None, "svd_solver": "auto"},
            {"modelo": "pca_2_componentes", "n_components": 2, "svd_solver": "auto"},
            {"modelo": "pca_90_varianza", "n_components": 0.90, "svd_solver": "full"},
            {"modelo": "pca_randomized_2", "n_components": 2, "svd_solver": "randomized"},
        ]

        resultados = []

        for cfg in configuraciones:
            modelo = PCA(
                n_components=cfg["n_components"],
                svd_solver=cfg["svd_solver"],
                random_state=42
            )
            modelo.fit(self.X_scaled)

            resultados.append({
                "modelo": cfg["modelo"],
                "n_componentes": len(modelo.explained_variance_ratio_),
                "varianza_explicada_total": modelo.explained_variance_ratio_.sum(),
                "svd_solver": cfg["svd_solver"]
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_pca.csv", index=False)
        return tabla

    def scree_plot(self, guardar: bool = True):
        # Graficar la varianza explicada por componente
        if self.pca_model is None:
            raise ValueError("Primero debe ejecutar ejecutar_pca().")

        varianza = self.pca_model.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(varianza) + 1), varianza, marker="o")
        ax.set_title("Scree Plot")
        ax.set_xlabel("Componente principal")
        ax.set_ylabel("Varianza explicada")
        ax.set_xticks(range(1, len(varianza) + 1))
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / "scree_plot.png", dpi=300)

        plt.close(fig)
        return fig

    def pca_loadings(self) -> pd.DataFrame:
        # Calcular la contribución de cada variable a los componentes
        if self.pca_model is None:
            raise ValueError("Primero debe ejecutar ejecutar_pca().")

        columnas = [f"PC{i + 1}" for i in range(self.pca_model.components_.shape[0])]

        loadings = pd.DataFrame(
            self.pca_model.components_.T,
            index=self.X.columns,
            columns=columnas
        )

        loadings.to_csv(self.tab_path / "pca_loadings.csv")
        return loadings

    def scatter_pca(self, color_by: str = None, guardar: bool = True) -> pd.DataFrame:
        # Proyectar los datos en dos componentes principales
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        modelo = PCA(n_components=2, random_state=42)
        coords = modelo.fit_transform(self.X_scaled)

        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])

        fig, ax = plt.subplots(figsize=(8, 6))

        if color_by == "quality":
            df_plot["color"] = self.y.values
            scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["color"])
            fig.colorbar(scatter)
        elif color_by in self.df.columns:
            df_plot["color"] = self.df[color_by].values
            scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["color"])
            fig.colorbar(scatter)
        else:
            ax.scatter(df_plot["PC1"], df_plot["PC2"])

        ax.set_title("Proyección PCA 2D")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()

        if guardar:
            nombre = f"scatter_pca_{color_by if color_by else 'simple'}.png"
            fig.savefig(self.fig_path / nombre, dpi=300)

        df_plot.to_csv(self.tab_path / "scatter_pca_2d.csv", index=False)
        self._ultima_figura = fig
        return df_plot

    def evaluar_kmeans(self, k_min: int = 2, k_max: int = 6, n_init: int = 10) -> pd.DataFrame:
        # Evaluar distintos valores de k con KMeans
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        resultados = []

        for k in range(k_min, k_max + 1):
            modelo = KMeans(n_clusters=k, random_state=42, n_init=n_init)
            labels = modelo.fit_predict(self.X_scaled)

            resultados.append({
                "k": k,
                "inertia": modelo.inertia_,
                "silhouette_score": silhouette_score(self.X_scaled, labels)
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "evaluacion_kmeans.csv", index=False)
        return tabla

    def grafico_elbow(self, k_min: int = 2, k_max: int = 6, guardar: bool = True):
        # Graficar la inercia para elegir k
        tabla = self.evaluar_kmeans(k_min=k_min, k_max=k_max)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tabla["k"], tabla["inertia"], marker="o")
        ax.set_title("Método del codo")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Inercia")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / "kmeans_elbow.png", dpi=300)

        plt.close(fig)
        return fig

    def grafico_silhouette_kmeans(self, k_min: int = 2, k_max: int = 6, guardar: bool = True):
        # Graficar silhouette score para comparar k
        tabla = self.evaluar_kmeans(k_min=k_min, k_max=k_max)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(tabla["k"], tabla["silhouette_score"], marker="o")
        ax.set_title("Silhouette Score por k")
        ax.set_xlabel("Número de clusters (k)")
        ax.set_ylabel("Silhouette score")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / "kmeans_silhouette.png", dpi=300)

        plt.close(fig)
        return fig

    def ejecutar_kmeans(self, k: int = 3, n_init: int = 10) -> pd.DataFrame:
        # Ejecutar clustering KMeans con el k seleccionado
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = self.kmeans_model.fit_predict(self.X_scaled)

        resultado = self.df.copy()
        resultado["cluster_kmeans"] = labels
        resultado.to_csv(self.tab_path / f"kmeans_clusters_k_{k}.csv", index=False)
        return resultado

    def comparar_kmeans_configuraciones(self) -> pd.DataFrame:
        # Comparar varias configuraciones de KMeans
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        configuraciones = [
            {"modelo": "kmeans_k2", "k": 2, "n_init": 10},
            {"modelo": "kmeans_k3", "k": 3, "n_init": 10},
            {"modelo": "kmeans_k4", "k": 4, "n_init": 10},
            {"modelo": "kmeans_k5", "k": 5, "n_init": 10},
        ]

        resultados = []

        for cfg in configuraciones:
            modelo = KMeans(
                n_clusters=cfg["k"],
                random_state=42,
                n_init=cfg["n_init"]
            )
            labels = modelo.fit_predict(self.X_scaled)

            resultados.append({
                "modelo": cfg["modelo"],
                "k": cfg["k"],
                "inertia": modelo.inertia_,
                "silhouette_score": silhouette_score(self.X_scaled, labels)
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_kmeans.csv", index=False)
        return tabla

    def scatter_clusters_pca(self, cluster_col: str, guardar: bool = True) -> pd.DataFrame:
        # Visualizar clusters en el espacio PCA
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna {cluster_col} no existe en self.df.")

        modelo = PCA(n_components=2, random_state=42)
        coords = modelo.fit_transform(self.X_scaled)

        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
        df_plot[cluster_col] = self.df[cluster_col].values

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot[cluster_col])
        fig.colorbar(scatter)
        ax.set_title(f"Clusters en espacio PCA - {cluster_col}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / f"scatter_pca_{cluster_col}.png", dpi=300)

        df_plot.to_csv(self.tab_path / f"scatter_pca_{cluster_col}.csv", index=False)
        self._ultima_figura = fig
        return df_plot

    def resumen_clusters(self, cluster_col: str) -> pd.DataFrame:
        # Resumir promedios por cluster
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna {cluster_col} no existe en self.df.")

        resumen = self.df.groupby(cluster_col).mean(numeric_only=True)
        resumen.to_csv(self.tab_path / f"resumen_{cluster_col}.csv")
        return resumen

    def ejecutar_hac(self, n_clusters: int = 3, linkage_method: str = "ward") -> pd.DataFrame:
        # Ejecutar clustering jerárquico aglomerativo
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        self.hac_model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage_method
        )
        labels = self.hac_model.fit_predict(self.X_scaled)

        resultado = self.df.copy()
        resultado["cluster_hac"] = labels
        resultado.to_csv(
            self.tab_path / f"hac_clusters_{linkage_method}_k_{n_clusters}.csv",
            index=False
        )
        return resultado

    def comparar_hac_configuraciones(self, n_clusters: int = 3) -> pd.DataFrame:
        # Comparar distintos métodos de linkage
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        metodos = ["ward", "complete", "average"]
        resultados = []

        for metodo in metodos:
            modelo = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=metodo
            )
            labels = modelo.fit_predict(self.X_scaled)

            resultados.append({
                "modelo": f"hac_{metodo}",
                "linkage": metodo,
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_score(self.X_scaled, labels)
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_hac.csv", index=False)
        return tabla

    def graficar_dendrograma(self, linkage_method: str = "ward", guardar: bool = True):
        # Graficar la estructura jerárquica de los datos
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        z = linkage(self.X_scaled, method=linkage_method)

        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(z, truncate_mode="level", p=5, ax=ax)
        ax.set_title(f"Dendrograma ({linkage_method})")
        ax.set_xlabel("Observaciones")
        ax.set_ylabel("Distancia")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / f"dendrograma_{linkage_method}.png", dpi=300)

        plt.close(fig)
        return fig

    def ejecutar_tsne(
        self,
        n_components: int = 2,
        perplexity: int = 30,
        learning_rate: str = "auto",
        max_iter: int = 1000
    ) -> pd.DataFrame:
        # Ejecutar t-SNE con la configuración indicada
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        modelo = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42
        )

        coords = modelo.fit_transform(self.X_scaled)

        df_tsne = pd.DataFrame(coords, columns=["TSNE1", "TSNE2"])
        df_tsne["quality"] = self.y.values
        df_tsne.to_csv(self.tab_path / f"tsne_perplexity_{perplexity}.csv", index=False)
        return df_tsne

    def comparar_tsne_configuraciones(self) -> pd.DataFrame:
        # Comparar varias configuraciones de t-SNE
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        configuraciones = [5, 30, 50]
        resultados = []

        for perplexity in configuraciones:
            modelo = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                max_iter=1000,
                random_state=42
            )
            coords = modelo.fit_transform(self.X_scaled)

            df_temp = pd.DataFrame(coords, columns=["TSNE1", "TSNE2"])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_temp)

            resultados.append({
                "modelo": f"tsne_perplexity_{perplexity}",
                "perplexity": perplexity,
                "silhouette_score_2d": silhouette_score(df_temp, labels)
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_tsne.csv", index=False)
        return tabla

    def graficar_tsne(
        self,
        df_tsne: pd.DataFrame,
        color_by: str = "quality",
        nombre_archivo: str = "tsne_plot.png",
        guardar: bool = True
    ):
        # Visualizar la proyección t-SNE
        fig, ax = plt.subplots(figsize=(8, 6))

        if color_by in df_tsne.columns:
            scatter = ax.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"], c=df_tsne[color_by])
            fig.colorbar(scatter)
        else:
            ax.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"])

        ax.set_title("Proyección t-SNE")
        ax.set_xlabel("TSNE1")
        ax.set_ylabel("TSNE2")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / nombre_archivo, dpi=300)

        plt.close(fig)
        return fig

    def ejecutar_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> pd.DataFrame:
        # Ejecutar UMAP con la configuración indicada
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        modelo = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )

        coords = modelo.fit_transform(self.X_scaled)

        df_umap = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
        df_umap["quality"] = self.y.values
        df_umap.to_csv(
            self.tab_path / f"umap_neighbors_{n_neighbors}_mindist_{min_dist}.csv",
            index=False
        )
        return df_umap

    def comparar_umap_configuraciones(self) -> pd.DataFrame:
        # Comparar varias configuraciones de UMAP
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        configuraciones = [
            {"n_neighbors": 10, "min_dist": 0.1},
            {"n_neighbors": 15, "min_dist": 0.1},
            {"n_neighbors": 30, "min_dist": 0.3},
        ]

        resultados = []

        for cfg in configuraciones:
            modelo = umap.UMAP(
                n_components=2,
                n_neighbors=cfg["n_neighbors"],
                min_dist=cfg["min_dist"],
                random_state=42
            )
            coords = modelo.fit_transform(self.X_scaled)

            df_temp = pd.DataFrame(coords, columns=["UMAP1", "UMAP2"])
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels = kmeans.fit_predict(df_temp)

            resultados.append({
                "modelo": f"umap_n{cfg['n_neighbors']}_d{cfg['min_dist']}",
                "n_neighbors": cfg["n_neighbors"],
                "min_dist": cfg["min_dist"],
                "silhouette_score_2d": silhouette_score(df_temp, labels)
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_umap.csv", index=False)
        return tabla

    def graficar_umap(
        self,
        df_umap: pd.DataFrame,
        color_by: str = "quality",
        nombre_archivo: str = "umap_plot.png",
        guardar: bool = True
    ):
        # Visualizar la proyección UMAP
        fig, ax = plt.subplots(figsize=(8, 6))

        if color_by in df_umap.columns:
            scatter = ax.scatter(df_umap["UMAP1"], df_umap["UMAP2"], c=df_umap[color_by])
            fig.colorbar(scatter)
        else:
            ax.scatter(df_umap["UMAP1"], df_umap["UMAP2"])

        ax.set_title("Proyección UMAP")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        fig.tight_layout()

        if guardar:
            fig.savefig(self.fig_path / nombre_archivo, dpi=300)

        plt.close(fig)
        return fig