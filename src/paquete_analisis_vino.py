from pathlib import Path

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


class AnalisisVino:
    def __init__(self, ruta_csv: str):
        # Guardar ruta original
        self.ruta_csv = ruta_csv

        # Detectar raíz del proyecto
        self.project_root = Path(__file__).resolve().parents[1]

        # Construir ruta absoluta del csv
        self.csv_path = (self.project_root / ruta_csv).resolve()

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"No se encontró el archivo en: {self.csv_path}"
            )

        # Rutas de salida
        self.fig_path = self.project_root / "outputs" / "figures"
        self.tab_path = self.project_root / "outputs" / "tables"

        self.fig_path.mkdir(parents=True, exist_ok=True)
        self.tab_path.mkdir(parents=True, exist_ok=True)

        # Cargar dataset
        self.df = pd.read_csv(self.csv_path, sep=",")

        # Atributos de trabajo
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = None

        self.pca_model = None
        self.pca_scores = None

        self.kmeans_model = None
        self.hac_model = None

    def resumen_datos(self) -> dict:
        # Resumen general
        resumen = {
            "n_filas": self.df.shape[0],
            "n_columnas": self.df.shape[1],
            "columnas": list(self.df.columns),
            "nulos_por_columna": self.df.isnull().sum().to_dict(),
            "duplicados": int(self.df.duplicated().sum())
        }
        return resumen

    def estadisticas(self) -> pd.DataFrame:
        # Estadísticas descriptivas
        tabla = self.df.describe().T
        tabla.to_csv(self.tab_path / "estadisticas_descriptivas.csv")
        return tabla

    def separar_quality(self) -> None:
        # Separar quality
        self.y = self.df["quality"].copy()
        self.X = self.df.drop(columns=["quality"]).copy()

    def escalar_variables(self) -> None:
        # Escalar variables
        if self.X is None:
            raise ValueError("Primero debe ejecutar separar_quality().")

        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

    def distribucion_quality(self, guardar: bool = True) -> None:
        # Graficar distribución de quality
        plt.figure(figsize=(8, 5))
        sns.countplot(x=self.df["quality"])
        plt.title("Distribución de la variable quality")
        plt.xlabel("Quality")
        plt.ylabel("Frecuencia")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / "distribucion_quality.png", dpi=300)

        plt.show()
        plt.close()

    def matriz_correlacion(self, guardar: bool = True) -> pd.DataFrame:
        # Calcular matriz de correlación
        corr = self.df.corr(numeric_only=True)

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="coolwarm", annot=False)
        plt.title("Matriz de correlación")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / "matriz_correlacion.png", dpi=300)

        plt.show()
        plt.close()

        corr.to_csv(self.tab_path / "matriz_correlacion.csv")
        return corr

    def ejecutar_pca(
        self,
        n_components=None,
        svd_solver="auto",
        random_state: int = 42
    ) -> pd.DataFrame:
        # Ejecutar PCA
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

        return df_pca

    def comparar_pca_configuraciones(self) -> pd.DataFrame:
        # Comparar configuraciones de PCA
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        configuraciones = [
            {"nombre": "pca_standard", "n_components": None, "svd_solver": "auto"},
            {"nombre": "pca_2_componentes", "n_components": 2, "svd_solver": "auto"},
            {"nombre": "pca_90_varianza", "n_components": 0.90, "svd_solver": "full"},
            {"nombre": "pca_randomized_2", "n_components": 2, "svd_solver": "randomized"},
        ]

        resultados = []

        for cfg in configuraciones:
            modelo = PCA(
                n_components=cfg["n_components"],
                svd_solver=cfg["svd_solver"],
                random_state=42
            )
            modelo.fit(self.X_scaled)

            if hasattr(modelo, "explained_variance_ratio_"):
                varianza_total = modelo.explained_variance_ratio_.sum()
                n_comp = len(modelo.explained_variance_ratio_)
            else:
                varianza_total = None
                n_comp = None

            resultados.append({
                "modelo": cfg["nombre"],
                "n_components": n_comp,
                "varianza_explicada_total": varianza_total,
                "svd_solver": cfg["svd_solver"]
            })

        tabla = pd.DataFrame(resultados)
        tabla.to_csv(self.tab_path / "comparacion_pca.csv", index=False)
        return tabla

    def scree_plot(self, guardar: bool = True) -> None:
        # Graficar scree plot
        if self.pca_model is None:
            raise ValueError("Primero debe ejecutar ejecutar_pca().")

        varianza = self.pca_model.explained_variance_ratio_

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(varianza) + 1), varianza, marker="o")
        plt.title("Scree Plot")
        plt.xlabel("Componente principal")
        plt.ylabel("Varianza explicada")
        plt.xticks(range(1, len(varianza) + 1))
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / "scree_plot.png", dpi=300)

        plt.show()
        plt.close()

    def pca_loadings(self) -> pd.DataFrame:
        # Obtener loadings de PCA
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

    def scatter_pca(
        self,
        color_by: str = None,
        guardar: bool = True
    ) -> pd.DataFrame:
        # Graficar proyección PCA en 2D
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        modelo = PCA(n_components=2, random_state=42)
        coords = modelo.fit_transform(self.X_scaled)

        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])

        if color_by == "quality":
            df_plot["color"] = self.y.values
        elif color_by in self.df.columns:
            df_plot["color"] = self.df[color_by].values

        plt.figure(figsize=(8, 6))

        if "color" in df_plot.columns:
            plt.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot["color"])
            plt.colorbar()
        else:
            plt.scatter(df_plot["PC1"], df_plot["PC2"])

        plt.title("Proyección PCA 2D")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        if guardar:
            nombre = f"scatter_pca_{color_by if color_by else 'simple'}.png"
            plt.savefig(self.fig_path / nombre, dpi=300)

        plt.show()
        plt.close()

        df_plot.to_csv(self.tab_path / "scatter_pca_2d.csv", index=False)
        return df_plot

    def evaluar_kmeans(
        self,
        k_min: int = 2,
        k_max: int = 6,
        n_init: int = 10
    ) -> pd.DataFrame:
        # Evaluar KMeans con varios valores de k
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

    def grafico_elbow(self, k_min: int = 2, k_max: int = 6, guardar: bool = True) -> None:
        # Graficar método del codo
        tabla = self.evaluar_kmeans(k_min=k_min, k_max=k_max)

        plt.figure(figsize=(8, 5))
        plt.plot(tabla["k"], tabla["inertia"], marker="o")
        plt.title("Método del codo")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Inercia")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / "kmeans_elbow.png", dpi=300)

        plt.show()
        plt.close()

    def grafico_silhouette_kmeans(
        self,
        k_min: int = 2,
        k_max: int = 6,
        guardar: bool = True
    ) -> None:
        # Graficar silhouette de KMeans
        tabla = self.evaluar_kmeans(k_min=k_min, k_max=k_max)

        plt.figure(figsize=(8, 5))
        plt.plot(tabla["k"], tabla["silhouette_score"], marker="o")
        plt.title("Silhouette Score por k")
        plt.xlabel("Número de clusters (k)")
        plt.ylabel("Silhouette score")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / "kmeans_silhouette.png", dpi=300)

        plt.show()
        plt.close()

    def ejecutar_kmeans(self, k: int = 3, n_init: int = 10) -> pd.DataFrame:
        # Ejecutar KMeans
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        self.kmeans_model = KMeans(n_clusters=k, random_state=42, n_init=n_init)
        labels = self.kmeans_model.fit_predict(self.X_scaled)

        resultado = self.df.copy()
        resultado["cluster_kmeans"] = labels
        resultado.to_csv(self.tab_path / f"kmeans_clusters_k_{k}.csv", index=False)

        return resultado

    def comparar_kmeans_configuraciones(self) -> pd.DataFrame:
        # Comparar configuraciones de KMeans
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

    def scatter_clusters_pca(
        self,
        cluster_col: str,
        guardar: bool = True
    ) -> pd.DataFrame:
        # Graficar clusters en espacio PCA
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna {cluster_col} no existe en self.df.")

        modelo = PCA(n_components=2, random_state=42)
        coords = modelo.fit_transform(self.X_scaled)

        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
        df_plot[cluster_col] = self.df[cluster_col].values

        plt.figure(figsize=(8, 6))
        plt.scatter(df_plot["PC1"], df_plot["PC2"], c=df_plot[cluster_col])
        plt.colorbar()
        plt.title(f"Clusters en espacio PCA - {cluster_col}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / f"scatter_pca_{cluster_col}.png", dpi=300)

        plt.show()
        plt.close()

        df_plot.to_csv(self.tab_path / f"scatter_pca_{cluster_col}.csv", index=False)
        return df_plot

    def resumen_clusters(self, cluster_col: str) -> pd.DataFrame:
        # Resumen por cluster
        if cluster_col not in self.df.columns:
            raise ValueError(f"La columna {cluster_col} no existe en self.df.")

        resumen = self.df.groupby(cluster_col).mean(numeric_only=True)
        resumen.to_csv(self.tab_path / f"resumen_{cluster_col}.csv")
        return resumen

    def ejecutar_hac(
        self,
        n_clusters: int = 3,
        linkage_method: str = "ward"
    ) -> pd.DataFrame:
        # Ejecutar HAC
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
        # Comparar configuraciones de HAC
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

    def graficar_dendrograma(
        self,
        linkage_method: str = "ward",
        guardar: bool = True
    ) -> None:
        # Graficar dendrograma
        if self.X_scaled is None:
            raise ValueError("Primero debe ejecutar escalar_variables().")

        z = linkage(self.X_scaled, method=linkage_method)

        plt.figure(figsize=(12, 6))
        dendrogram(z, truncate_mode="level", p=5)
        plt.title(f"Dendrograma ({linkage_method})")
        plt.xlabel("Observaciones")
        plt.ylabel("Distancia")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / f"dendrograma_{linkage_method}.png", dpi=300)

        plt.show()
        plt.close()

    def ejecutar_tsne(
        self,
        n_components: int = 2,
        perplexity: int = 30,
        learning_rate: str = "auto",
        max_iter: int = 1000
    ) -> pd.DataFrame:
        # Ejecutar t-SNE
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
        df_tsne.to_csv(
            self.tab_path / f"tsne_perplexity_{perplexity}.csv",
            index=False
        )

        return df_tsne

    def comparar_tsne_configuraciones(self) -> pd.DataFrame:
        # Comparar configuraciones de t-SNE
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
    ) -> None:
        # Graficar t-SNE
        plt.figure(figsize=(8, 6))

        if color_by in df_tsne.columns:
            plt.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"], c=df_tsne[color_by])
            plt.colorbar()
        else:
            plt.scatter(df_tsne["TSNE1"], df_tsne["TSNE2"])

        plt.title("Proyección t-SNE")
        plt.xlabel("TSNE1")
        plt.ylabel("TSNE2")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / nombre_archivo, dpi=300)

        plt.show()
        plt.close()

    def ejecutar_umap(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1
    ) -> pd.DataFrame:
        # Ejecutar UMAP
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
        # Comparar configuraciones de UMAP
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
    ) -> None:
        # Graficar UMAP
        plt.figure(figsize=(8, 6))

        if color_by in df_umap.columns:
            plt.scatter(df_umap["UMAP1"], df_umap["UMAP2"], c=df_umap[color_by])
            plt.colorbar()
        else:
            plt.scatter(df_umap["UMAP1"], df_umap["UMAP2"])

        plt.title("Proyección UMAP")
        plt.xlabel("UMAP1")
        plt.ylabel("UMAP2")
        plt.tight_layout()

        if guardar:
            plt.savefig(self.fig_path / nombre_archivo, dpi=300)

        plt.show()
        plt.close()