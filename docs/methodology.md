# Metodología

## 1. Objetivo

El objetivo de este proyecto es comparar la calidad geométrica de dos métodos de reconstrucción de superficies basados en Gaussian Splatting: **Quadratic Gaussian Splatting (QGS)** y **VCR-GauS**.

Ambos métodos buscan reconstruir geometría tridimensional a partir de imágenes tomadas desde múltiples puntos de vista, pero utilizan estrategias diferentes para representar y optimizar las superficies.

La comparación se realiza sobre el conjunto de datos DTU, utilizando dos escenas (`scan24` y `scan37`) y métricas geométricas basadas en distancias entre las superficies reconstruidas y la geometría de referencia.

## 2. Métodos de reconstrucción

### 2.1. Quadratic Gaussian Splatting (QGS)

QGS propone una representación geométrica basada en primitivas cuadráticas deformables, como elipses y paraboloides, en lugar de utilizar únicamente primitivas planas.

Una característica importante del método es el uso de distancias geodésicas para modelar la distribución de densidad sobre las primitivas. A diferencia de la distancia euclidiana, la distancia geodésica considera la curvatura intrínseca de la superficie.

Esto permite adaptar la distribución de densidad a los cambios geométricos de cada primitiva y representar superficies curvas con mayor flexibilidad.

El método también utiliza intersecciones entre rayos y superficies cuadráticas para realizar el proceso de renderizado.

**Referencia:** Zhang et al., *Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives*, ICCV 2025.

### 2.2. VCR-GauS

VCR-GauS utiliza una representación basada en Gaussianas 3D y propone un regularizador de profundidad y normales (*Depth-Normal Regularizer*) para mejorar la reconstrucción geométrica.

El método aborda dos dificultades de los enfoques anteriores:

- La supervisión basada únicamente en normales puede limitar la actualización de los parámetros geométricos de las Gaussianas.
- Las inconsistencias entre mapas de normales predichos desde diferentes vistas pueden introducir artefactos en la reconstrucción.

Para abordar estos problemas, VCR-GauS incorpora un regularizador que relaciona las normales con otros parámetros geométricos de las Gaussianas. También utiliza un término de confianza para reducir el efecto de las inconsistencias entre vistas.

Adicionalmente, propone una estrategia de densificación y división de Gaussianas que regulariza su tamaño y distribución espacial.

**Referencia:** Chen et al., *VCR-GauS: View-Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction*, NeurIPS 2024.

## 3. Conjunto de datos

Se utiliza el conjunto de datos DTU (*Multi-View Stereo Evaluation*), desarrollado para evaluar algoritmos de reconstrucción tridimensional a partir de múltiples imágenes.

Para este proyecto se seleccionaron dos escenas (`scan24` y `scan37`), ambas con imágenes multivista, información de calibración y una reconstrucción de referencia obtenida mediante escaneo de luz estructurada.

Para evaluar la geometría se utilizan los siguientes archivos:

- `scan24`: `stl024_total.ply`, `ObsMask24_10.mat` y `Plane24.mat`.
- `scan37`: `stl037_total.ply`, `ObsMask37_10.mat` y `Plane37.mat`.

El experimento se limita a este escaneo y no representa una evaluación exhaustiva de todos los escenarios del conjunto DTU.

## 4. Procedimiento experimental

La comparación se organiza en cinco etapas:

### 4.1. Preparación de los datos

Se prepara cada escena de DTU junto con sus imágenes, parámetros de cámara y archivos de referencia. Los dos métodos utilizan la misma escena de entrada y configuraciones específicas compatibles con sus respectivos repositorios.

Para VCR-GauS, `load_normal` permanece activado y `load_depth` desactivado. En `scan37` se generaron los 49 mapas de normales `normal_npz_indoor` con GeoWizard, usando dominio `indoor`, `ensemble_size=3`, `denoise_steps=10`, `processing_res=768` y `seed=0`.

### 4.2. Entrenamiento

Se ejecuta QGS y VCR-GauS en entornos independientes para mantener la compatibilidad de sus versiones de Python, PyTorch y CUDA.

Cada método utiliza su propia representación y procedimiento de optimización. Los detalles de software y los parámetros experimentales se encuentran en `experimental_setup.md`.

### 4.3. Extracción de superficies

Una vez finalizado el entrenamiento, se obtiene una malla tridimensional de cada método.

QGS y VCR-GauS utilizan sus respectivos procedimientos de reconstrucción de malla. Los resultados se conservan en formato PLY para su posterior evaluación.

### 4.4. Filtrado mediante una máscara común

Para reducir las diferencias derivadas de las regiones no observadas, se aplica un procedimiento común de filtrado utilizando la máscara de observación de DTU.

Este procedimiento permite evaluar ambas reconstrucciones bajo las mismas condiciones espaciales.

Las mallas filtradas y los informes correspondientes se almacenan por escena, por ejemplo en `results/dtu/scan24/comparison/` y `results/dtu/scan37/comparison/`, cada una con subdirectorios `qgs/` y `vcr_gaus/`.

En `scan37`, el filtrado común procesó las 49 vistas, pero no eliminó vértices: QGS conservó 195 715 de 195 715 y VCR-GauS conservó 195 664 de 195 664. La evaluación de métricas aplica además la caja delimitadora y la máscara de observación DTU a los puntos muestreados.

### 4.5. Evaluación geométrica

Finalmente, se comparan las superficies reconstruidas con la nube de puntos de referencia.

Se utiliza el mismo procedimiento de evaluación y los mismos parámetros para ambos métodos. De esta manera, las diferencias observadas en las métricas corresponden a las reconstrucciones obtenidas bajo un protocolo común.

## 5. Métricas de evaluación

La calidad geométrica se evalúa mediante tres métricas basadas en distancias entre puntos.

Sea \(M\) el conjunto de puntos de la superficie reconstruida y \(G\) el conjunto de puntos del *ground truth*.

### 5.1. Accuracy (D2S)

Mide la distancia media desde los puntos de la reconstrucción hacia sus puntos más cercanos en el *ground truth*.

\[
\operatorname{Accuracy} =
\frac{1}{|M|}
\sum_{m \in M}
\min_{g \in G} \|m-g\|_2
\]

Un valor menor indica que la superficie reconstruida se encuentra más cerca de la geometría de referencia.

### 5.2. Completeness (S2D)

Mide la distancia media desde los puntos del *ground truth* hacia sus puntos más cercanos en la superficie reconstruida.

\[
\operatorname{Completeness} =
\frac{1}{|G|}
\sum_{g \in G}
\min_{m \in M} \|g-m\|_2
\]

Un valor menor indica que una mayor proporción de la geometría de referencia está representada por la reconstrucción.

### 5.3. Overall

Representa el promedio de Accuracy y Completeness:

\[
\operatorname{Overall} =
\frac{\operatorname{Accuracy}+\operatorname{Completeness}}{2}
\]

Esta métrica permite resumir en un solo valor el comportamiento de ambas direcciones de distancia.

Las tres métricas se expresan en milímetros. En todos los casos, los valores menores representan una mejor correspondencia geométrica.

## 6. Criterio de comparación

La comparación considera conjuntamente las tres métricas:

- Accuracy permite analizar la proximidad de la reconstrucción respecto al *ground truth*.
- Completeness permite analizar cuánto de la geometría de referencia logra recuperar el método.
- Overall resume ambas medidas mediante su promedio.

No se considera suficiente analizar únicamente Accuracy, ya que una reconstrucción puede aproximarse bien a las regiones que representa y, aun así, omitir partes importantes de la superficie.

Por ello, Overall se utiliza como indicador resumido, mientras que las dos métricas direccionales permiten interpretar las diferencias geométricas.

## 7. Resultados experimentales

Los resultados siguientes corresponden al evaluador común, con `density=0.2`, `patch=60`, `max_dist=20` y `seed=42`.

| Escena | Método | Accuracy (mm) | Completeness (mm) | Overall (mm) |
|---|---|---:|---:|---:|
| scan24 | QGS | 0.37360 | 0.45668 | **0.41514** |
| scan24 | VCR-GauS | 0.48166 | 0.60121 | **0.54143** |
| scan37 | QGS | 0.71810 | 0.56244 | **0.64027** |
| scan37 | VCR-GauS | 0.89858 | 0.90154 | **0.90006** |

QGS obtuvo un Overall aproximadamente 23.33 % menor que VCR-GauS en `scan24` y 28.86 % menor en `scan37`. Estas diferencias describen únicamente las dos escenas evaluadas.

Para `scan37`, el evaluador propio de QGS informó un Overall de 0.64020618 mm en `results.json`; el evaluador común informó 0.64026834 mm. La comparación entre métodos utiliza los resultados del evaluador común para mantener el mismo protocolo.

## 8. Alcance y limitaciones

Los resultados obtenidos corresponden a dos escenas seleccionadas del conjunto DTU (`scan24` y `scan37`).

Por tanto, permiten comparar el comportamiento de QGS y VCR-GauS en este escenario concreto, pero no permiten establecer una conclusión definitiva sobre cuál método obtiene mejores resultados en todos los escenarios.

Asimismo, la comparación está centrada en la calidad geométrica de las reconstrucciones y no pretende evaluar de manera exhaustiva la calidad visual del renderizado, el tiempo de entrenamiento o el consumo de memoria.

## 9. Referencias

- Zhang, Z., Huang, B., Jiang, H., Zhou, L., Xiang, X., & Shen, S. (2025). *Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

- Chen, H., Wei, F., Li, C., Huang, T., Wang, Y., & Lee, G. H. (2024). *VCR-GauS: View-Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction*. Advances in Neural Information Processing Systems (NeurIPS).

- Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., & Aanæs, H. (2014). *Large Scale Multi-view Stereopsis Evaluation*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
