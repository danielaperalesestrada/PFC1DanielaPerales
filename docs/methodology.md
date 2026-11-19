# Metodología

## 1. Objetivo

El objetivo de este proyecto es comparar la calidad geométrica de dos métodos de reconstrucción de superficies basados en Gaussian Splatting: **Quadratic Gaussian Splatting (QGS)** y **VCR-GauS**.

Ambos métodos buscan reconstruir geometría tridimensional a partir de imágenes tomadas desde múltiples puntos de vista, pero utilizan estrategias diferentes para representar y optimizar las superficies.

La comparación se realiza sobre el conjunto de datos DTU, utilizando siete escenas (`scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65` y `scan69`) y métricas geométricas basadas en distancias entre las superficies reconstruidas y la geometría de referencia.

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

Para este proyecto se seleccionaron siete escenas (`scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65` y `scan69`), todas con imágenes multivista, información de calibración y una reconstrucción de referencia obtenida mediante escaneo de luz estructurada.

Para evaluar la geometría se utilizan los siguientes archivos:

- `scan24`: `stl024_total.ply`, `ObsMask24_10.mat` y `Plane24.mat`.
- `scan37`: `stl037_total.ply`, `ObsMask37_10.mat` y `Plane37.mat`.
- `scan40`: `stl040_total.ply`, `ObsMask40_10.mat` y `Plane40.mat`.
- `scan55`: `stl055_total.ply`, `ObsMask55_10.mat` y `Plane55.mat`.
- `scan63`: `stl063_total.ply`, `ObsMask63_10.mat` y `Plane63.mat`.
- `scan65`: `stl065_total.ply`, `ObsMask65_10.mat` y `Plane65.mat`.
- `scan69`: `stl069_total.ply`, `ObsMask69_10.mat` y `Plane69.mat`.

El experimento se limita a este escaneo y no representa una evaluación exhaustiva de todos los escenarios del conjunto DTU.

## 4. Procedimiento experimental

La comparación se organiza en cinco etapas:

### 4.1. Preparación de los datos

Se prepara cada escena de DTU junto con sus imágenes, parámetros de cámara y archivos de referencia. Los dos métodos utilizan la misma escena de entrada y configuraciones específicas compatibles con sus respectivos repositorios.

Para VCR-GauS, `load_normal` permanece activado y `load_depth` desactivado. En `scan37`, `scan40`, `scan55`, `scan63`, `scan65` y `scan69` se generaron 49 mapas de normales `normal_npz_indoor` por escena con GeoWizard, usando dominio `indoor`, `ensemble_size=3`, `denoise_steps=10`, `processing_res=768` y `seed=0`. En todas ellas se validó la correspondencia de los mapas con las 49 imágenes; los mapas de normales tienen forma `(1162, 1554, 3)`, tipo `float32` y valores finitos.

### 4.2. Entrenamiento

Se ejecuta QGS y VCR-GauS en entornos independientes para mantener la compatibilidad de sus versiones de Python, PyTorch y CUDA.

Cada método utiliza su propia representación y procedimiento de optimización. Los detalles de software y los parámetros experimentales se encuentran en `experimental_setup.md`.

### 4.3. Extracción de superficies

Una vez finalizado el entrenamiento, se obtiene una malla tridimensional de cada método.

QGS y VCR-GauS utilizan sus respectivos procedimientos de reconstrucción de malla. Los resultados se conservan en formato PLY para su posterior evaluación.

### 4.4. Filtrado mediante una máscara común

Para reducir las diferencias derivadas de las regiones no observadas, se aplica un procedimiento común de filtrado utilizando la máscara de observación de DTU.

Este procedimiento permite evaluar ambas reconstrucciones bajo las mismas condiciones espaciales.

Las mallas filtradas y los informes correspondientes se almacenan por escena, por ejemplo en `results/dtu/scan24/comparison/`, `results/dtu/scan37/comparison/` y `results/dtu/scan40/comparison/`, cada una con subdirectorios `qgs/` y `vcr_gaus/`.

En `scan37`, el filtrado común procesó las 49 vistas, pero no eliminó vértices: QGS conservó 195 715 de 195 715 y VCR-GauS conservó 195 664 de 195 664. En `scan40` también se procesaron 49 vistas y se conservaron todos los vértices: QGS mantuvo 166 700 de 166 700 y VCR-GauS 170 596 de 170 596. La evaluación de métricas aplica además la caja delimitadora y la máscara de observación DTU a los puntos muestreados.

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
| scan40 | QGS | 0.33954 | 0.38065 | **0.36009** |
| scan40 | VCR-GauS | 0.34400 | 0.38703 | **0.36552** |
| scan55 | QGS | 0.33458 | 0.40657 | **0.37057** |
| scan55 | VCR-GauS | 0.37248 | 0.45784 | **0.41516** |
| scan63 | QGS | 1.04744 | 0.63538 | **0.84141** |
| scan63 | VCR-GauS | 1.04021 | 0.86669 | **0.95345** |
| scan65 | QGS | 0.56241 | 0.64932 | **0.60587** |
| scan65 | VCR-GauS | 0.80002 | 1.06374 | **0.93188** |
| scan69 | QGS | 0.51688 | 0.48544 | **0.50116** |
| scan69 | VCR-GauS | 0.74572 | 0.84782 | **0.79677** |

QGS obtuvo un Overall aproximadamente 23.33 % menor que VCR-GauS en `scan24`, 28.86 % menor en `scan37`, 1.48 % menor en `scan40`, 10.74 % menor en `scan55`, 11.75 % menor en `scan63`, 34.98 % menor en `scan65` y 37.10 % menor en `scan69`. Estas diferencias describen únicamente las siete escenas evaluadas.

Para `scan37`, el evaluador propio de QGS informó un Overall de 0.64020618 mm en `results.json`; el evaluador común informó 0.64026834 mm. Para `scan40`, los valores fueron 0.36002644 mm con el evaluador propio de QGS y 0.36009064 mm con el evaluador común. La comparación entre métodos utiliza los resultados del evaluador común para mantener el mismo protocolo.

### 7.1. Fotometría (PSNR, SSIM y LPIPS)

Se calcularon en `scan63`, `scan65` y `scan69` sobre las 49 vistas de entrenamiento (`scripts/photometric/`):

- Cada método renderiza sus vistas con el nombre de la imagen original, y el evaluador aborta si un render se parece más a otra imagen que a la suya.
- Resolución común de 777×581 (la nativa de QGS). La referencia y los renders de VCR-GauS, que salen a 1554×1162, se reducen con filtro Lanczos.
- Fondo negro en ambos renders. PSNR y SSIM se calculan solo sobre los píxeles del objeto (canal alfa de la imagen original, umbral 0.5) y LPIPS (VGG) sobre la caja delimitadora del objeto.
- Son vistas de entrenamiento: miden cuánto se ajusta el modelo a las imágenes y no su generalización. Los artículos de QGS y VCR-GauS reportan estas métricas en Mip-NeRF 360 y no en DTU, por lo que no hay valores publicados con los que compararlas.
- La reducción de resolución podría favorecer a VCR-GauS (promedia el ruido de su render, mientras que el de QGS es nativo). No se ha cuantificado.

### 7.2. Eficiencia computacional

El tiempo y la memoria se midieron en `scan55`, `scan63`, `scan65` y `scan69` con `scripts/monitoring/monitor_training.sh`, una ejecución por método, fase y escena, en una NVIDIA GeForce RTX 4060 Laptop GPU de 8 GB. El script ejecuta el comando sin modificarlo y registra duración, memoria de GPU (global y del proceso), utilización, temperatura, potencia y RAM del sistema con un muestreo de 1 s.

- `scan63`, `scan65` y `scan69` se ejecutaron una sola vez con `scripts/run_scene.sh`: geometría, fotometría y eficiencia proceden de la misma ejecución. En `scan55` la eficiencia procede de una repetición aparte; la geometría, del entrenamiento original.
- En estas mediciones VCR-GauS usó `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`.
- Razón de tiempo de optimización VCR-GauS/QGS: de 1.16 a 1.55 (media 1.43, 4 escenas). La variación entre escenas es mayor en VCR-GauS y no se ha identificado su causa.
- Memoria del proceso en la optimización: VCR-GauS entre 7 618 y 7 648 MiB (la GPU tiene 8 188 MiB) y QGS entre 2 544 y 3 160 MiB.
- Limitaciones: muestreo de 1 s (puede omitir picos breves); la RAM es la del sistema completo; la potencia y la temperatura son máximos, no energía consumida; el tiempo de optimización de QGS incluye unos segundos de una evaluación automática final que falla; el mallado no es estrictamente comparable (QGS conserva 10 clústeres con `render.py` y VCR-GauS usa `depth2mesh.py` con `--clean` y 1 clúster); el preprocesado de VCR-GauS en `scan55` no se midió con el monitor.

Las tablas están en el README y se generan con `scripts/analysis/build_summary_tables.py`.

## 8. Alcance y limitaciones

Los resultados obtenidos corresponden a siete escenas seleccionadas del conjunto DTU (`scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65` y `scan69`).

Por tanto, permiten comparar el comportamiento de QGS y VCR-GauS en este escenario concreto, pero no permiten establecer una conclusión definitiva sobre cuál método obtiene mejores resultados en todos los escenarios.

Asimismo, la comparación de fidelidad está centrada en la calidad geométrica de las reconstrucciones. La fotometría se midió solo en tres escenas y sobre vistas de entrenamiento (sección 7.1), y la eficiencia computacional en cuatro escenas con una ejecución por configuración (sección 7.2); ninguna de las dos permite generalizar.

## 9. Referencias

- Zhang, Z., Huang, B., Jiang, H., Zhou, L., Xiang, X., & Shen, S. (2025). *Quadratic Gaussian Splatting: High Quality Surface Reconstruction with Second-order Geometric Primitives*. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

- Chen, H., Wei, F., Li, C., Huang, T., Wang, Y., & Lee, G. H. (2024). *VCR-GauS: View-Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction*. Advances in Neural Information Processing Systems (NeurIPS).

- Jensen, R., Dahl, A., Vogiatzis, G., Tola, E., & Aanæs, H. (2014). *Large Scale Multi-view Stereopsis Evaluation*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
