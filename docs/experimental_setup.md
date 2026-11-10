# Configuración experimental

## 1. Hardware y software

Los experimentos se realizaron utilizando una NVIDIA GeForce RTX 4060 Laptop GPU con 8 GB de VRAM, bajo Linux.

Debido a los diferentes requisitos de ambos métodos, se utilizaron entornos Conda independientes.

| Componente | QGS | VCR-GauS |
|---|---|---|
| Python | 3.9.19 | 3.10.22 |
| PyTorch | 2.2.2 | 2.0.1 |
| Torchvision | 0.17.2 | 0.15.2 |
| CUDA | 12.1 | 11.8 |
| GCC | 12 | 11 |

Las instrucciones de instalación y compilación se encuentran en `environments/qgs/` y `environments/vcr_gaus/`.

## 2. Conjunto de datos

Se utilizó el conjunto de datos **DTU (Multi-View Stereo Evaluation)**, seleccionando el escaneo 24 (`scan24`).

El conjunto se utiliza tanto para entrenar los modelos como para evaluar geométricamente las superficies reconstruidas.

Para la evaluación se utilizaron los siguientes archivos de referencia:

- `Points/stl/stl024_total.ply`: nube de puntos de referencia.
- `ObsMask/ObsMask24_10.mat`: máscara de observación.
- `ObsMask/Plane24.mat`: información del plano de referencia.

Los datos y archivos de referencia se almacenan localmente y no se incluyen en el repositorio.

## 3. Configuración de entrenamiento

### 3.1. QGS

Se utilizó la configuración DTU adaptada al escaneo 24, almacenada en:

`configs/qgs/dtu_scan24.yaml`

Entre los parámetros específicos utilizados se encuentran:

| Parámetro | Valor |
|---|---:|
| `depth_ratio` | 0.0 |
| `mesh_res` | 1024 |
| `voxel_size` | 0.004 |
| `sdf_trunc` | 0.016 |
| `depth_trunc` | 3.0 |
| `percent_dense` | 0.0001 |
| `densify_grad_threshold` | 1.0 |
| `multi_view_weight_from_iter` | 7000 |
| `multi_view_weight_until_iter` | 30000 |
| `normal_from_iter` | 7000 |
| `multi_view_ncc_weight` | 0.5 |
| `multi_view_geo_weight` | 0.03 |
| `lambda_dist` | 10000.0 |
| `lambda_normal` | 0.03 |

La configuración completa se encuentra en el archivo YAML correspondiente.

### 3.2. VCR-GauS

Se utilizó la configuración DTU del repositorio, adaptada al escaneo 24 y almacenada en:

`configs/vcr_gaus/dtu_scan24.yaml`

Durante el entrenamiento se utilizaron los siguientes argumentos específicos:

| Parámetro | Valor |
|---|---|
| `model.data_device` | `cuda` |
| `model.resolution` | `-1` |
| `model.white_background` | `True` |
| `pipline.convert_SHs_python` | `True` |
| Iteraciones finales | 30000 |
| Checkpoints | Cada 2000 iteraciones, desde 2000 hasta 28000 |
| Guardado del modelo | 14000, 29000 y 30000 |

El entrenamiento utilizó cuatro hilos de OpenMP y la configuración de memoria CUDA `max_split_size_mb:128`.

## 4. Reconstrucción y evaluación geométrica

Para comparar ambos métodos se utilizó un procedimiento común de filtrado de las mallas mediante la información de observación del conjunto DTU.

La evaluación se realizó utilizando los mismos parámetros para ambas reconstrucciones:

| Parámetro | Valor |
|---|---:|
| `density` | 0.2 |
| `patch` | 60 |
| `max_dist` | 20 |
| `seed` | 42 |
| `workers` | 4 |

De esta manera, las dos mallas se evaluaron bajo las mismas condiciones.

## 5. Métricas de evaluación

Se utilizaron tres métricas geométricas basadas en distancias entre las superficies reconstruidas y el ground truth.

- **Accuracy (D2S):** distancia media desde los puntos de la superficie reconstruida hasta los puntos más cercanos del ground truth.
- **Completeness (S2D):** distancia media desde los puntos del ground truth hasta la superficie reconstruida.
- **Overall:** promedio de Accuracy y Completeness.

Los resultados se expresan en milímetros. Un valor menor representa una menor distancia respecto a la geometría de referencia.

## 6. Resultados experimentales

Los resultados correspondientes al DTU `scan24` son:

| Método | Accuracy (mm) | Completeness (mm) | Overall (mm) |
|---|---:|---:|---:|
| QGS | 0.37360 | 0.45668 | **0.41514** |
| VCR-GauS | 0.48166 | 0.60121 | **0.54143** |

QGS obtuvo un valor Overall aproximadamente un 23.33 % menor que VCR-GauS en este escaneo.

Estos resultados corresponden exclusivamente al escaneo 24 y no deben interpretarse como una comparación sobre el conjunto completo DTU.

Los resultados detallados se encuentran en `results/metrics/comparison.csv`, mientras que su representación gráfica se almacena en `results/figures/metrics_comparison_scan24.png`.

## 7. Reproducibilidad

Los archivos necesarios para reproducir los experimentos se organizan de la siguiente manera:

- `configs/`: configuraciones utilizadas para ambos métodos.
- `scripts/training/`: scripts de entrenamiento.
- `scripts/evaluation/`: procedimientos de evaluación.
- `scripts/analysis/`: scripts para comparar métricas y generar visualizaciones.
- `environments/`: configuraciones de los entornos de ejecución.
- `patches/vcr_gaus/`: correcciones documentadas para VCR-GauS.
- `results/`: métricas y visualizaciones obtenidas.

Los datasets, checkpoints y registros de entrenamiento no se incluyen en el repositorio.
