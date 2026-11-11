# Configuración experimental

## 1. Hardware y software

Los experimentos se realizaron en Ubuntu 24.04.3 LTS utilizando una NVIDIA GeForce RTX 4060 Laptop GPU con 8 GB de VRAM. Durante las ejecuciones registradas se utilizó el kernel `6.14.0-36-generic` y el controlador NVIDIA `580.95.05`.

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

Se utilizó el conjunto de datos **DTU (Multi-View Stereo Evaluation)**, seleccionando dos escenas: `scan24` y `scan37`.

El conjunto se utiliza tanto para entrenar los modelos como para evaluar geométricamente las superficies reconstruidas.

Para la evaluación se utilizaron los siguientes archivos de referencia:

- `scan24`: `Points/stl/stl024_total.ply`, `ObsMask/ObsMask24_10.mat` y `ObsMask/Plane24.mat`.
- `scan37`: `Points/stl/stl037_total.ply`, `ObsMask/ObsMask37_10.mat` y `ObsMask/Plane37.mat`.

Los datos y archivos de referencia se almacenan localmente y no se incluyen en el repositorio.

## 3. Configuración de entrenamiento

### 3.1. QGS

Se utilizaron configuraciones DTU independientes para cada escena, almacenadas en `configs/qgs/dtu_scan24.yaml` y `configs/qgs/dtu_scan37.yaml`. La configuración de `scan37` conserva los parámetros de `scan24` y cambia las rutas de entrada/salida y el identificador de escena.

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

Se utilizaron `configs/vcr_gaus/dtu_scan24.yaml` y `configs/vcr_gaus/dtu_scan37.yaml`. Ambas conservan los mismos parámetros; la escena y el directorio de salida se proporcionan mediante los argumentos del script de entrenamiento.

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

VCR-GauS se ejecutó con `load_normal: true`, `normal_folder: normal_npz_indoor` y `load_depth: false`. Para `scan37` se generaron 49 mapas de normales con GeoWizard (`lemonaddie/geowizard`) en dominio `indoor`, con `ensemble_size=3`, `denoise_steps=10`, `processing_res=768` y `seed=0`. Se verificó que cada mapa tuviera forma `(1162, 1554, 3)`, tipo `float32`, valores finitos y correspondencia con las 49 imágenes.

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

Para la extracción de geometría se utilizó el checkpoint de 30 000 iteraciones de cada método. En VCR-GauS se ejecutó `depth2mesh.py` con `method=tsdf_cpu`, `split=1`, `voxel_size=0.004`, `num_cluster=1`, `max_depth=3` y `prob_thres=0.15`.

En QGS se utilizó la configuración de malla con `mesh_res=1024`, `voxel_size=0.004`, `sdf_trunc=0.016` y `depth_trunc=3.0`. Durante la extracción de `scan37`, QGS indicó que, para un radio de contorno estimado de 2.15, se recomendaba `depth_trunc` de al menos 4.29. La malla se generó manteniendo el valor configurado de 3.0; esta advertencia queda registrada como una limitación que conviene estudiar.

## 5. Métricas de evaluación

Se utilizaron tres métricas geométricas basadas en distancias entre las superficies reconstruidas y el ground truth.

- **Accuracy (D2S):** distancia media desde los puntos de la superficie reconstruida hasta los puntos más cercanos del ground truth.
- **Completeness (S2D):** distancia media desde los puntos del ground truth hasta la superficie reconstruida.
- **Overall:** promedio de Accuracy y Completeness.

Los resultados se expresan en milímetros. Un valor menor representa una menor distancia respecto a la geometría de referencia.

## 6. Resultados experimentales

Los resultados del evaluador común para ambas escenas son:

| Escena | Método | Accuracy (mm) | Completeness (mm) | Overall (mm) |
|---|---|---:|---:|---:|
| scan24 | QGS | 0.37360 | 0.45668 | **0.41514** |
| scan24 | VCR-GauS | 0.48166 | 0.60121 | **0.54143** |
| scan37 | QGS | 0.71810 | 0.56244 | **0.64027** |
| scan37 | VCR-GauS | 0.89858 | 0.90154 | **0.90006** |

QGS obtuvo un Overall aproximadamente un 23.33 % menor que VCR-GauS en `scan24` y un 28.86 % menor en `scan37`. Estos resultados corresponden únicamente a las dos escenas evaluadas y no representan el conjunto completo DTU.

Los resultados detallados se encuentran en `results/metrics/comparison.csv` y `results/metrics/comparison_scan37.csv`. Las figuras se almacenan en `results/figures/metrics_comparison_scan24.png` y `results/figures/metrics_comparison_scan37.png`.

En `scan37`, el filtrado común procesó las 49 vistas y conservó todos los vértices de ambas mallas: 195 715/195 715 para QGS y 195 664/195 664 para VCR-GauS. La evaluación de métricas aplicó posteriormente la caja delimitadora y la máscara de observación DTU a los puntos muestreados.

El evaluador propio de QGS informó un Overall de 0.64020618 mm para `scan37`, mientras que el evaluador común informó 0.64026834 mm. La tabla comparativa utiliza el evaluador común para mantener el mismo protocolo entre métodos.

## 7. Reproducibilidad

Los archivos necesarios para reproducir los experimentos se organizan de la siguiente manera:

- `configs/`: configuraciones utilizadas para ambos métodos y escenas.
- `scripts/training/`: scripts de entrenamiento.
- `scripts/evaluation/`: procedimientos de evaluación.
- `scripts/analysis/`: scripts para comparar métricas y generar visualizaciones.
- `environments/`: configuraciones de los entornos de ejecución.
- `patches/vcr_gaus/`: correcciones documentadas para VCR-GauS.
- `results/`: métricas y visualizaciones obtenidas.

Los datasets, checkpoints y registros de entrenamiento no se incluyen en el repositorio.
