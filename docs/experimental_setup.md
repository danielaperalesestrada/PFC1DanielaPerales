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

Se utilizó el conjunto de datos **DTU (Multi-View Stereo Evaluation)**, seleccionando siete escenas: `scan24`, `scan37`, `scan40`, `scan55`, `scan63`, `scan65` y `scan69`.

El conjunto se utiliza tanto para entrenar los modelos como para evaluar geométricamente las superficies reconstruidas.

Para la evaluación se utilizaron los siguientes archivos de referencia:

- `scan24`: `Points/stl/stl024_total.ply`, `ObsMask/ObsMask24_10.mat` y `ObsMask/Plane24.mat`.
- `scan37`: `Points/stl/stl037_total.ply`, `ObsMask/ObsMask37_10.mat` y `ObsMask/Plane37.mat`.
- `scan40`: `Points/stl/stl040_total.ply`, `ObsMask/ObsMask40_10.mat` y `ObsMask/Plane40.mat`.
- `scan55`: `Points/stl/stl055_total.ply`, `ObsMask/ObsMask55_10.mat` y `ObsMask/Plane55.mat`.
- `scan63`: `Points/stl/stl063_total.ply`, `ObsMask/ObsMask63_10.mat` y `ObsMask/Plane63.mat`.
- `scan65`: `Points/stl/stl065_total.ply`, `ObsMask/ObsMask65_10.mat` y `ObsMask/Plane65.mat`.
- `scan69`: `Points/stl/stl069_total.ply`, `ObsMask/ObsMask69_10.mat` y `ObsMask/Plane69.mat`.

Los datos y archivos de referencia se almacenan localmente y no se incluyen en el repositorio.

## 3. Configuración de entrenamiento

### 3.1. QGS

Se utilizaron configuraciones DTU independientes para cada escena, almacenadas en `configs/qgs/dtu_scan24.yaml`, `configs/qgs/dtu_scan37.yaml` y `configs/qgs/dtu_scan40.yaml`. Las configuraciones de `scan37` y `scan40` conservan los parámetros de `scan24` y cambian las rutas de entrada/salida y el identificador de escena.

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

Se utilizaron `configs/vcr_gaus/dtu_scan24.yaml`, `configs/vcr_gaus/dtu_scan37.yaml` y `configs/vcr_gaus/dtu_scan40.yaml`. Las configuraciones conservan los mismos parámetros; la escena y el directorio de salida se proporcionan mediante los argumentos del script de entrenamiento.

Durante el entrenamiento se utilizaron los siguientes argumentos específicos:

| Parámetro | Valor |
|---|---|
| `model.data_device` | `cuda` en el intento inicial; `cpu` al reanudar scan40 desde el checkpoint de 4000 iteraciones |
| `model.resolution` | `-1` |
| `model.white_background` | `True` |
| `pipline.convert_SHs_python` | `True` |
| Iteraciones finales | 30000 |
| Checkpoints | Cada 2000 iteraciones, desde 2000 hasta 28000 |
| Guardado del modelo | 14000, 29000 y 30000 |

El entrenamiento utilizó cuatro hilos de OpenMP y la configuración de memoria CUDA `max_split_size_mb:128` (`scan24`, `scan37`, `scan40` y el entrenamiento original de `scan55`; en `scan63`, `scan65` y `scan69` se usó `max_split_size_mb:64`, véase más abajo).

VCR-GauS se ejecutó con `load_normal: true`, `normal_folder: normal_npz_indoor` y `load_depth: false`. Para `scan37` y `scan40` se generaron 49 mapas de normales por escena con GeoWizard (`lemonaddie/geowizard`) en dominio `indoor`, con `ensemble_size=3`, `denoise_steps=10`, `processing_res=768` y `seed=0`. En ambas escenas se verificó que los mapas tuvieran forma `(1162, 1554, 3)`, tipo `float32`, valores finitos y correspondencia con las 49 imágenes.

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

En QGS se utilizó la configuración de malla con `mesh_res=1024`, `voxel_size=0.004`, `sdf_trunc=0.016` y `depth_trunc=3.0`. Durante la extracción de `scan37`, QGS indicó que, para un radio de contorno estimado de 2.15, se recomendaba `depth_trunc` de al menos 4.29. En `scan40`, el radio estimado fue 1.81 y se recomendó `depth_trunc` de al menos 3.61. En ambas escenas se generaron las mallas manteniendo el valor configurado de 3.0; estas advertencias quedan registradas como limitaciones que conviene estudiar.

## 5. Métricas de evaluación

Se utilizaron tres métricas geométricas basadas en distancias entre las superficies reconstruidas y el ground truth.

- **Accuracy (D2S):** distancia media desde los puntos de la superficie reconstruida hasta los puntos más cercanos del ground truth.
- **Completeness (S2D):** distancia media desde los puntos del ground truth hasta la superficie reconstruida.
- **Overall:** promedio de Accuracy y Completeness.

Los resultados se expresan en milímetros. Un valor menor representa una menor distancia respecto a la geometría de referencia.

## 6. Resultados experimentales

Los resultados del evaluador común para las siete escenas son:

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

QGS obtuvo un Overall aproximadamente un 23.33 % menor que VCR-GauS en `scan24`, un 28.86 % menor en `scan37`, un 1.48 % menor en `scan40`, un 10.74 % menor en `scan55`, un 11.75 % menor en `scan63`, un 34.98 % menor en `scan65` y un 37.10 % menor en `scan69`. Estos resultados corresponden únicamente a las siete escenas evaluadas y no representan el conjunto completo DTU.

Los resultados detallados se encuentran en `results/metrics/comparison.csv`, `results/metrics/comparison_scan37.csv` y `results/metrics/comparison_scan40.csv`. Las figuras se almacenan en `results/figures/metrics_comparison_scan24.png`, `results/figures/metrics_comparison_scan37.png` y `results/figures/metrics_comparison_scan40.png`.

En `scan37`, el filtrado común procesó las 49 vistas y conservó todos los vértices de ambas mallas: 195 715/195 715 para QGS y 195 664/195 664 para VCR-GauS. En `scan40` también se procesaron 49 vistas y se conservaron todos los vértices: 166 700/166 700 para QGS y 170 596/170 596 para VCR-GauS. La evaluación de métricas aplicó posteriormente la caja delimitadora y la máscara de observación DTU a los puntos muestreados.

El evaluador propio de QGS informó un Overall de 0.64020618 mm para `scan37`, mientras que el evaluador común informó 0.64026834 mm. Para `scan40`, los valores fueron 0.36002644 mm con el evaluador propio y 0.36009064 mm con el evaluador común. La tabla comparativa utiliza el evaluador común para mantener el mismo protocolo entre métodos.

### Incidencias durante la ejecución de scan40

El primer intento de entrenamiento de VCR-GauS se interrumpió por falta de memoria de GPU alrededor de la iteración 5700. El último checkpoint guardado correspondía a la iteración 4000. Se reanudó desde `chkpnt4000.pth` mediante `train.start_checkpoint`, utilizando `model.data_device=cpu` para almacenar las imágenes y normales de las cámaras en RAM y transferirlas a CUDA durante el cálculo. Se mantuvo la resolución original y el entrenamiento completó las 30 000 iteraciones.

### Scan55: configuración, incidencias y medición de eficiencia

- **QGS:** `configs/qgs/dtu_scan55.yaml` conserva los hiperparámetros de las demás escenas. El entrenamiento original se lanzó con una configuración no versionada cuyo `case_name` heredó `scan24` de `base.yaml`; en el código, `case_name` solo se usa para lanzar la evaluación automática final y para elegir el GT en `eval.py`, por lo que no afectó a los datos de entrenamiento (`root_dir` apuntaba a `scan55`). Esa evaluación automática comparó con el GT de `scan24` y se descartó. Las métricas oficiales proceden de `evaluate_metrics.py --scan-id 55`. Los hiperparámetros comprobados coinciden con los de `dtu_scan55.yaml`.
- **VCR-GauS:** se generaron 49 mapas de normales con GeoWizard (`indoor`, `ensemble_size=3`, `denoise_steps=10`, `processing_res=768`, `seed=0`), con forma `(1162, 1554, 3)`, tipo `float32` y valores finitos. El `meta.json` lo generó VCR-GauS automáticamente al arrancar.
- **Filtrado común:** se procesaron 49 vistas. QGS conservó 135 010 de 135 461 vértices y VCR-GauS 124 184 de 124 185.
- **Postproceso de malla:** QGS conserva los 10 clústeres más grandes (`render.py`); en VCR-GauS se usó `--clean` con `num_cluster=1`. Es el protocolo original de cada método.
- **Medición de eficiencia:** repeticiones guardadas aparte; la configuración de `scan55` para la repetición de QGS es `configs/qgs/dtu_scan55_eff.yaml` (solo cambia `model_path`) y `train_qgs.sh` acepta `QGS_CONFIG` y `QGS_OUTPUT_DIR`.
- **Incidencia de memoria en VCR-GauS:** una repetición con `data_device=cuda` y `max_split_size_mb:128` se interrumpió por falta de memoria de GPU cerca de la iteración 12 000 (tras 6 572 s). La repetición que completó las 30 000 iteraciones se lanzó con `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` y también con `cuda`. Con una sola ejecución por configuración no se puede atribuir la diferencia a ese parámetro. Un intento previo con `expandable_segments` falló al arrancar porque PyTorch 2.0.1 no lo soporta.
- **Repetibilidad:** las mallas de las repeticiones difieren algo de las originales (QGS: 134 711 frente a 135 461 vértices; VCR-GauS: 124 620 frente a 124 185 tras la limpieza). Las métricas de las repeticiones no se evaluaron.

### Escenas scan63, scan65 y scan69 (pipeline único)

- Se ejecutaron con `scripts/run_scene.sh <id>`, que encadena normales (GeoWizard), entrenamiento y malla de VCR-GauS, entrenamiento y malla de QGS, evaluación geométrica, métricas fotométricas y resumen de eficiencia. Cada fase pesada se ejecuta bajo `monitor_training.sh`, omite lo que ya existe y aborta si encuentra una carpeta de salida a medias.
- Los configs se crean a partir de los de `scan40` (QGS, cambiando el identificador de escena) y de `scan63` (VCR-GauS); el script comprueba que `case_name` coincida con la escena y que `eval` sea `false` en ambos.
- VCR-GauS se entrenó con `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64` (frente a `128` en los entrenamientos anteriores) y `data_device=cuda`. Los tres entrenamientos completaron las 30 000 iteraciones.
- Al terminar el entrenamiento de QGS, `train.py` lanza una evaluación automática que falla porque aún no existe `fuse_post.ply`; añade unos segundos al tiempo medido y no afecta al modelo. La malla se genera después con `render.py`.
- En `scan63`, QGS estimó un radio de contorno de 2.70 y recomendó un `depth_trunc` de al menos 5.40; se mantuvo 3.0.
- Los mapas de normales se midieron con el monitor (entre 1 079 y 1 091 s por escena).
- Al terminar, se borran los checkpoints intermedios de VCR-GauS (`CLEAN_CKPT=0` los conserva); se conserva `chkpnt30000.pth`.
- Las métricas fotométricas se describen en la sección 7.1 de `methodology.md`.

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
