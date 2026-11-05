# Benchmark: QGS vs VCR-GauS en DTU

Este proyecto compara dos métodos de reconstrucción de superficies
basados en Gaussian Splatting:

-   **QGS (Quadratic Gaussian Splatting, ICCV 2025):** utiliza
    primitivas geométricas cuadráticas de segundo orden.
-   **VCR-GauS (NeurIPS 2024):** incorpora un regularizador de
    profundidad y normales consistente entre múltiples vistas sobre
    surfels 2D.

El objetivo es evaluar la calidad geométrica de ambos métodos mediante
métricas de Chamfer Distance (Accuracy, Completeness y Overall),
utilizando el conjunto de datos DTU. Asimismo, se documentan las
correcciones aplicadas al repositorio oficial de VCR-GauS para conseguir
un entrenamiento estable.

## Estructura del proyecto

``` text
PFC1DanielaPerales/
├── QGS/                  Repositorio de QGS
├── VCR-GauS/             Repositorio de VCR-GauS
├── datasets/             Conjuntos de datos y ground truth
├── outputs/              Resultados de los entrenamientos
├── benchmarks/           Resultados del benchmark de QGS
├── analysis/             Configuraciones y análisis experimental
│   ├── configs/          Configuraciones utilizadas
│   ├── depth_maps/       Mapas de profundidad de VCR-GauS
│   ├── logs/             Registros del entrenamiento
│   └── resultados_benchmark.csv
└── README.md
```

Los conjuntos de datos y los resultados de entrenamiento no están
incluidos en el repositorio.

## Requisitos

-   Sistema operativo Linux (Ubuntu 22.04 o 24.04).
-   GPU NVIDIA con al menos 8 GB de VRAM.
-   CUDA 12.1 para QGS y CUDA 11.8 para VCR-GauS.
-   Miniconda o Anaconda.
-   GCC 11 y GCC 12.

Las configuraciones se probaron en una NVIDIA GeForce RTX 4060 Laptop
GPU de 8 GB.

## Instalación y compilación

Cada método utiliza un entorno independiente debido a sus diferentes
versiones de CUDA y PyTorch.

### QGS

Configuración utilizada: CUDA 12.1 y PyTorch 2.2.2.

``` bash
cd QGS

conda env create -f environment.yml
conda activate QGS

conda install -c conda-forge cuda-cccl -y

export CC=gcc-12 CXX=g++-12
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CONDA_PREFIX/include/cccl:$CPATH

pip install ./submodules/diff-quadratic-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

### VCR-GauS

Configuración utilizada: CUDA 11.8 y PyTorch 2.0.1.

``` bash
cd VCR-GauS

conda create -n vcr python=3.10 -y
conda activate vcr

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

conda install -c nvidia cuda-nvcc=11.8.89 -y
conda install -c "nvidia/label/cuda-11.8.0" cuda-cudart-dev -y

pip install -e ".[train]" --no-build-isolation
pip install "setuptools<81"

git submodule update --init --recursive

export CC=gcc-11 CXX=g++-11

pip install ./submodules/diff-gaussian-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

## Conjunto de datos DTU

Se utilizan dos conjuntos de datos: uno para el entrenamiento y otro
para la evaluación de las reconstrucciones.

**Datos de entrenamiento:**

``` bash
gdown 1ODiOu72tAGPTnhVn0cFZ9MvymDgcoHxQ

mkdir -p datasets/dtu
tar -xzf dtu.tar.gz -C datasets/dtu
```

**Ground truth:**

``` bash
unzip dtu-sample.zip
mv "SampleSet/MVS Data" SampleSet/MVS_Data
```

## Entrenamiento y evaluación

### QGS

``` bash
conda activate QGS
cd QGS

python train.py -s ../datasets/dtu/DTU/scan24 -m ../outputs/qgs_scan24 --conf_path ./config/DTU.yaml

python render.py --conf_path ../outputs/qgs_scan24/config.yaml --skip_train --skip_test

python ./scripts/eval_dtu/eval.py --conf_path ../outputs/qgs_scan24/config.yaml
```

### VCR-GauS

``` bash
conda activate vcr
cd VCR-GauS

python train.py --config=configs/dtu/base.yaml --logdir=../outputs/vcr_gaus/dtu/scan24 --model.source_path=../datasets/dtu/DTU/scan24/ --model.data_device=cuda --model.white_background=True --model.sh_degree=3 --optim.iterations=30000

python tools/depth2mesh.py --mesh_name ours.ply --split 1 --method tsdf_cpu --voxel_size 0.004 --num_cluster 1 --max_depth 3 --clean --prob_thres 0.15 --cfg_path ../outputs/vcr_gaus/dtu/scan24/config.yaml
```

## Resultados

### Evaluación geométrica: DTU scan24

Los resultados de QGS obtenidos para el escaneo 24 son los siguientes:

  Método     Accuracy (mm)   Completeness (mm)   Overall (mm)
  -------- --------------- ------------------- --------------
  QGS               0.3735              0.4567         0.4151

Como referencia, el artículo original de QGS reporta un valor Overall de
0.54 mm, correspondiente al promedio de 15 escaneos.

### Estabilidad del entrenamiento: VCR-GauS

Después de aplicar las correcciones descritas en este proyecto, VCR-GauS
completó las 30 000 iteraciones de entrenamiento sin presentar picos de
pérdida. Se registraron cero eventos de este tipo, frente a los 102
observados con la configuración original.

También se corrigió el rango de los valores de renderizado, limitándolos
al intervalo \[0, 1\]. Los mapas de profundidad obtenidos muestran una
geometría coherente, con bordes definidos y superficies planas.

Los resultados visuales se encuentran en `analysis/depth_maps/`.

### Métricas de evaluación

Se utilizan las siguientes métricas para evaluar la calidad geométrica
de las reconstrucciones:

-   **Accuracy:** distancia media de cada punto de la malla reconstruida
    al punto más cercano del ground truth.
-   **Completeness:** distancia media de cada punto del ground truth al
    punto más cercano de la malla reconstruida.
-   **Overall:** promedio de Accuracy y Completeness.

## Correcciones aplicadas a VCR-GauS

Se identificaron y corrigieron tres problemas principales en la
implementación original:

**1. Inestabilidad numérica del rasterizador**

Las escalas muy anisotrópicas de algunos gaussianos generaban
covarianzas 2D degeneradas en el kernel CUDA. Esto provocaba picos de
pérdida L1 y valores extremos durante el renderizado.

Se estableció un límite mínimo para las escalas de los gaussianos y se
restringieron los valores del renderizado al intervalo \[0, 1\].

**2. Duplicación de `loss_weight` en la configuración YAML**

El archivo `base.yaml` definía dos bloques `loss_weight`, por lo que el
segundo sobrescribía al primero y eliminaba el parámetro `l1_scale`.

Se unificaron ambos bloques y se estableció `l1_scale: 1.0`.

**3. Poda excesiva de gaussianos**

La configuración heredada de `reconstruct.yaml` aplicaba una poda
agresiva que eliminaba una proporción considerable de gaussianos durante
el entrenamiento.

Se modificó la configuración de poda para evitar su ejecución durante
las 30 000 iteraciones.

## Problemas conocidos

  ----------------------------------------------------------------------------------------------------------------
  Problema                    Solución
  --------------------------- ------------------------------------------------------------------------------------
  `libtiff.so.5` no           `sudo ln -s /lib/x86_64-linux-gnu/libtiff.so.6 /lib/x86_64-linux-gnu/libtiff.so.5`
  encontrado                  

  Incompatibilidad entre      `pip install "numpy<2"`
  NumPy 2.x y PyTorch         

  `pkg_resources` no          `pip install "setuptools<81"`
  encontrado                  

  `unsupported GNU version`   `export CC=gcc-11 CXX=g++-11`
  con CUDA 11.8               

  `cub/cub.cuh` no encontrado Instalar `cuda-cccl` y configurar `CPATH`.
  ----------------------------------------------------------------------------------------------------------------

## Referencias

-   Zhang et al. *Quadratic Gaussian Splatting: High Quality Surface
    Reconstruction with Second-order Geometric Primitives*. ICCV 2025.
-   Chen, H., Wei, F., Li, C., Huang, T., Wang, Y., & Lee, G. H.
    *VCR-GauS: View-Consistent Depth-Normal Regularizer for Gaussian
    Surface Reconstruction*. NeurIPS 2024.
-   Jensen et al. *Large Scale Multi-view Stereopsis Evaluation*. CVPR
    2014.
