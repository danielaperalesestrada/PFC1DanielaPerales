# Entorno de QGS

Este directorio contiene la configuración del entorno utilizado para ejecutar **Quadratic Gaussian Splatting (QGS)** en el benchmark con el conjunto de datos DTU.

La configuración se basa en el archivo `environment.yml` del repositorio oficial de QGS, con algunos ajustes para separar la instalación de dependencias de la compilación de los módulos CUDA personalizados.

## 1. Entorno de ejecución

Configuración comprobada en el proyecto:

| Componente | Versión |
|---|---|
| Python | 3.9.19 |
| PyTorch | 2.2.2 |
| Torchvision | 0.17.2 |
| CUDA de PyTorch | 12.1 |
| CUDA Toolkit (nvcc) | 12.1 |
| GCC | 12 |
| NumPy | 1.26.4 |
| Open3D | 0.18.0 |
| Mediapy | 1.2.4 |

Hardware utilizado: NVIDIA GeForce RTX 4060 Laptop GPU con 8 GB de VRAM.

## 2. Crear el entorno

Desde la raíz del repositorio:

```bash
conda env create -f environments/qgs/environment.yml
conda activate QGS
```

Si el entorno ya existe, basta con activarlo:

```bash
conda activate QGS
```

## 3. Preparar CUDA y el compilador

Se utiliza CUDA 12.1 y GCC 12 para compilar las extensiones personalizadas.

Instalar las cabeceras CCCL necesarias:

```bash
conda install -c conda-forge cuda-cccl -y
```

Configurar las variables de entorno:

```bash
export CC=gcc-12
export CXX=g++-12

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$CONDA_PREFIX/include/cccl${CPATH:+:$CPATH}
```

Comprobar las versiones:

```bash
gcc-12 --version
g++-12 --version
nvcc --version
```

## 4. Compilar los módulos CUDA

QGS utiliza dos extensiones CUDA personalizadas:

- `diff-quadratic-rasterization`
- `simple-knn`

Desde la raíz del proyecto:

```bash
cd QGS

pip install ./submodules/diff-quadratic-rasterization --no-build-isolation
pip install ./submodules/simple-knn --no-build-isolation
```

La opción `--no-build-isolation` permite utilizar las dependencias del entorno activo, especialmente PyTorch, durante la compilación.

## 5. Verificar la instalación

Comprobar que PyTorch reconoce CUDA y que los módulos personalizados se pueden importar:

```bash
python - <<'PY'
import torch
import torchvision
import numpy as np
import open3d as o3d
import mediapy

from diff_quadratic_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch no reconoce una GPU CUDA")

print("GPU:", torch.cuda.get_device_name(0))

points = torch.rand(100, 3, device="cuda")
distances = distCUDA2(points)

print("simple-knn:", distances.shape)
print("Módulos CUDA importados correctamente")
PY
```

Finalmente, comprobar las dependencias del entorno:

```bash
python -m pip check
```

## 6. Dependencia opcional: LPIPS

El paquete `lpips` no está incluido en el entorno principal porque no es necesario para el benchmark geométrico de este proyecto.

Sin embargo, el archivo `QGS/metrics.py` lo utiliza para calcular métricas de calidad de imagen como LPIPS, SSIM y PSNR.

Para ejecutar ese procedimiento adicional, instalar:

```bash
pip install lpips==0.1.4
```

## 7. Consideraciones

- Mantener un entorno independiente para QGS debido a sus requisitos específicos de PyTorch y CUDA.
- Compilar las extensiones CUDA dentro del entorno `QGS` después de instalar sus dependencias.
- No es necesario instalar LPIPS para ejecutar el benchmark geométrico de DTU.
- Los conjuntos de datos, checkpoints y resultados de entrenamiento se almacenan fuera de este directorio y no forman parte de la configuración del entorno.
