# Entorno de VCR-GauS

Este directorio contiene la configuración del entorno utilizado para ejecutar **VCR-GauS (View-Consistent Depth-Normal Regularizer for Gaussian Surface Reconstruction)** en el benchmark con el conjunto de datos DTU.

Se utiliza un entorno independiente de QGS debido a las diferencias en las versiones de Python, PyTorch y CUDA requeridas por ambos métodos.

## 1. Entorno de ejecución

Configuración utilizada en los experimentos:

| Componente | Versión |
|---|---|
| Python | 3.10.22 |
| PyTorch | 2.0.1+cu118 |
| Torchvision | 0.15.2+cu118 |
| Torchaudio | 2.0.2+cu118 |
| CUDA Toolkit (nvcc) | 11.8.89 |
| GCC | 11 |
| NumPy | 1.26.1 |
| Open3D | 0.18.0 |
| PyTorch3D | 0.7.4 |
| LPIPS | 0.1.4 |

Hardware utilizado: NVIDIA GeForce RTX 4060 Laptop GPU con 8 GB de VRAM.

## 2. Crear el entorno

Desde la raíz del repositorio:

```bash
conda env create -f environments/vcr_gaus/environment.yml
conda activate vcr
```

Si el entorno ya existe:

```bash
conda activate vcr
```

## 3. Instalar las dependencias de entrenamiento

Ingresar al repositorio de VCR-GauS:

```bash
cd VCR-GauS
```

Instalar el paquete y las dependencias declaradas en el extra `train` de `pyproject.toml`:

```bash
pip install -e ".[train]" --no-build-isolation
```

Esta instalación incluye las dependencias utilizadas durante el entrenamiento, entre ellas Open3D, LPIPS, Trimesh, PyMeshLab, OpenCV, WandB, ImageIO y scikit-image.

## 4. Instalar PyTorch3D

El archivo `requirements.txt` del repositorio incluye PyTorch3D como dependencia externa.

Instalarlo desde su repositorio oficial:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

La versión utilizada en el entorno experimental fue PyTorch3D 0.7.4.

## 5. Preparar los submódulos y el compilador

En una clonación limpia del repositorio, inicializar los submódulos:

```bash
git submodule update --init --recursive
```

Se requiere GCC 11 para compilar las extensiones CUDA:

```bash
export CC=gcc-11
export CXX=g++-11
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
```

Comprobar las versiones:

```bash
gcc-11 --version
g++-11 --version
nvcc --version
```

## 6. Compilar las extensiones CUDA

VCR-GauS utiliza dos módulos CUDA personalizados:

- `diff-gaussian-rasterization`
- `simple-knn`

Desde el directorio `VCR-GauS`:

```bash
pip install ./submodules/diff-gaussian-rasterization --no-build-isolation

pip install ./submodules/simple-knn --no-build-isolation
```

La opción `--no-build-isolation` permite utilizar las dependencias del entorno activo durante la compilación.

## 7. Verificar la instalación

Ejecutar el siguiente código para comprobar PyTorch, CUDA y las extensiones compiladas:

```bash
python - <<'PY'
import torch
import torchvision
import numpy as np
import open3d as o3d
import pytorch3d

from diff_gaussian_rasterization import GaussianRasterizer
from simple_knn._C import distCUDA2

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
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

Comprobar también las dependencias:

```bash
python -m pip check
```

## 8. Consideraciones

- Mantener VCR-GauS en un entorno independiente del utilizado para QGS.
- Utilizar CUDA 11.8 y GCC 11 para compilar las extensiones.
- Inicializar los submódulos en clonaciones limpias antes de compilar.
- No ejecutar actualizaciones de submódulos sobre una copia con modificaciones locales sin revisar previamente sus cambios.
- Los conjuntos de datos, checkpoints, registros y resultados de entrenamiento se almacenan fuera de este directorio.
