# Réplica Experimental: SSCAD 2025 (LLVM MPP vs MPI+OpenMP)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Platform: Linux x86_64](https://img.shields.io/badge/Platform-Linux%20x86__64-orange.svg)](#1-arquitectura-del-clúster-de-pruebas)
[![Accelerator: NVIDIA RTX A1000](https://img.shields.io/badge/GPU-NVIDIA%20RTX%20A1000-green.svg)](#1-arquitectura-del-clúster-de-pruebas)

Este repositorio contiene el kit autocontenido de replicación, orquestación, datasets experimentales y análisis del artículo de investigación:

> **"Can OpenMP Scale Beyond the Node? A Performance Evaluation of Remote Offloading via the MPI Proxy Plugin"**  
> *Jhonatan Cléto, Guilherme Valarini, Hervé Yviquel*  
> Universidade Estadual de Campinas (UNICAMP) — Simpósio em Sistemas Computacionais de Alto Desempenho (**SSCAD 2025**).  
> Paper original disponible en [`paper/MPP_Paper.pdf`](paper/MPP_Paper.pdf).

---

## 1. Arquitectura del Clúster de Pruebas

Mientras que el paper original evaluó supercomputadores de alta gama (*Santos Dumont 2* con NVIDIA H100 SXM y AMD Instinct MI300A sobre InfiniBand a 400 Gbps), esta réplica evalúa la viabilidad y sobrecostos de **LLVM MPP** en un clúster físico compacto de tipo Mini-PC (Edge/Workstation HPC):

| Parámetro | Supercomputador Santos Dumont 2 (Paper) | Clúster Replicado (Este Repositorio) |
| :--- | :--- | :--- |
| **Topología** | 2x Nodos Supercomputador | **10x Lenovo ThinkStation P3 Tiny (Gen 2)** |
| **CPU** | Intel Xeon M9468 / AMD APU Zen 4 | **Intel Core Ultra 9 285K/T** (24C / 24T) |
| **Memoria RAM** | 512 GB / HBM3 | **32 GB DDR5 por nodo** |
| **Aceleradores (GPU)** | NVIDIA H100 SXM (80 GB) / AMD MI300A (128 GB) | **10x NVIDIA RTX A1000 Laptop GPU (4 GB VRAM, sm_86)** |
| **Red de Interconexión** | InfiniBand NDR (400 Gbps) / RDMA | **Ethernet Local Gigabit (1 Gbps)** |
| **Rendimiento Pico FP32**| ~60 TFLOP/s (H100) / ~45 TFLOP/s (MI300A) | **~4.42 TFLOP/s FP32 medido por nodo** (~44.2 TFLOP/s clúster) |
| **Pila de Software** | Clang 21.0, MPICH 4.3, UCX 1.18, CUDA 12.8 | Clang 21.0 (MPP patched), OpenMPI 4.1.6, CUDA 12.0 |

- **Nodo Maestro / Orquestador:** `node11` (`alumno11`)
- **Nodos Esclavos / Workers:** `node12` al `node20` (`alumno12` al `alumno20`)

---

## 2. Estructura del Repositorio

```text
replica_mpp_paper/
├── README.md                          # Guía operativa y técnica principal (este archivo)
├── LICENSE                            # Licencia GNU General Public License v3.0
├── requirements.txt                   # Dependencias de Python (pandas, matplotlib, seaborn, etc.)
│
├── paper/
│   └── MPP_Paper.pdf                  # PDF del artículo científico original de SSCAD 2025
│
├── config/
│   ├── cluster_specs.json             # Ficha técnica JSON del clúster (CPU, GPU, RAM, Red)
│   ├── nodes.conf                     # Tabla de hostnames, IPs y credenciales de los 10 nodos
│   └── openmpi-mca-params.conf        # Parámetros MCA de OpenMPI para aislamiento de interfaces
│
├── scripts/
│   ├── setup_cluster_ips.sh           # Automatización de /etc/hosts, SSH sin contraseña y validación
│   ├── deploy_cluster.sh              # Despliegue/sincronización de /opt/llvm-mpp a todos los workers
│   ├── run_experiments.py             # Runner de barridos (weak, strong, comm) tolerante a fallos
│   ├── process_results.py             # Parser de trazas JSONL a .pkl y generación de figuras
│   └── utils.py                       # Funciones de cálculo de METG e interpolación gráfica
│
├── patches/
│   ├── llvm_mpp_fixes.patch           # Corrección del bug de doble registro en libomptarget
│   ├── task_bench_fixes.patch         # Kernel compute_bound_gpu y flags en Task Bench
│   ├── patch_tb.py                    # Script de aplicación automática de parches en Task Bench
│   └── README_PATCHES.md              # Documentación técnica exhaustiva de los bugs solventados
│
├── results/
│   ├── smoke_test.jsonl               # Prueba de humo inicial
│   ├── tb_wscaling_stencil_5s.jsonl   # Traza cruda de 540 corridas de weak scaling en RTX A1000
│   ├── tb_wscaling_stencil_5s_mpi_a1000_03_11_26.pkl  # Dataset procesado MPI (RTX A1000)
│   ├── tb_wscaling_stencil_5s_mpp_a1000_03_11_26.pkl  # Dataset procesado MPP (RTX A1000)
│   ├── tb_wscaling_stencil_5s_*_h100_*.pkl            # Datasets de comparación H100 (original)
│   ├── tb_wscaling_stencil_5s_*_mi300a_*.pkl          # Datasets de comparación MI300A (original)
│   └── original_paper_datasets/       # Colección íntegra de datasets .pkl del paper original
│
├── figures/
│   ├── a1000_granularity_metg_devices_1.pdf   # METG con 1 GPU
│   ├── a1000_granularity_metg_devices_2.pdf   # METG con 2 GPUs
│   ├── a1000_granularity_metg_devices_5.pdf   # METG con 5 GPUs
│   ├── a1000_granularity_metg_devices_10.pdf  # METG con 10 GPUs
│   └── a1000_metg_vs_devices.pdf              # Escalado del METG de 1 a 10 dispositivos
│
└── docs/
    ├── INFORME_REPLICACION_SSCAD2025.md # Informe técnico detallado con análisis y conclusiones
    ├── analisis_interactivo.ipynb     # Jupyter Notebook para visualización y cómputo interactivo
    └── MPP_Paper.pdf                  # Copia local del paper de referencia
```

---

## 3. Instalación de Dependencias

Para procesar datos y generar las figuras en local:

```bash
git clone git@github.com:Iv4n0f/replica_mpp_paper.git
cd replica_mpp_paper
pip install -r requirements.txt
```

---

## 4. Guía Rápida de Operación

### 4.1. Configuración de Red del Clúster
Si las IPs del laboratorio cambian:
1. Editar `config/nodes.conf` con las IPs actuales (`<hostname> <ip> <usuario> <contraseña>`).
2. Ejecutar el script automatizado:
   ```bash
   ./scripts/setup_cluster_ips.sh
   ```
   *Actualiza `/etc/hosts` en los 10 nodos, propaga llaves SSH y ejecuta un smoke test distribuido con `mpirun`.*

### 4.2. Sincronizar Binarios y Runtime
```bash
./scripts/deploy_cluster.sh
```

### 4.3. Ejecutar los Experimentos de Escalabilidad
Conectarse al nodo maestro (`node11`) y lanzar el barrido:
```bash
ssh alumno11@node11 "nohup python3 -u ~/run_experiments.py --experiment weak --trials 3 > ~/results/weak_scaling.log 2>&1 &"
ssh alumno11@node11 "tail -f ~/results/weak_scaling.log"
```

> **Tolerancia a Fallos:** El runner sincroniza con `fsync` tras cada ensayo individual. Si se interrumpe, se reanuda automáticamente omitiendo los puntos ya completados.

### 4.4. Generar Figuras Comparativas
Una vez descargada la traza `.jsonl`:
```bash
python3 scripts/process_results.py --jsonl results/tb_wscaling_stencil_5s.jsonl
```
El script generará automáticamente los archivos `.pkl` y las figuras vectoriales (`.pdf`) en `figures/`, comparando el clúster RTX A1000 contra los datos de NVIDIA H100 y AMD MI300A provistos por los autores.

---

## 5. Resumen de Hallazgos Experimentales

### 5.1. Rendimiento Pico FP32 en RTX A1000
- **Elapsed Time:** 9.94 s para 20 pasos de stencil 1D (`iter = 524288`, `size = 0`).
- **TFLOP/s sostenidos:** **4.42 TFLOP/s FP32** por nodo (100% de eficiencia de hardware sobre arquitectura GA107 con 16 SMs).
- **Rendimiento agregado a 10 nodos:** **41.5 TFLOP/s sostenidos**.

### 5.2. Comparativa de METG(50%)

| Configuración | METG(50%) en H100 (InfiniBand) | METG(50%) en MI300A (InfiniBand) | METG(50%) en RTX A1000 (Ethernet) |
| :--- | :---: | :---: | :---: |
| **MPI + OpenMP** | **0.81 ms** | **1.63 ms** | **~0.65 ms** |
| **MPI Proxy Plugin (MPP)** | **2.22 ms** | **1.34 ms** | **~7.88 ms** |

**Conclusiones Clave:**
1. **Transparencia vs Sobrecosto:** LLVM MPP logra orquestar exitosamente las 10 GPUs distribuidas en 10 máquinas a partir de un único proceso OpenMP sin escribir código de red MPI explícito.
2. **Impacto de la Red:** En redes Ethernet convencionales (sin RDMA), la sobrecarga de roundtrip de los proxies MPP sitúa el umbral de granularidad efectiva en **$\ge 10\text{ ms}$**. Para tareas con duración superior a 10 ms, MPP escala casi al 100% del rendimiento pico de la GPU.
3. Para una discusión exhaustiva de la metodología, gráficas e implicancias técnicas, consulte [`docs/INFORME_REPLICACION_SSCAD2025.md`](docs/INFORME_REPLICACION_SSCAD2025.md).

---

## 6. Licencia y Cita

El código y artefactos de este kit están licenciados bajo **GNU General Public License v3.0** ([LICENSE](LICENSE)).

Si utilizas este trabajo o los parches provistos, por favor cita el artículo original:

```bibtex
@inproceedings{cleto2025mpp,
  author    = {Jhonatan Cl{\'{e}}to and Guilherme Valarini and Herv{\'{e}} Yviquel},
  title     = {Can OpenMP Scale Beyond the Node? A Performance Evaluation of Remote Offloading via the MPI Proxy Plugin},
  booktitle = {Simp{\'{o}}sio em Sistemas Computacionais de Alto Desempenho (SSCAD 2025)},
  year      = {2025},
  publisher = {SBC}
}
```
