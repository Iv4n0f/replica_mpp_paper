# Bitácora Técnica y Post-Mortem: Fase 1 (OpenMPI) y Reestructuración a MPICH + UCX

- **Fecha de Ejecución Fase 1:** 03 de Noviembre de 2026
- **Entorno de Pruebas:** Clúster de 10 nodos Lenovo ThinkStation P3 Tiny (Gen 2)
- **Hardware:** Intel Core Ultra 9 285K/T (24C/24T), 32 GB RAM DDR5, NVIDIA RTX A1000 Laptop GPU (4 GB GDDR6, sm_86)
- **Red:** Ethernet Gigabit (1 Gbps) sobre interfaz `wlp129s0f0` (TCP/IP)
- **Compilador:** LLVM / Clang 21.0 con soporte MPI Proxy Plugin (MPP)

---

## 1. Resumen Ejecutivo de la Incidencia

Durante la primera campaña experimental de réplica del artículo científico *"Can OpenMP Scale Beyond the Node? A Performance Evaluation of Remote Offloading via the MPI Proxy Plugin"* (SSCAD 2025), se ejecutó un barrido completo de **540 corridas de Weak Scaling** (Stencil 1D, Radix 5, granularidades de 2 a 18, 1 a 10 nodos) utilizando **OpenMPI 4.1.6**.

Los resultados mostraron dos anomalías severas:
1. **Serialización Forzada en MPP:** El tiempo de ejecución en granularidad gruesa (`size = 2`) escaló linealmente con el número de nodos ($10.2\text{ s} \to 25.3\text{ s} \to 69.1\text{ s} \to 140.5\text{ s}$), provocando que el rendimiento agregado se estancara en ~3.13 TFLOP/s (la capacidad de una sola GPU) en lugar de alcanzar los ~35 TFLOP/s esperados a 10 nodos.
2. **Indeterminación en METG(50%):** La métrica de Granularidad Mínima Efectiva al 50% de eficiencia retornó **`NaN`** para 2, 3, ..., 10 dispositivos en MPP, ya que la eficiencia relativa al clúster nunca superó el 46.2% (cayendo al 10.1% a 10 nodos).

Esta bitácora documenta la cadena causal de eventos que condujo al uso de OpenMPI, la causa raíz del cuello de botella y la justificación técnica de la reestructuración hacia **MPICH 4.3 + UCX 1.18**.

---

## 2. El Origen: Bug Oficial de MPICH en Ubuntu 24.04 LTS

Originalmente se intentó desplegar el experimento utilizando la distribución oficial de MPICH provista por Canonical en Ubuntu 24.04 LTS (`apt install mpich`). Sin embargo, el experimento no pudo iniciar debido a un fallo crítico upstream:

- **Identificador Oficial:** [Ubuntu Launchpad Bug #2072338](https://bugs.launchpad.net/ubuntu/+source/mpich/+bug/2072338)
- **Paquete Afectado:** `mpich 4.2.0-5build3` en Ubuntu 24.04 (Noble Numbat).
- **Descripción del Fallo:**
  Un defecto en el empaquetado y en la inicialización de grupos de procesos MPMD (*Multiple Program, Multiple Data*) en el gestor de procesos Hydra provocaba que la llamada `MPI_Init` / `MPI_COMM_WORLD` fallara en asignar los rangos globales.
  Como resultado:
  - `MPI_Comm_rank` retornaba **`0` en todos los procesos** de todos los nodos.
  - `MPI_Comm_size` retornaba **`1`** independientemente del número de nodos y procesos especificados en `mpirun`.
- **Impacto en MPP:**
  El runtime de LLVM MPP requiere lanzar una topología MPMD heterogénea:
  ```bash
  mpirun -np 10 llvm-offload-mpi-proxy-device : -np 1 tb_mpp
  ```
  Al retornar todos los procesos el rango 0, los 10 proxies y el proceso host colisionaban en la inicialización de los comunicadores de red, provocando *deadlocks* estáticos o abortos inmediatos del runtime.

---

## 3. La Medida de Contingencia Adoptada (Workaround de Fase 1)

Ante la imposibilidad de arrancar con el paquete MPICH de Ubuntu, se tomó la decisión operativa de migrar la pila a **OpenMPI 4.1.6**, disponible en los repositorios oficiales (`openmpi-bin`, `libopenmpi-dev`):

1. OpenMPI soportaba la sintaxis de lanzamiento heterogéneo MPMD con `:` y el flag `--oversubscribe`.
2. Se configuraron parámetros MCA (`config/openmpi-mca-params.conf`) para aislar la interfaz de red local `wlp129s0f0` y evitar colisiones con interfaces virtuales de Docker o puentes de red.
3. Se parcheó Task Bench (`task_bench_fixes.patch`) para añadir una guarda de inicialización `MPI_Initialized(&initialized)` antes de `MPI_Init`.

Con este entorno, el clúster completó exitosamente las 540 ejecuciones sin caídas. Sin embargo, los datos arrojaron las anomalías de rendimiento.

---

## 4. Diagnóstico Técnico y Causa Raíz (Post-Mortem)

El análisis exhaustivo de los resultados reveló que el cuello de botella fue provocado por la combinación de tres factores:

### A. El Candado Global de OpenMPI en Modo Multi-Thread (`ompi_mutex`)
En OpenMPI (cuando no está respaldado por hardware con soporte RDMA/UCX dedicado), la capa de transporte BTL sobre sockets TCP (`btl_tcp`) implementa un **cerrojo global de exclusión mutua**.

En el kernel de Task Bench (`core_kernel.cc`):
```cpp
#pragma omp target teams distribute parallel for device(device_id) map(tofrom: sink)
```
Al no contar con la cláusula `nowait`, cada hilo del host entra en una llamada síncrona bloqueante:
1. El Hilo 0 envía la tarea a la GPU 0 por TCP y entra a `MPI_Wait` esperando que la GPU termine de computar y devuelva `sink`.
2. Durante los **~14 segundos** que dura el cómputo de la GPU 0, el Hilo 0 **retiene el candado global del motor OpenMPI**.
3. El Hilo 1 (que debe despachar a la GPU 1) queda congelado en la cola del candado en la CPU.
4. Solo cuando la GPU 0 termina y el mensaje TCP de retorno es procesado, el Hilo 1 puede enviar su tarea a la GPU 1, y así sucesivamente.
5. El tiempo total para 10 nodos resulta en:
   $$\text{Tiempo Total} = 10 \text{ nodos} \times 14\text{ s} \approx \mathbf{140.5\text{ segundos}}$$

### B. La Advertencia Textual del Paper Original
Al revisar el texto completo del artículo de SSCAD 2025 (*Section 3.3 Experiments, página 5*), los propios autores advirtieron explícitamente sobre este comportamiento:

> *"Experiments used **only MPICH**, as other MPI libraries like **OpenMPI showed inferior performance across platforms and benchmarks**."*

Los autores originales ya habían detectado que OpenMPI serializaba las operaciones internas del plugin MPP, por lo que descartaron su uso y basaron todo su estudio en **MPICH 4.3 + UCX 1.18**.

---

## 5. Datos Preservados de la Fase 1

Todos los artefactos, trazas y figuras generadas durante la Fase 1 se han archivado íntegramente en esta subcarpeta para trazabilidad científica y comparativa en el informe académico:

- **Traza Cruda JSONL:** [`results/tb_wscaling_stencil_5s.jsonl`](results/tb_wscaling_stencil_5s.jsonl) (540 ejecuciones validadas).
- **Test Preliminar:** [`results/smoke_test.jsonl`](results/smoke_test.jsonl).
- **Datasets Pandas:** [`results/tb_wscaling_stencil_5s_mpi_a1000_03_11_26.pkl`](results/tb_wscaling_stencil_5s_mpi_a1000_03_11_26.pkl) y [`results/tb_wscaling_stencil_5s_mpp_a1000_03_11_26.pkl`](results/tb_wscaling_stencil_5s_mpp_a1000_03_11_26.pkl).
- **Figuras Vectoriales con Anomalía:**
  - [`figures/a1000_metg_vs_devices.pdf`](figures/a1000_metg_vs_devices.pdf) (Curva de MPP ausente a partir de 2 dispositivos debido a `NaN`).
  - [`figures/a1000_granularity_metg_devices_1.pdf`](figures/a1000_granularity_metg_devices_1.pdf) (1 nodo: eficiencia normal ~98%).
  - [`figures/a1000_granularity_metg_devices_10.pdf`](figures/a1000_granularity_metg_devices_10.pdf) (10 nodos: eficiencia estancada en 10.1%, por debajo de la línea roja del 50%).

---

## 6. Plan de Acción: Reestructuración a MPICH 4.3 + UCX 1.18 (Fase 2)

Para eliminar definitivamente la serialización y validar el paralelismo real de MPP en el clúster Lenovo P3 Tiny:

1. **Bypass del Bug de Ubuntu:** Compilar directamente el tarball oficial upstream **MPICH 4.3.0** (`mpich.org`), el cual no posee el fallo del empaquetado de Canonical.
2. **Motor Asíncrono Lock-Free:** Compilar **UCX 1.18.0** con `--with-cuda` y configurar MPICH con `--with-device=ch4:ucx --enable-threads=multiple`. Esto permite que los hilos del host despachen concurrentemente sin cerrojos globales en la capa de red.
3. **Automatización:** Ejecución mediante el script [`scripts/setup_mpich_ucx.sh`](../scripts/setup_mpich_ucx.sh).
4. **Punto de Control Obligatorio:** Detener el procedimiento tras la prueba de humo de 10 GPUs (Paso 3) para confirmar la concurrencia y validación del usuario antes de lanzar el barrido masivo.
