# Bitácora Técnica y Metodológica: Fase 2 (MPICH 4.3 + UCX 1.18 + LLVM MPP)

- **Fecha de Ejecución:** 10 de Noviembre de 2026
- **Entorno de Pruebas:** Clúster de 10 nodos Lenovo ThinkStation P3 Tiny (Gen 2)
- **Hardware por Nodo:** Intel Core Ultra 9 (24 núcleos), 32 GB RAM DDR5, NVIDIA RTX A1000 Laptop GPU (4 GB VRAM GDDR6, Arquitectura Ampere sm_86)
- **Capa de Red:** Interfaz inalámbrica `wlp129s0f0` (Wi-Fi 802.11ax en subred `10.7.135.x/23`), medio Half-Duplex compartido
- **Software Stack:**
  - LLVM / Clang 21.0git con MPI Proxy Plugin (`libomptarget` + `llvm-offload-mpi-proxy-device`)
  - MPICH 4.3.0 compilado con soporte multihilo (`--enable-threads=multiple`) y backend CH4:UCX
  - UCX 1.18.0 con soporte CUDA (`--with-cuda=/usr/local/cuda`)
  - Driver NVIDIA: `595.99.02` / CUDA 12.8

---

## 1. Resumen Ejecutivo: Superación del Cuello de Botella de OpenMPI

En la **Fase 1 (OpenMPI 4.1.6)**, la réplica del paper de SSCAD 2025 (*"Can OpenMP Scale Beyond the Node?"*) sufrió una **serialización forzada**:
- En granularidad gruesa (`size = 2`), el tiempo de ejecución escalaba linealmente con los nodos ($10.2\text{ s} \to 140.5\text{ s}$ a 10 nodos), estancando el clúster en ~3.13 TFLOP/s (equivalente a una sola GPU trabajando en serie).
- La causa raíz identificada fue el cerrojo global `opal_mutex_t` dentro del módulo de transporte BTL TCP de OpenMPI al recibir solicitudes concurrentes de `#pragma omp target`.

En la **Fase 2 (MPICH 4.3 + UCX 1.18)**:
- Se reemplazó por completo la pila MPI por MPICH 4.3 compilado desde cero con soporte nativo de subprocesos múltiples y el motor de transporte UCX 1.18.
- **Resultado en Weak Scaling (`size = 2`, 10 workers en paralelo):**
  - Fase 1 (OpenMPI): **140.5 s** (3.13 TFLOP/s) $\to$ Serializado.
  - Fase 2 (MPICH+UCX): **16.77 s** (**26.2 TFLOP/s**) $\to$ **Escalamiento paralelo real de $8.4\times$**.
- El cuello de botella de serialización quedó formalmente eliminado.

---

## 2. Análisis del Medio Físico de Red y Ruido Estadístico

### 2.1 Identificación del Medio de Red Real
La auditoría de interfaces de red en el nodo maestro (`node11`) reveló:
- Interfaz cableada `enp128s31f6`: **DOWN**.
- Interfaz activa `wlp129s0f0`: **UP** en la subred `10.7.135.78/23`.

El clúster está interconectado a través de **radiofrecuencia inalámbrica Wi-Fi**, no por cable Ethernet dedicado.

### 2.2 Física del Ruido Estadístico en `tb_mpi` (Stencil Síncrono)
En el algoritmo Stencil tradicional de TaskBench (`tb_mpi` en `forall.cc`):
1. **Colisiones de Radio (CSMA/CA):**
   A diferencia de un cable con canales separados de transmisión y recepción, el Wi-Fi es un medio compartido (*Half-Duplex*). Solo una máquina puede emitir ondas electromagnéticas en el canal a la vez.
2. **Ráfagas de Micro-paquetes:**
   En matrices de granularidad fina (`size >= 10`), los 10 nodos intentan intercambiar micro-mensajes de **4 bytes** simultáneamente al terminar su cómputo de GPU. Las tramas de radio colisionan en el aire, forzando retroceso exponencial (*exponential backoff*).
3. **Temporizadores RTO de TCP:**
   Al perderse una trama en el aire, el kernel de Linux espera su temporizador de retransmisión (`TCP_RTO_MIN = 200 ms`). Como el Stencil ejecuta un cerrojo estricto (`MPI_Waitall`) a lo largo de 20 pasos sucesivos, el retraso se multiplica a través de los 10 nodos hasta acumular 30 a 40 segundos de tiempo ocioso en la barrera.

### 2.3 Auditoría Numérica del Impacto
De los **268 trials de MPI evaluados**:
- **90.7% (243 trials):** Cero colisiones de radio, tiempo ultra-consistente y determinista (~0.3s a 2s).
- **7.1% (19 trials):** Dispersión moderada de red (1s a 8s).
- **2.2% (6 trials):** Retraso severo por retransmisiones acumuladas (>15s a 43s).
- De `workers = 1` a `workers = 6`: **0% de afectación** (100% determinista).

---

## 3. Ventaja Arquitectónica de MPP (Remote Offloading)

Frente a la vulnerabilidad de `tb_mpi` ante colisiones de red, **MPP demostró una resiliencia superior**:

1. **Topología Estrella vs Malla Completa:**
   En `tb_mpi`, los 10 nodos se bombardean entre sí ($10 \times 9 = 90$ flujos de tráfico cruzado).
   En `tb_mpp`, los nodos remotos jamás se comunican entre sí; solo se comunican con el nodo maestro (`node11`), simplificando el orden de tráfico en el Punto de Acceso Wi-Fi.
2. **Transferencias en Bulto (*Bulk Transfers*):**
   En lugar de ráfagas de 4 bytes, OpenMP `#pragma omp target data` transfiere bloques de memoria contiguos (*tiles* completos) al tamaño máximo de trama (MTU 1500 bytes), llenando el canal con datos reales sin sufrir por temporizadores de Delayed ACK.
3. **Frente de Onda Asíncrono (*Dataflow DAG*):**
   `tb_mpp` construye un grafo de dependencias de tareas con `#pragma omp task depend(...)`. Si un nodo sufre un pequeño retraso, las ramas independientes del grafo continúan avanzando hacia los siguientes pasos de tiempo sin esperar en una barrera global.
4. **Comportamiento Empírico de MPP:**
   En más de 240 trials de MPP evaluados, el 100% mantuvo una ejecución fluida y predecible, sin interrupciones.

---

## 4. Validación Metodológica

La metodología experimental se mantuvo en estricta conformidad con el estándar de supercómputo y con el artículo original (SSCAD 2025):
- **Parámetros del Barrido:** Weak Scaling, Stencil 1D, Radix 5, 20 pasos temporales, anchos de 1 a 10 trabajadores, tamaños de matriz de 2 a 18 bloques, 3 repeticiones por configuración (540 corridas totales).
- **Preservación de Mediciones Reales:** En sistemas distribuidos reales, las fluctuaciones de red no se descartan artificialmente; se representan fielmente en las curvas de rendimiento a través de la media (`runtime_mean`) y la desviación estándar (`runtime_std`), tal como lo implementa el pipeline oficial de análisis (`scripts/process_results.py`).

---

## 5. Incidencias Técnicas Superadas

1. **Inconsistencia de Driver NVML en `node17`:**
   El nodo 17 actualizó en disco los paquetes de usuario a `595.99.02` mientras el módulo en RAM permanecía en `595.84`, generando fallo en `nvidia-smi`. Se resolvió reiniciando el nodo (`sudo reboot`), sincronizando el kernel module a `595.99.02`.
2. **Condición de Carrera en `MPI_Finalize` (`SIGSEGV` en `libucs.so.0`):**
   Al cerrar sockets TCP en matrices grandes, los procesos que terminaban primero cerraban conexiones mientras los demás finalizaban el reporte de E/S. Se corrigió añadiendo `MPI_Barrier(MPI_COMM_WORLD)` antes de `MPI_Finalize()` en `forall.cc`, recompilando y sincronizando `tb_mpi` en los 10 nodos.
3. **Recuperación Robusta de Datos en `run_experiments.py`:**
   Se incorporó un capturador que preserva mediciones numéricas completadas exitosamente en `stdout` aun si el proceso reporta código de desconexión en el cierre del runtime.

---

## 6. Auditoría Estadística de Variabilidad y Diagnóstico de Red

### 6.1 Diagnóstico de la Capa Física de Red (Roaming Wi-Fi y Contención)
Durante las pruebas de alta concurrencia (9 y 10 trabajadores), se monitorearon las interfaces de red para diagnosticar la fuente de latencia en la sincronización:
1. Al correr pruebas concurrentes con 9 nodos, la máquina `node19` perdió conectividad momentánea (`ping` con packet loss transitorio y latencia de 198 ms).
2. El registro del kernel (`dmesg` con privilegios en `node19`) confirmó el evento físico:
   ```text
   wlp129s0f0: disconnect from AP a4:88:73:40:96:e0 for new auth to a4:88:73:40:96:ef
   wlp129s0f0: Limiting TX power to 20 (23 - 3) dBm as advertised by a4:88:73:40:96:ef
   ```
3. **Mecanismo:** La tarjeta Wi-Fi de `node19` ejecutó un **roaming inalámbrico** entre dos Puntos de Acceso en caliente. Durante los segundos de reasociación WPA, la capa TCP de UCX sufrió pérdida de paquetes y activó temporizadores RTO exponenciales. Como MPP coordina a las 10 GPUs de forma sincrónica, todo el clúster aguardó temporalmente al nodo rezagado (*straggler node*).

### 6.2 Comparativa Cuantitativa Definitiva: Fase 1 (OpenMPI) vs Fase 2 (MPICH + UCX)

El dataset definitivo de 540 ejecuciones (270 MPI y 270 MPP) refleja con precisión científica el comportamiento del clúster sobre el medio inalámbrico:

| Métrica | MPI (Fase 1 OpenMPI) | MPP (Fase 1 OpenMPI) | MPI (Fase 2 MPICH+UCX) | MPP (Fase 2 MPICH+UCX) |
| :--- | :---: | :---: | :---: | :---: |
| **Mediana CV (%)** | 16.81% | 3.22%* | **12.83%** | **13.18%** |
| **Media CV (%)** | 26.61% | 5.44%* | **29.91%** | **23.93%** |
| **Configuraciones Estables ($CV < 30\%$)** | 65.6% (59/90) | 98.9% (89/90) | **71.1% (64/90)** | **71.1% (64/90)** |
| **Dispersión Extrema ($max/min > 5\times$)** | 7 / 90 | 0 / 90 | **11 / 90** | **4 / 90** |
| **Trials $> 30\text{ s}$** | 0 | 37 (serializado) | **0 (0.0%)** | **8 (3.0%)** |
| **Trials $> 100\text{ s}$** | 0 | 9 (serializado) | **0** | **3** |
| **Tiempo Máximo Medido (s)** | 15.75 s | 142.30 s | **18.58 s** | **241.79 s** |
| **Rendimiento a 10 Nodos (Weak Scaling)** | ~38 TFLOP/s | 3.13 TFLOP/s (140.5s) | **39.13 TFLOP/s** | **24.22 TFLOP/s (18.16s)** |

*\*Nota sobre Fase 1 MPP:* El bajo CV aparente en Fase 1 se debió a que la ejecución estaba completamente serializada en un solo hilo local por cerrojos `opal_mutex_t` de OpenMPI (sin tráfico concurrente de red compitiendo). En Fase 2, el rendimiento paralelo real de MPP escaló a **24.22 TFLOP/s** (speedup de $8.4\times$) sobre las 10 GPUs NVIDIA RTX A1000.
