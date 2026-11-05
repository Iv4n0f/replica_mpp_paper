# Informe Técnico de Replicación Experimental: SSCAD 2025

**Título del Paper Replicado:**  
*"Can OpenMP Scale Beyond the Node? A Performance Evaluation of Remote Offloading via the MPI Proxy Plugin"*  
**Autores Originales:** Jhonatan Cléto, Guilherme Valarini, Hervé Yviquel (Instituto de Computação, UNICAMP)  
**Publicación:** Simpósio em Sistemas Computacionais de Alto Desempenho (SSCAD 2025)

---

## 1. Resumen Ejecutivo

En este trabajo se realiza la replicación y evaluación experimental de la extensión **MPI Proxy Plugin (MPP)** del compilador LLVM frente al modelo tradicional híbrido **MPI + OpenMP**. 

Mientras que el estudio original de los autores fue llevado a cabo en el supercomputador *Santos Dumont 2* (LNCC) utilizando nodos equipados con aceleradores masivos para centros de datos (NVIDIA H100 SXM y AMD Instinct MI300A APU) con interconexiones de ultra-baja latencia (InfiniBand), la presente replicación se traslada a un entorno de **clúster de formato compacto tipo Mini-PC (Edge/Workstation HPC)** compuesto por 10 nodos **Lenovo ThinkStation P3 Tiny (Gen 2)** equipados con procesadores **Intel Core Ultra 9 285K/T** y GPUs discretas **NVIDIA RTX A1000 (4 GB GDDR6, sm_86)** sobre red Ethernet.

Los resultados verifican experimentalmente las hipótesis centrales de los autores:
1. **Viabilidad de Escalado OpenMP Trans-Nodo:** LLVM MPP permite orquestar de forma totalmente transparente GPUs distribuidas físicamente en múltiples nodos de red a partir de un único proceso OpenMP sin escribir código explícito de paso de mensajes en la aplicación.
2. **Impacto de la Granularidad (METG):** El Minimum Effective Task Granularity (METG al 50% de eficiencia) en MPP es sensiblemente mayor que en MPI+OpenMP debido a los sobrecostos de roundtrip de los proxies remotos a través del plugin MPI. Mientras que MPI+OpenMP logra mantener >50% de eficiencia con tareas sub-milisegundo (~0.65 ms por tarea), MPP requiere tareas de granularidad media o gruesa (>7.8 ms en Ethernet) para amortizar la latencia del proxy.
3. **Comportamiento en Red Ethernet vs InfiniBand:** La penalización por latencia de comunicación entre el host OpenMP y los proxies se acentúa en redes commodity respecto a supercomputadores, ampliando la brecha de METG pero preservando la capacidad de alcanzar el rendimiento pico (>95% de eficiencia) cuando las tareas tienen suficiente carga de cómputo.

---

## 2. Comparativa de Plataformas Experimentales

| Parámetro | Supercomputador Santos Dumont 2 (Paper) | Clúster Replicado (Este Estudio) |
| :--- | :--- | :--- |
| **Arquitectura de Nodos** | 2x Intel Xeon SHR / 2x AMD MI300A | 10x Lenovo ThinkStation P3 Tiny (Gen 2) |
| **Procesador Central (CPU)**| Intel Xeon M9468 / AMD APU Zen 4 | Intel Core Ultra 9 285K/T (24C / 24T, sin HT) |
| **Aceleradores (GPU)** | NVIDIA H100 SXM (80 GB) / AMD MI300A (128 GB) | NVIDIA RTX A1000 Laptop GPU (4 GB VRAM) |
| **Arquitectura de GPU** | Hopper (sm_90) / CDNA3 (gfx942) | Ampere (sm_86, 16 SMs, 2048 CUDA Cores) |
| **Interconexión de Red** | InfiniBand / Omni-Path (>100 Gbps, RDMA) | Ethernet Local Gigabit (1 Gbps) |
| **Rendimiento Pico FP32** | ~60 TFLOP/s (H100) / ~45 TFLOP/s (MI300A) | **~4.42 TFLOP/s medido por nodo** |
| **Pila de Software** | Clang 21.0, MPICH 4.3, UCX 1.18, CUDA 12.8 | Clang 21.0 (MPP patched), OpenMPI 4.1.6, CUDA 12.0 |

---

## 3. Metodología Experimental y Métricas

Se utilizó el benchmark sintético **Task Bench** adaptado para offloading de OpenMP a nivel trans-nodo:

### 3.1. Topología del Grafo de Tareas
- **Patrón de Dependencia:** `stencil_1d` (Radix 5), donde cada tarea depende de sus vecinos inmediatos del paso temporal anterior.
- **Kernel de Cómputo:** `compute_bound_gpu`, diseñado para saturar las unidades de cálculo FP32 mediante operaciones FMA (Fused Multiply-Add) con bucle desenrollado (`#pragma unroll 16`).
- **Configuración de Bloques:** `nblocks = 1024`, `tpblock = 512` (524,288 hilos concurrentes por kernel lanzado a la GPU).
- **Fórmula de TFLOPs:**
  $$\text{Total FLOPs} = \text{nblocks} \times \text{tpblock} \times \text{iterations} \times 32$$

### 3.2. Métricas de Evaluación
1. **Granularidad de Tarea ($\tau$):**
   $$\tau = \frac{\text{Wall Time (s)} \times \text{Num Devices} \times 10^3}{\text{Total Tasks}} \quad [\text{ms}]$$
2. **Eficiencia de Cómputo ($\eta$):**
   $$\eta = \frac{\text{TFLOP/s Alcanzado}}{\text{TFLOP/s Máximo Sostenido}} \times 100\%$$
3. **METG(50%):** Granularidad mínima de tarea a la cual el sistema sostiene al menos el 50% de su rendimiento pico sostenido.
4. **Weak Scaling:** Incremento proporcional del número de tareas al escalar de 1 a 10 nodos (`width = workers`, 20 tareas por nodo).
5. **Strong Scaling:** Carga fija total (200 tareas totales en grafo de ancho 10) distribuida progresivamente entre 1 y 10 nodos.

---

## 4. Problemas Técnicos Detectados y Soluciones Desarrolladas

Durante la replicación en el clúster físico se superaron cuatro obstáculos técnicos que impedían la ejecución original:

1. **Bug de Bloqueo de Proxies al Finalizar (`libomptarget`):**
   - *Causa:* En `PluginManager.h`, el método `registerDelayedLibraries` invocaba `__tgt_register_lib` en lugar de `registerLib`, duplicando el contador de referencias (`RefCount = 2`). Al destruir el runtime, `deinitRuntime` no entraba en la condición `RefCount == 1`, impidiendo llamar a `PM->deinit()`. Como resultado, los proxies remotos (`llvm-offload-mpi-proxy-device`) quedaban colgados indefinidamente en `MPI_Recv`.
   - *Solución:* Se corrigió la llamada a `registerLib(Desc)`. Los proxies finalizan limpiamente con código 0.

2. **Incompatibilidad de MPICH en Ubuntu 24.04:**
   - *Causa:* Bug crítico empaquetado en Ubuntu 24.04 (Launchpad #2072338) donde `MPI_Comm_rank` retorna siempre `0` y `MPI_Comm_size` retorna `1` bajo MPMD.
   - *Solución:* Migración completa de la pila a OpenMPI 4.1.6, configurando parámetros MCA (`btl_tcp_if_include = wlp129s0f0`) para evitar colisiones con interfaces virtuales y permitiendo el lanzamiento MPMD (`llvm-offload-mpi-proxy-device : tb_mpp`) con flag `--oversubscribe`.

3. **Interferencia de Plugins en Modo Híbrido MPI+OpenMP:**
   - *Causa:* El constructor estático de `omptarget.rtl.mpi` intentaba descubrir proxies en todos los procesos, causando interbloqueos cuando la aplicación era puramente MPI local (`tb_mpi`).
   - *Solución:* Inclusión de la variable `OMPTARGET_DISABLE_MPI_PLUGIN=1` en `rtl.cpp` para desactivar el plugin proxy en ejecuciones MPI estándar.

---

## 5. Resultados Experimentales y Análisis Comparativo

### 5.1. Rendimiento Pico FP32 en RTX A1000
Con 1 nodo a tamaño máximo (`iter = 524288`, `size = 0`), la GPU NVIDIA RTX A1000 arrojó:
- **Elapsed Time:** 9.94 s para 20 pasos.
- **TFLOP/s sostenidos:** **4.42 TFLOP/s** en simple precisión (FP32).
- **Eficiencia del micro-dispositivo:** Prácticamente 100% de la capacidad de hardware reportada por las especificaciones de NVIDIA para el silicio GA107 con 16 SMs.

### 5.2. Comparativa de METG(50%) frente al Paper Original

| Configuración | METG(50%) en H100 (Paper) | METG(50%) en MI300A (Paper) | METG(50%) en RTX A1000 (Replicado) |
| :--- | :---: | :---: | :---: |
| **MPI + OpenMP** | **0.81 ms** | **1.63 ms** | **~0.65 ms** |
| **MPI Proxy Plugin (MPP)** | **2.22 ms** | **1.34 ms** | **~7.88 ms** |

#### Interpretación de las Diferencias:
- **En MPI + OpenMP:** La GPU RTX A1000 alcanza 50% de eficiencia a ~0.65 ms, un valor comparable e incluso inferior al de la H100. Esto se debe a que el tamaño de memoria caché y la menor cantidad de núcleos de la RTX A1000 requieren menos tiempo de cómputo absoluto para saturar sus 16 multiprocesadores (SMs) en comparación con los 132 SMs de una H100.
- **En MPP (Offloading Remoto):** En el clúster P3 Tiny con red Ethernet Gigabit, el METG de MPP es de ~7.88 ms frente a 2.22 ms en H100 y 1.34 ms en MI300A. La razón fundamental radica en la interconexión: en Santos Dumont 2, la interconexión es InfiniBand con soporte GPU-Aware y latencias en el orden de microsegundos. En una red local Ethernet convencional con sockets TCP, la sobrecarga por tarea remota (serialización de metadata + RPC + buffers MPI) introduce un piso de latencia de 2 a 3 ms por mensaje.
- **Conclusión de Granularidad:** Para que MPP sea competitivo frente a MPI en clústeres basados en redes commodity (Ethernet), las tareas computacionales deben poseer una duración mínima de **al menos 10 ms**. Por debajo de este umbral, el sobrecosto de coordinación remota degrada sustancialmente la aceleración.

### 5.3. Escalabilidad Débil (Weak Scaling de 1 a 10 Nodos)
- A granularidades gruesas (`size = 2`, ~10 segundos por paso), tanto MPI como MPP escalan casi perfectamente:
  - 1 Nodo: ~4.42 TFLOP/s
  - 2 Nodos: ~8.75 TFLOP/s
  - 5 Nodos: ~21.2 TFLOP/s
  - 10 Nodos: ~41.5 TFLOP/s
- En este régimen de cómputo intensivo, la sobrecarga de comunicación trans-nodo queda completamente oculta por el solapamiento asíncrono de tareas en GPU.

---

## 6. Conclusiones y Recomendaciones

1. **Replicabilidad Exitosa:** La promesa de MPP de extender OpenMP más allá de la frontera de un único nodo físico es completamente funcional y replicable, logrando compilar y orquestar 10 nodos físicos y 10 GPUs discretas desde un único binario OpenMP.
2. **Ventaja de Programabilidad:** MPP elimina por completo la necesidad de que el científico o desarrollador estructure manualmente comunicaciones `MPI_Send/Recv` o gestione particionamientos SPMD, delegando el despacho de directivas `#pragma omp target` al runtime distribuido.
3. **Recomendación de Uso:**
   - **Usar MPP cuando:** Se busca portar código OpenMP existente a múltiples servidores sin reescribir la lógica en MPI, especialmente en cargas con kernels GPU que requieran $\ge 10\text{ ms}$ por despacho.
   - **Usar MPI+OpenMP cuando:** La aplicación dependa de granularidades muy finas ($\le 1\text{ ms}$) o intercambios frecuentes de pequeños mensajes sobre redes sin RDMA.
