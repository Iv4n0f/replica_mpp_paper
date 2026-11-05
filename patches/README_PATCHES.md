# Parches Críticos del Entorno de Replicación SSCAD 2025

Este directorio documenta las modificaciones aplicadas a LLVM/Clang y a Task Bench para hacer viable la replicación en el clúster con OpenMPI y GPUs NVIDIA.

---

## 1. Parches en LLVM Project (`llvm_mpp_fixes.patch`)

Ramas afectadas: `offload/include/PluginManager.h`, `offload/plugins-nextgen/mpi/src/rtl.cpp`, `offload/plugins-nextgen/mpi/event_system/EventSystem.cpp`.

1. **Corrección de Referencia en `registerDelayedLibraries` (`PluginManager.h`):**
   - **Causa:** LLVM llamaba a `__tgt_register_lib(Desc)` para imágenes demoradas, lo cual re-registraba la librería aumentando `RefCount` a 2. Al salir, `deinitRuntime` exigía `RefCount == 1` para llamar a `PM->deinit()`. Al no cumplirse, los proxies `llvm-offload-mpi-proxy-device` nunca recibían la señal de shutdown y se quedaban colgados indefinidamente.
   - **Solución:** Se reemplazó la llamada por `registerLib(Desc)`. Con esto, `RefCount` decrementa limpiamente a 1 al salir y todos los proxies se cierran con código de salida 0.

2. **Aislamiento de Plugin mediante Variable de Entorno (`rtl.cpp`):**
   - **Causa:** En modo MPI puro (`tb_mpi`), el plugin de offload MPI de LLVM intentaba sondear proxies en todos los rangos al inicio de la aplicación. Al no haber proxies esperando, el proceso caía en deadlock en la inicialización estática de C++.
   - **Solución:** Se añadió la comprobación de `OMPTARGET_DISABLE_MPI_PLUGIN=1` en `initImpl()` e `is_plugin_compatible()`. Si está activa, el plugin MPI se auto-inhabilita inmediatamente, permitiendo que Task Bench utilice CUDA local normalmente.

3. **Protección en Destructor de EventSystem (`EventSystem.cpp`):**
   - **Causa:** `~EventSystemTy()` arrojaba segfault si el sistema de eventos ya había sido finalizado o aún estaba en estado `CREATED`.
   - **Solución:** Se añadió guarda condicional `if (EventSystemState == FINALIZED || EventSystemState == CREATED) return;`.

---

## 2. Parches en Task Bench (`task_bench_fixes.patch`)

1. **Nuevo Kernel GPU `compute_bound_gpu`:**
   - Implementa un bucle `target teams distribute parallel for` con `num_teams(nblocks)` y `thread_limit(tpblock)`.
   - Realiza 16 operaciones unrolled por iteración interna (FMA FP32).

2. **Argumentos de Línea de Comandos:**
   - Soporte para `-nblocks` y `-tpblock`.

3. **Cálculo de TFLOPs:**
   - Fórmula: `nblocks * tpblock * iterations * 32`.

4. **Guarda de Inicialización MPI (`mpi_openmp/forall.cc`):**
   - Uso de `MPI_Initialized(&initialized)` antes de `MPI_Init` para evitar dobles inicializaciones accidentales.
