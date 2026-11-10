#!/usr/bin/env bash
# ==============================================================================
# Script de Automatización: Compilación y Despliegue de UCX 1.18 + MPICH 4.3
# Clúster Lenovo ThinkStation P3 Tiny (10 Nodos)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${ROOT_DIR}/config/nodes.conf"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[-] Error: Archivo de configuración $CONFIG_FILE no encontrado."
    exit 1
fi

declare -A NODE_IPS
declare -A NODE_USERS
declare -A NODE_PWDS
HOSTNAMES=()

while read -r line || [[ -n "$line" ]]; do
    [[ "$line" =~ ^[[:space:]]*# ]] && continue
    [[ -z "${line// }" ]] && continue
    read -r h ip u pwd <<< "$line"
    HOSTNAMES+=("$h")
    NODE_IPS["$h"]="$ip"
    NODE_USERS["$h"]="$u"
    NODE_PWDS["$h"]="$pwd"
done < "$CONFIG_FILE"

MASTER_NODE="${HOSTNAMES[0]}"
MASTER_IP="${NODE_IPS[$MASTER_NODE]}"
MASTER_USER="${NODE_USERS[$MASTER_NODE]}"
MASTER_PWD="${NODE_PWDS[$MASTER_NODE]}"

echo "======================================================================"
echo " Configurando UCX 1.18 + MPICH 4.3 (CH4:UCX) en el Maestro ($MASTER_NODE)"
echo "======================================================================"

sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" 'bash -s' << 'REMOTE_SCRIPT'
set -euo pipefail

BUILD_DIR="$HOME/src_build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

CUDA_DIR="/usr/local/cuda"
if [ ! -d "$CUDA_DIR" ]; then
    CUDA_DIR=$(dirname $(dirname $(which nvcc 2>/dev/null || echo "/usr/local/cuda/bin/nvcc")))
fi
echo "[+] CUDA Toolkit detectado en: $CUDA_DIR"

# ------------------------------------------------------------------------------
# 1. Compilar UCX 1.18.0
# ------------------------------------------------------------------------------
if [ ! -f /opt/ucx/lib/libucp.so ]; then
    echo "[*] Descargando y compilando UCX 1.18.0..."
    if [ ! -f ucx-1.18.0.tar.gz ]; then
        wget -q --show-progress https://github.com/openucx/ucx/releases/download/v1.18.0/ucx-1.18.0.tar.gz
    fi
    tar -xzf ucx-1.18.0.tar.gz
    cd ucx-1.18.0
    ./configure --prefix=/opt/ucx \
                --with-cuda="$CUDA_DIR" \
                --enable-optimizations \
                --disable-logging \
                --disable-debug \
                --disable-assertions
    make -j24
    echo "123456" | sudo -S make install
    cd ..
    echo "[+] UCX 1.18.0 instalado exitosamente en /opt/ucx."
else
    echo "[+] UCX 1.18.0 ya se encuentra instalado en /opt/ucx."
fi

# ------------------------------------------------------------------------------
# 2. Compilar MPICH 4.3.0 con CH4:UCX (Sin el bug de Ubuntu)
# ------------------------------------------------------------------------------
if [ ! -f /opt/mpich/bin/mpiexec ]; then
    echo "[*] Descargando y compilando MPICH 4.3.0..."
    if [ ! -f mpich-4.3.0.tar.gz ]; then
        wget -q --show-progress https://www.mpich.org/static/downloads/4.3.0/mpich-4.3.0.tar.gz
    fi
    tar -xzf mpich-4.3.0.tar.gz
    cd mpich-4.3.0
    ./configure --prefix=/opt/mpich \
                --with-device=ch4:ucx \
                --with-ucx=/opt/ucx \
                --with-cuda="$CUDA_DIR" \
                --enable-threads=multiple \
                --disable-fortran
    make -j24
    echo "123456" | sudo -S make install
    cd ..
    echo "[+] MPICH 4.3.0 instalado exitosamente en /opt/mpich."
else
    echo "[+] MPICH 4.3.0 ya se encuentra instalado en /opt/mpich."
fi

# ------------------------------------------------------------------------------
# 3. Recompilar LLVM MPI Proxy Plugin y Task Bench contra /opt/mpich
# ------------------------------------------------------------------------------
echo "[*] Recompilando librerías de LLVM MPP con /opt/mpich..."
LLVM_BUILD_DIR="$HOME/src/llvm-project/build"
if [ -d "$LLVM_BUILD_DIR" ]; then
    cd "$LLVM_BUILD_DIR"
    cmake . -DMPI_HOME=/opt/mpich -DMPI_C_COMPILER=/opt/mpich/bin/mpicc -DMPI_CXX_COMPILER=/opt/mpich/bin/mpicxx
    ninja omptarget.rtl.mpi llvm-offload-mpi-proxy-device
    echo "123456" | sudo -S cp bin/llvm-offload-mpi-proxy-device /opt/llvm-mpp/bin/
    echo "123456" | sudo -S cp lib/omptarget.rtl.mpi.so /opt/llvm-mpp/lib/
    echo "[+] Plugins de LLVM MPP actualizados."
fi

TB_DIR="$HOME/src/task-bench"
if [ -d "$TB_DIR" ]; then
    echo "[*] Recompilando Task Bench con /opt/mpich..."
    cd "$TB_DIR"
    export PATH=/opt/mpich/bin:$PATH
    export LD_LIBRARY_PATH=/opt/mpich/lib:/opt/ucx/lib:${LD_LIBRARY_PATH:-}
    make clean || true
    make -j24
    echo "123456" | sudo -S cp mpi_openmp/task_bench /opt/llvm-mpp/bin/tb_mpi
    echo "123456" | sudo -S cp openmp/task_bench /opt/llvm-mpp/bin/tb_mpp || true
    echo "[+] Binarios de Task Bench actualizados en /opt/llvm-mpp/bin/."
fi

REMOTE_SCRIPT

echo "======================================================================"
echo "[+] Compilación completada en el maestro. Sincronizando hacia los nodos..."
echo "======================================================================"

# Sincronizar /opt/ucx y /opt/mpich junto con /opt/llvm-mpp a todos los nodos
SLAVE_NODES=("${HOSTNAMES[@]:1}")

for slave in "${SLAVE_NODES[@]}"; do
    slave_ip="${NODE_IPS[$slave]}"
    slave_pwd="${NODE_PWDS[$slave]}"
    slave_user="${NODE_USERS[$slave]}"
    echo "[*] Sincronizando /opt/{ucx,mpich,llvm-mpp} hacia $slave ($slave_ip)..."

    sshpass -p "$slave_pwd" ssh -o StrictHostKeyChecking=no "${slave_user}@${slave_ip}" "
        echo '$slave_pwd' | sudo -S mkdir -p /opt/ucx /opt/mpich /opt/llvm-mpp
        echo '$slave_pwd' | sudo -S chown -R ${slave_user}:${slave_user} /opt/ucx /opt/mpich /opt/llvm-mpp
    "

    sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "
        rsync -avz --delete /opt/ucx/ ${slave}:/opt/ucx/
        rsync -avz --delete /opt/mpich/ ${slave}:/opt/mpich/
        rsync -avz --delete /opt/llvm-mpp/ ${slave}:/opt/llvm-mpp/
    "
    echo "[+] $slave sincronizado correctamente."
done

echo "======================================================================"
echo " [OK] Clúster completamente desplegado con MPICH 4.3 + UCX 1.18."
echo "======================================================================"
