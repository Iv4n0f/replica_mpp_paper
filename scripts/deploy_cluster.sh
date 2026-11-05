#!/usr/bin/env bash
# ==============================================================================
# Sincronización y Despliegue de Binarios /opt/llvm-mpp en el Clúster
# Replicación SSCAD 2025 - Lenovo ThinkStation P3 Tiny (10 Nodos)
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
echo " Desplegando /opt/llvm-mpp desde $MASTER_NODE hacia los nodos esclavos"
echo "======================================================================"

# Verificar que el maestro tenga los binarios
sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "
    if [ ! -d /opt/llvm-mpp/bin ] || [ ! -f /opt/llvm-mpp/bin/tb_mpi ] || [ ! -f /opt/llvm-mpp/bin/tb_mpp ]; then
        echo '[-] Error: Faltan binarios requeridos en /opt/llvm-mpp/bin en $MASTER_NODE.'
        exit 1
    fi
    echo '[+] Binarios base verificados en $MASTER_NODE:'
    ls -l /opt/llvm-mpp/bin/{tb_mpi,tb_mpp,llvm-offload-mpi-proxy-device}
"

# Sincronizar hacia los nodos 2 en adelante
SLAVE_NODES=("${HOSTNAMES[@]:1}")

for slave in "${SLAVE_NODES[@]}"; do
    slave_ip="${NODE_IPS[$slave]}"
    slave_pwd="${NODE_PWDS[$slave]}"
    slave_user="${NODE_USERS[$slave]}"
    echo "[*] Sincronizando hacia $slave ($slave_ip)..."

    # Asegurar que el directorio destino existe y tiene permisos
    sshpass -p "$slave_pwd" ssh -o StrictHostKeyChecking=no "${slave_user}@${slave_ip}" "
        echo '$slave_pwd' | sudo -S mkdir -p /opt/llvm-mpp
        echo '$slave_pwd' | sudo -S chown -R ${slave_user}:${slave_user} /opt/llvm-mpp
    "

    # Ejecutar rsync desde el maestro hacia el esclavo
    sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "
        rsync -avz --delete /opt/llvm-mpp/ ${slave}:/opt/llvm-mpp/
    "
    
    echo "[+] $slave sincronizado correctamente."
done

echo "======================================================================"
echo " [OK] Despliegue de /opt/llvm-mpp completado en todos los nodos."
echo "======================================================================"
