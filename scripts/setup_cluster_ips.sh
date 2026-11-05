#!/usr/bin/env bash
# ==============================================================================
# Setup y Actualización de IPs Dinámicas del Clúster
# Replicación SSCAD 2025 - Lenovo ThinkStation P3 Tiny (10 Nodos)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="${ROOT_DIR}/config/nodes.conf"
MCA_FILE="${ROOT_DIR}/config/openmpi-mca-params.conf"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "[-] Error: Archivo de configuración $CONFIG_FILE no encontrado."
    exit 1
fi

echo "======================================================================"
echo " Actualizando configuración de red del Clúster desde $CONFIG_FILE"
echo "======================================================================"

declare -A NODE_IPS
declare -A NODE_USERS
declare -A NODE_PWDS
HOSTNAMES=()

# 1. Parsear archivo de configuración
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

echo "[+] Nodo maestro identificado: $MASTER_NODE ($MASTER_IP) como usuario $MASTER_USER"
echo "[+] Total nodos cargados: ${#HOSTNAMES[@]}"

# 2. Generar bloque temporal de /etc/hosts
HOSTS_BLOCK_FILE=$(mktemp)
echo "# --- CLUSTER NODES MANAGED BY REPLICATION KIT ---" > "$HOSTS_BLOCK_FILE"
for h in "${HOSTNAMES[@]}"; do
    echo "${NODE_IPS[$h]} $h" >> "$HOSTS_BLOCK_FILE"
done
echo "# --- END CLUSTER NODES ---" >> "$HOSTS_BLOCK_FILE"

echo "[+] Contenido de mapeo de hosts generado:"
cat "$HOSTS_BLOCK_FILE"
echo ""

# Helper para ejecutar comandos remotos con sshpass si es necesario
run_remote_sudo() {
    local ip="$1"
    local user="$2"
    local pwd="$3"
    local cmd="$4"
    sshpass -p "$pwd" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${user}@${ip}" "echo '$pwd' | sudo -S bash -c '$cmd'"
}

# 3. Aplicar /etc/hosts en todos los nodos
echo "[*] Aplicando nuevo /etc/hosts en todos los nodos..."
for h in "${HOSTNAMES[@]}"; do
    ip="${NODE_IPS[$h]}"
    u="${NODE_USERS[$h]}"
    p="${NODE_PWDS[$h]}"
    echo "  -> Configurando $h ($ip)..."
    
    # Remover bloque previo y añadir nuevo
    sshpass -p "$p" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${u}@${ip}" "
        echo '$p' | sudo -S sed -i '/# --- CLUSTER NODES MANAGED BY REPLICATION KIT ---/,/# --- END CLUSTER NODES ---/d' /etc/hosts
    " || { echo "[-] Fallo al limpiar /etc/hosts en $h"; continue; }

    sshpass -p "$p" scp -o StrictHostKeyChecking=no "$HOSTS_BLOCK_FILE" "${u}@${ip}:/tmp/cluster_hosts"
    sshpass -p "$p" ssh -o StrictHostKeyChecking=no "${u}@${ip}" "
        echo '$p' | sudo -S bash -c 'cat /tmp/cluster_hosts >> /etc/hosts && rm -f /tmp/cluster_hosts'
    "
done
rm -f "$HOSTS_BLOCK_FILE"

# 4. Asegurar parámetros MCA de OpenMPI en todos los nodos
if [[ -f "$MCA_FILE" ]]; then
    echo "[*] Sincronizando parámetros MCA de OpenMPI..."
    for h in "${HOSTNAMES[@]}"; do
        ip="${NODE_IPS[$h]}"
        u="${NODE_USERS[$h]}"
        p="${NODE_PWDS[$h]}"
        sshpass -p "$p" scp -o StrictHostKeyChecking=no "$MCA_FILE" "${u}@${ip}:/tmp/openmpi-mca-params.conf"
        sshpass -p "$p" ssh -o StrictHostKeyChecking=no "${u}@${ip}" "
            echo '$p' | sudo -S mv /tmp/openmpi-mca-params.conf /etc/openmpi/openmpi-mca-params.conf
        "
    done
fi

# 5. Asegurar llaves SSH de master hacia todos los nodos
echo "[*] Configurando autenticación SSH sin contraseña desde $MASTER_NODE..."
sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "
    if [ ! -f ~/.ssh/id_rsa.pub ]; then
        ssh-keygen -t rsa -N '' -f ~/.ssh/id_rsa
    fi
"
PUBKEY=$(sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "cat ~/.ssh/id_rsa.pub")

for h in "${HOSTNAMES[@]}"; do
    ip="${NODE_IPS[$h]}"
    u="${NODE_USERS[$h]}"
    p="${NODE_PWDS[$h]}"
    sshpass -p "$p" ssh -o StrictHostKeyChecking=no "${u}@${ip}" "
        mkdir -p ~/.ssh && chmod 700 ~/.ssh
        touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
        grep -qF '$PUBKEY' ~/.ssh/authorized_keys || echo '$PUBKEY' >> ~/.ssh/authorized_keys
    "
done

# 6. Probar conectividad SSH y OpenMPI entre todos los nodos
echo "[*] Verificando comunicación OpenMPI en el clúster..."
HOST_LIST=$(IFS=, ; echo "${HOSTNAMES[*]}")

sshpass -p "$MASTER_PWD" ssh -o StrictHostKeyChecking=no "${MASTER_USER}@${MASTER_IP}" "
    echo '  [Master Check] Probando conectividad hacia todos los nodos:'
    for node in ${HOSTNAMES[*]}; do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 \$node 'echo -n \"    Conectado a \$HOSTNAME: \" && nvidia-smi --query-gpu=name --format=csv,noheader' || echo \"    ERROR conectando a \$node\"
    done

    echo '  [MPI Check] Ejecutando mpirun en todos los ${#HOSTNAMES[@]} nodos...'
    mpirun --host $HOST_LIST -N 1 -np ${#HOSTNAMES[@]} hostname
"

echo "======================================================================"
echo " [OK] Clúster configurado y validado exitosamente con las nuevas IPs."
echo "======================================================================"
