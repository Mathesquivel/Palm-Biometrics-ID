#!/usr/bin/env bash
# ============================================================
# install.sh — Configuração do Raspberry Pi Zero
# Palm Biometrics ID
# ============================================================
# Execute uma vez no Pi Zero:
#   chmod +x install.sh && sudo ./install.sh
# ============================================================

set -e

echo "======================================"
echo " Palm Biometrics ID — Setup Pi Zero"
echo "======================================"

# ── Atualiza sistema ───────────────────────────────────────
echo "[1/5] Atualizando sistema..."
apt-get update -qq && apt-get upgrade -y -qq

# ── Instala dependências de sistema ───────────────────────
echo "[2/5] Instalando dependências..."
apt-get install -y -qq \
    python3-pip \
    python3-venv \
    python3-picamera2 \
    python3-libcamera \
    python3-opencv \
    python3-rpi.gpio \
    libopencv-dev

# ── Cria ambiente virtual e instala pacotes Python ─────────
echo "[3/5] Criando ambiente virtual..."
INSTALL_DIR="$(dirname "$(readlink -f "$0")")"
cd "$INSTALL_DIR"

python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install --quiet flask

# ── Descobre IP e exibe ────────────────────────────────────
echo "[4/5] Verificando IP da rede..."
IP=$(hostname -I | awk '{print $1}')
echo ""
echo "  IP do Pi Zero: $IP"
echo "  Atualize config.py no PC com: PI_ZERO_HOST = \"$IP\""
echo ""

# ── Cria serviço systemd para iniciar automaticamente ─────
echo "[5/5] Configurando serviço systemd..."

SERVICE_FILE="/etc/systemd/system/palm-pizero.service"

cat > "$SERVICE_FILE" <<EOF
[Unit]
Description=Palm Biometrics ID — Pi Zero Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=pi
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python3 $INSTALL_DIR/server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable palm-pizero
systemctl start palm-pizero

echo ""
echo "======================================"
echo " Instalacao concluida!"
echo ""
echo " Servidor rodando em: http://$IP:5000"
echo " Status: systemctl status palm-pizero"
echo " Logs:   journalctl -u palm-pizero -f"
echo "======================================"
