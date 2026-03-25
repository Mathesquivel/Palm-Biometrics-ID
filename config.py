"""
Configurações globais — Palm Biometrics ID
==========================================
Edite PI_ZERO_HOST com o IP do seu Raspberry Pi Zero na rede local.
Para descobrir o IP do Pi Zero, rode no Pi Zero: hostname -I
"""

# ── Raspberry Pi Zero ──────────────────────────────────────────────────────────
PI_ZERO_HOST = "192.168.1.100"   # << ALTERE para o IP do seu Pi Zero
PI_ZERO_PORT = 5000
PI_ZERO_URL  = f"http://{PI_ZERO_HOST}:{PI_ZERO_PORT}"

# ── Timeouts de rede (segundos) ────────────────────────────────────────────────
TIMEOUT_CONNECT = 3    # tempo máximo para estabelecer conexão
TIMEOUT_READ    = 8    # tempo máximo para receber resposta

# ── Câmera ─────────────────────────────────────────────────────────────────────
JPEG_QUALITY    = 85   # qualidade JPEG dos frames transmitidos (0–100)
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 960
