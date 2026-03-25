"""
Servidor Flask — Raspberry Pi Zero
====================================
Responsável por:
  - Controle dos LEDs IR via GPIO (pino 17)
  - Captura de imagens com Picamera2
  - Envio de frames ao PC principal via HTTP

Rotas disponíveis:
  GET  /status      → health check
  GET  /frame       → retorna um frame JPEG
  POST /leds/on     → liga LEDs (body JSON: {"intensity": 98})
  POST /leds/off    → desliga LEDs

Uso:
  python3 server.py
  # ou para iniciar automaticamente ao boot: ver install.sh
"""

import time
import cv2
import numpy as np
from flask import Flask, request, jsonify, Response
from picamera2 import Picamera2
import RPi.GPIO as GPIO

# ── Configuração de hardware ───────────────────────────────────────────────────
LED_PIN  = 17
PWM_FREQ = 90   # Hz

# ── Configuração da câmera ─────────────────────────────────────────────────────
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 960
EXPOSURE     = 20000  # microssegundos
GAIN         = 1.0

# ── Inicialização GPIO ─────────────────────────────────────────────────────────
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
pwm = GPIO.PWM(LED_PIN, PWM_FREQ)
pwm.start(0)

# ── Inicialização da câmera ────────────────────────────────────────────────────
picam2 = Picamera2()
cfg = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT)}
)
cfg["controls"]["ExposureTime"] = EXPOSURE
cfg["controls"]["AnalogueGain"] = GAIN
picam2.configure(cfg)
picam2.start()
time.sleep(1.5)   # aguarda câmera estabilizar

print(f"[Pi Zero] Servidor pronto — câmera {FRAME_WIDTH}x{FRAME_HEIGHT}")

# ── Flask ──────────────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/status")
def status():
    return jsonify({"status": "ok", "device": "Pi Zero"})


@app.route("/frame")
def frame():
    """Captura um frame e retorna como JPEG."""
    quality = int(request.args.get("quality", 85))
    quality = max(10, min(100, quality))

    img = picam2.capture_array()

    # Picamera2 pode retornar XRGB (4 canais) — converte para BGR
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return Response(jpeg.tobytes(), mimetype="image/jpeg")


@app.route("/leds/on", methods=["POST"])
def leds_on():
    data      = request.get_json(silent=True) or {}
    intensity = int(data.get("intensity", 98))
    intensity = max(0, min(100, intensity))
    pwm.ChangeDutyCycle(intensity)
    return jsonify({"status": "on", "intensity": intensity})


@app.route("/leds/off", methods=["POST"])
def leds_off():
    pwm.ChangeDutyCycle(0)
    return jsonify({"status": "off"})


# ── Ponto de entrada ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        print("[Pi Zero] Aguardando conexões na porta 5000...")
        app.run(host="0.0.0.0", port=5000, threaded=True)
    finally:
        print("[Pi Zero] Encerrando...")
        pwm.stop()
        GPIO.cleanup()
        picam2.stop()
        picam2.close()
