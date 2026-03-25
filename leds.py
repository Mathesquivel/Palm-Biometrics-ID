"""
LED client (PC) — Palm Biometrics ID
======================================
Controla os LEDs IR do Raspberry Pi Zero via HTTP.
Mantém a mesma API de leds.py original para compatibilidade total.
"""

import requests
from config import PI_ZERO_URL, TIMEOUT_CONNECT


def setup():
    """Verifica conexão com o Pi Zero ao iniciar."""
    try:
        resp = requests.get(f"{PI_ZERO_URL}/status", timeout=TIMEOUT_CONNECT)
        if resp.status_code == 200:
            data = resp.json()
            print(f"[leds] Conectado ao Pi Zero — {data.get('device', 'OK')}")
        else:
            print(f"[leds] Pi Zero respondeu com status inesperado: {resp.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"[leds] AVISO: Pi Zero não encontrado em {PI_ZERO_URL}")
        print("       Verifique se o servidor está rodando e o IP em config.py")
    except Exception as e:
        print(f"[leds] AVISO: {e}")


def liga_leds(intensity: int = 98):
    """Liga os LEDs IR no Pi Zero com a intensidade especificada (0-100)."""
    try:
        requests.post(
            f"{PI_ZERO_URL}/leds/on",
            json={"intensity": intensity},
            timeout=TIMEOUT_CONNECT,
        )
    except Exception as e:
        print(f"[leds] Erro ao ligar LEDs: {e}")


def desliga_leds():
    """Desliga os LEDs IR no Pi Zero."""
    try:
        requests.post(f"{PI_ZERO_URL}/leds/off", timeout=TIMEOUT_CONNECT)
    except Exception as e:
        print(f"[leds] Erro ao desligar LEDs: {e}")


def cleanup():
    desliga_leds()
