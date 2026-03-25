"""
Camera client (PC) — Palm Biometrics ID
========================================
Conecta-se ao servidor Flask no Raspberry Pi Zero via HTTP.
Mantém a mesma API da versão original (Picamera2) para compatibilidade total
com app.py, main.py e interface.py — nenhum desses arquivos precisa mudar.

Fluxo:
  1. start() → inicia thread de fundo que busca frames continuamente
  2. capture_array() → retorna o frame mais recente (numpy BGR)
  3. stop() / close() → encerra a thread
"""

import cv2
import time
import threading
import numpy as np
import requests

from config import PI_ZERO_URL, TIMEOUT_CONNECT, TIMEOUT_READ, JPEG_QUALITY


class PiZeroCamera:
    """
    Wrapper que imita a API da Picamera2, mas busca frames via HTTP
    do servidor Flask rodando no Raspberry Pi Zero.
    """

    def __init__(self):
        self._latest_frame: np.ndarray = np.zeros((960, 1280, 3), dtype=np.uint8)
        self._lock    = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._session = requests.Session()

    # ── API pública ────────────────────────────────────────────────────────────

    def start(self):
        """Inicia thread de captura contínua em background."""
        self._running = True
        self._thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._thread.start()

    def capture_array(self) -> np.ndarray:
        """
        Retorna o frame BGR mais recente como numpy array.
        Idêntico ao comportamento de Picamera2.capture_array().
        """
        with self._lock:
            return self._latest_frame.copy()

    def stop(self):
        """Para a thread de captura."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)

    def close(self):
        self.stop()
        self._session.close()

    # ── Thread interna ─────────────────────────────────────────────────────────

    def _fetch_loop(self):
        url = f"{PI_ZERO_URL}/frame?quality={JPEG_QUALITY}"
        while self._running:
            try:
                resp = self._session.get(
                    url, timeout=(TIMEOUT_CONNECT, TIMEOUT_READ)
                )
                if resp.status_code == 200:
                    arr   = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with self._lock:
                            self._latest_frame = frame
            except requests.exceptions.ConnectionError:
                print("[camera] Pi Zero não encontrado — tentando novamente...")
                time.sleep(1)
            except Exception as e:
                print(f"[camera] Erro: {e}")
                time.sleep(0.2)


# ── Funções de compatibilidade (mesma assinatura da versão Pi) ─────────────────

_cam: PiZeroCamera | None = None


def setup_camera() -> PiZeroCamera:
    global _cam
    _cam = PiZeroCamera()
    return _cam


def start_preview(picam2: PiZeroCamera):
    """Inicia thread de captura. O preview visual é gerido pela interface."""
    picam2.start()


def stop_preview(picam2: PiZeroCamera):
    picam2.stop()
    picam2.close()
