"""
Script de diagnóstico — verifica visibilidade das veias e testa a interface de guia.

Modo 1 (padrão): grade 2×2 mostrando os estágios do pipeline de pré-processamento
Modo 2 (tecla G): interface de guia de posicionamento

Teclas:
  s      — salva frame atual (raw + blackhat)
  g      — alterna para modo interface de guia
  + / -  — ajusta exposição
  q/ESC  — sai
"""

import cv2
import numpy as np
from picamera2 import Picamera2

from leds import setup, liga_leds, desliga_leds, cleanup
from interface import draw_overlay, _detect_hand, _sharpness, _to_bgr


def _pipeline_grid(gray: np.ndarray, exposure: int) -> np.ndarray:
    """Monta grade 2×2 com os estágios do pipeline."""
    raw = gray.copy()
    equalized = cv2.equalizeHist(gray)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)

    smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    blackhat = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    blackhat_norm = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)

    top = np.hstack([raw, equalized])
    bot = np.hstack([clahe_img, blackhat_norm])
    grid = np.vstack([top, bot])
    h, w = grid.shape
    display = cv2.resize(grid, (w // 2, h // 2))

    dh, dw = display.shape
    labels = [
        (10, 20, "Original"),
        (dw // 2 + 10, 20, "Equalizado"),
        (10, dh // 2 + 20, "CLAHE"),
        (dw // 2 + 10, dh // 2 + 20, "Black-Hat (veias)"),
    ]
    for x, y, txt in labels:
        cv2.putText(display, txt, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, 240, 1, cv2.LINE_AA)
    cv2.putText(display, f"Exp: {exposure} us  [+/-] ajustar  [g] guia  [s] salvar  [q] sair",
                (10, dh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, 200, 1, cv2.LINE_AA)
    return display, blackhat_norm


def main():
    setup()
    liga_leds()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1280, 960)})
    config["controls"]["ExposureTime"] = 20000
    config["controls"]["AnalogueGain"] = 2.0
    picam2.configure(config)
    picam2.start()

    print("Câmera iniciada.")
    print("Teclas: [s] salvar  [+/-] exposição  [g] modo guia  [q/ESC] sair\n")

    exposure = 20000
    modo_guia = False
    last_raw = None
    last_blackhat = None

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if len(frame.shape) == 3 else frame.copy()
        last_raw = gray.copy()

        if modo_guia:
            hand_info = _detect_hand(gray)
            if hand_info is not None:
                hand_info["sharpness"] = _sharpness(gray, hand_info["bbox"])
            display, _ = draw_overlay(_to_bgr(gray), hand_info)
            cv2.putText(display, "[g] pipeline  [q] sair",
                        (10, display.shape[0] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, 200, 1, cv2.LINE_AA)
            cv2.imshow("Diagnostico PalmBiometrics", display)
        else:
            display, blackhat_norm = _pipeline_grid(gray, exposure)
            last_blackhat = blackhat_norm
            cv2.imshow("Diagnostico PalmBiometrics", display)

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('g'):
            modo_guia = not modo_guia
            print(f"Modo: {'interface de guia' if modo_guia else 'pipeline'}")
        elif key == ord('s'):
            if last_raw is not None:
                cv2.imwrite("captura_teste.png", last_raw)
                print("Salvo: captura_teste.png")
            if last_blackhat is not None:
                cv2.imwrite("blackhat_teste.png", last_blackhat)
                print("Salvo: blackhat_teste.png")
        elif key in (ord('+'), ord('=')):
            exposure = min(60000, exposure + 2000)
            picam2.set_controls({"ExposureTime": exposure})
            print(f"Exposição: {exposure} µs")
        elif key == ord('-'):
            exposure = max(3000, exposure - 2000)
            picam2.set_controls({"ExposureTime": exposure})
            print(f"Exposição: {exposure} µs")

    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()
    desliga_leds()
    cleanup()


if __name__ == "__main__":
    main()
