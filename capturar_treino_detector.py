"""
Captura de imagens de treino para o detector mão / não-mão.

Uso:
    python capturar_treino_detector.py

Salva em:
    detector_dataset/mao/       ← palma corretamente posicionada
    detector_dataset/nao_mao/   ← qualquer outra coisa

Teclas:
    ESPAÇO  — captura e salva o frame atual na classe selecionada
    m       — seleciona classe "mao"
    n       — seleciona classe "nao_mao"
    q/ESC   — sai

Meta recomendada:
    80 imagens de mão  +  80 imagens de não-mão  =  160 total
    Mínimo funcional:  30 + 30 = 60
"""

import os
import cv2
import numpy as np
from picamera2 import Picamera2
from leds import setup, liga_leds, desliga_leds, cleanup

DATASET_DIR = "detector_dataset"
CLASSES = {
    "mao":     os.path.join(DATASET_DIR, "mao"),
    "nao_mao": os.path.join(DATASET_DIR, "nao_mao"),
}

# Meta por classe para feedback visual
META = 80


def _contar(classe):
    d = CLASSES[classe]
    if not os.path.isdir(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith(".png")])


def _barra(atual, meta, largura=120):
    fill = int(largura * min(atual, meta) / meta)
    return "[" + "=" * fill + " " * (largura - fill) + f"] {atual}/{meta}"


def main():
    for d in CLASSES.values():
        os.makedirs(d, exist_ok=True)

    setup()
    liga_leds()

    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    config["controls"]["ExposureTime"] = 20000
    config["controls"]["AnalogueGain"] = 2.0
    picam2.configure(config)
    picam2.start()

    classe_atual = "mao"
    print("\n=== Captura de treino do detector ===")
    print("ESPAÇO = capturar | m = classe 'mao' | n = classe 'nao_mao' | q = sair\n")
    print("Dicas para imagens de MAO:")
    print("  - Palma aberta, centrada, a diferentes distâncias e leve rotação")
    print("  - Várias pessoas se possível")
    print("  - Inclua casos com leve blur (mão em movimento lento)\n")
    print("Dicas para NAO_MAO:")
    print("  - Câmera vazia (sem nada)")
    print("  - Objetos: caixa, papel, caneta, tecido")
    print("  - Pulso/braço sem mostrar a palma")
    print("  - Mão de lado (perfil)\n")

    while True:
        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if len(frame.shape) == 3 else frame.copy()
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        n_mao     = _contar("mao")
        n_nao_mao = _contar("nao_mao")
        total     = n_mao + n_nao_mao

        # ── Painel de info ──────────────────────────────────────────────────
        cor_classe = (0, 220, 0) if classe_atual == "mao" else (30, 100, 255)
        label_classe = "MAO" if classe_atual == "mao" else "NAO_MAO"
        cv2.rectangle(display, (0, 0), (640, 110), (20, 20, 20), -1)

        cv2.putText(display, f"Classe: {label_classe}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_classe, 2)
        cv2.putText(display, f"mao: {n_mao}/{META}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0) if n_mao >= META else (180, 180, 180), 1)
        cv2.putText(display, f"nao_mao: {n_nao_mao}/{META}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 0) if n_nao_mao >= META else (180, 180, 180), 1)
        cv2.putText(display, f"Total: {total}  |  [ESPACO] capturar  [m/n] classe  [q] sair",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # Borda colorida indica classe selecionada
        cv2.rectangle(display, (2, 2), (637, 477), cor_classe, 3)

        cv2.imshow("Captura de Treino — Detector", display)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('m'):
            classe_atual = "mao"
            print(f"Classe selecionada: mao ({n_mao} capturadas)")
        elif key == ord('n'):
            classe_atual = "nao_mao"
            print(f"Classe selecionada: nao_mao ({n_nao_mao} capturadas)")
        elif key == ord(' '):
            count = _contar(classe_atual)
            fname = os.path.join(CLASSES[classe_atual], f"{classe_atual}_{count + 1:04d}.png")
            cv2.imwrite(fname, gray)
            count += 1
            print(f"  [{classe_atual}] {count:04d} capturada → {fname}")

            if count == META:
                print(f"\n  Meta de {META} imagens atingida para '{classe_atual}'!")
                if _contar("mao") >= META and _contar("nao_mao") >= META:
                    print("  Ambas as classes completas. Pode treinar: python treinar_detector.py\n")

    cv2.destroyAllWindows()
    picam2.stop()
    picam2.close()
    desliga_leds()
    cleanup()

    print(f"\nResumo final:")
    print(f"  mao:     {_contar('mao')} imagens")
    print(f"  nao_mao: {_contar('nao_mao')} imagens")
    print(f"\nPara treinar: python treinar_detector.py")


if __name__ == "__main__":
    main()
