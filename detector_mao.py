"""
Detector binário: mão / não-mão usando HOG + SVM treinado.

Se o modelo não existir, o sistema funciona sem detector
(fallback para detecção por contorno apenas).

API pública:
    is_hand(gray_img) -> (bool, float)
      bool:  True = é uma mão
      float: score de confiança 0.0–1.0
"""

import cv2
import numpy as np
import os

MODEL_PATH = "detector_mao.xml"
WIN_SIZE   = (64, 64)

_svm: cv2.ml.SVM | None = None
_hog: cv2.HOGDescriptor | None = None


def _get_hog() -> cv2.HOGDescriptor:
    global _hog
    if _hog is None:
        _hog = cv2.HOGDescriptor(
            _winSize    = WIN_SIZE,
            _blockSize  = (16, 16),
            _blockStride= (8, 8),
            _cellSize   = (8, 8),
            _nbins      = 9,
        )
    return _hog


def load_detector() -> bool:
    """
    Carrega o modelo SVM do disco.
    Retorna True se carregado com sucesso, False se arquivo não existe.
    """
    global _svm
    if not os.path.exists(MODEL_PATH):
        return False
    try:
        _svm = cv2.ml.SVM_load(MODEL_PATH)
        return True
    except Exception as e:
        print(f"[detector_mao] Erro ao carregar modelo: {e}")
        _svm = None
        return False


def _extract_hog(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, WIN_SIZE)
    desc = _get_hog().compute(resized)
    return desc.flatten().astype(np.float32)


def is_hand(roi: np.ndarray) -> tuple[bool, float]:
    """
    Verifica se a ROI recortada contém uma palma da mão.

    Args:
        roi: imagem em escala de cinza da região de interesse (resultado do
             bounding box de _detect_hand)

    Returns:
        (é_mão: bool, confiança: float 0–1)
        Se modelo não treinado ainda: retorna (True, 1.0) como fallback.
    """
    global _svm

    # Modelo não treinado → fallback permissivo
    if _svm is None:
        if os.path.exists(MODEL_PATH):
            load_detector()
        if _svm is None:
            return True, 1.0

    features = _extract_hog(roi).reshape(1, -1)

    # Predição + distância à margem de decisão (proxy de confiança)
    _, result   = _svm.predict(features)
    _, distances = _svm.predict(features, flags=cv2.ml.StatModel_RAW_OUTPUT)

    label = int(result[0][0])          # 1 = mão, -1 = não-mão
    dist  = float(distances[0][0])     # distância ao hiperplano (signed)

    # Transforma distância em score 0–1 via sigmoid
    # dist > 0  → SVM favorece classe 1 (mão)
    # dist < 0  → SVM favorece classe -1 (não-mão)
    score = 1.0 / (1.0 + np.exp(-dist))   # sigmoid: score ∈ (0, 1)

    return label == 1, score


def detector_disponivel() -> bool:
    """Retorna True se o modelo foi treinado e carregado."""
    return _svm is not None or os.path.exists(MODEL_PATH)


# Tenta carregar ao importar o módulo
load_detector()
