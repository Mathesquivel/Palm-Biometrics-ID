"""
Template Biométrico baseado em LBP (Local Binary Patterns).

Características:
- Extrai descritores LBP em grade espacial 8×8 sobre a imagem de veias
- Produz vetor de 640 floats — NÃO é possível reconstruir a imagem original a partir dele
- Matching por similaridade de cosseno com threshold ajustável
- Armazenamento em JSON (sem imagens)
"""

import cv2
import json
import numpy as np

# Tamanho interno de análise (independente da resolução capturada)
_LBP_SIZE = 256

# Grade espacial para extração de histogramas locais
_GRID_X = 8
_GRID_Y = 8
_BINS = 10  # bins por histograma de célula → vetor total = 8×8×10 = 640

# Threshold de similaridade de cosseno para aceitar match (0–1)
# Aumentar → mais restrito; diminuir → mais permissivo
MATCH_THRESHOLD = 0.82


def _lbp(gray: np.ndarray) -> np.ndarray:
    """
    LBP vetorizado com vizinhança quadrada de 8 pixels.
    Não depende de scikit-image. Rápido via operações numpy.
    """
    center = gray[1:-1, 1:-1].astype(np.int16)
    neighbors = [
        gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
        gray[1:-1, 2:],
        gray[2:,   2:],   gray[2:,   1:-1], gray[2:,   0:-2],
        gray[1:-1, 0:-2],
    ]
    result = np.zeros_like(center, dtype=np.uint8)
    for bit, nb in enumerate(neighbors):
        result += (nb.astype(np.int16) >= center).astype(np.uint8) * (1 << bit)
    return result


def extract_template(img: np.ndarray) -> np.ndarray:
    """
    Extrai template biométrico irreconstruível de uma imagem de veias pré-processada.

    Args:
        img: imagem em escala de cinza (resultado de preprocess_veins)

    Returns:
        np.ndarray float32 de 640 dimensões — o template biométrico.
    """
    img = cv2.resize(img, (_LBP_SIZE, _LBP_SIZE))
    lbp_img = _lbp(img)

    h, w = lbp_img.shape
    cell_h = h // _GRID_Y
    cell_w = w // _GRID_X

    features: list[float] = []
    for gy in range(_GRID_Y):
        for gx in range(_GRID_X):
            cell = lbp_img[
                gy * cell_h:(gy + 1) * cell_h,
                gx * cell_w:(gx + 1) * cell_w
            ]
            hist, _ = np.histogram(cell.ravel(), bins=_BINS, range=(0, 256))
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            features.extend(hist.tolist())

    return np.array(features, dtype=np.float32)


def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def match_template(
    probe: np.ndarray,
    stored_templates: list[np.ndarray],
    threshold: float = MATCH_THRESHOLD,
) -> tuple[bool, float]:
    """
    Compara um probe contra todos os templates armazenados de um usuário.

    Args:
        probe: template extraído da captura atual
        stored_templates: lista de templates do usuário cadastrado
        threshold: similaridade mínima para aceitar match

    Returns:
        (matched: bool, best_score: float)
    """
    best = 0.0
    for t in stored_templates:
        score = _cosine_similarity(probe, np.array(t, dtype=np.float32))
        if score > best:
            best = score
    return best >= threshold, best


def save_templates(templates: list[np.ndarray], path: str) -> None:
    """Salva lista de templates como JSON. Nenhuma imagem é salva."""
    data = [t.tolist() for t in templates]
    with open(path, "w") as f:
        json.dump(data, f)


def load_templates(path: str) -> list[np.ndarray]:
    """Carrega templates de um arquivo JSON."""
    with open(path) as f:
        data = json.load(f)
    return [np.array(t, dtype=np.float32) for t in data]
