"""
Interface de guia para posicionamento da mão.

Exibe em tempo real:
  - Elipse alvo no centro do frame
  - Setas indicando direção de ajuste (esquerda/direita/cima/baixo)
  - Indicador de distância (aproxime / afaste)
  - Indicador de foco (nitidez via variância do Laplacian)
  - Barra de progresso: captura automática após mão estável por STABLE_TIME segundos
  - Painel de status com cada critério individualmente

Uso em outros módulos:
    from interface import wait_for_hand
    gray = wait_for_hand(picam2)   # bloqueia até mão posicionada ou timeout
"""

import cv2
import time
import numpy as np
from detector_mao import is_hand, detector_disponivel

# ── Zona alvo (elipse centralizada) ───────────────────────────────────────────
TARGET_W_RATIO = 0.45   # largura da elipse / largura do frame
TARGET_H_RATIO = 0.55   # altura da elipse / altura do frame

# ── Distância (área da mão como fração do frame) ──────────────────────────────
AREA_MIN = 0.10   # muito longe abaixo disso
AREA_MAX = 0.35   # muito perto acima disso

# ── Tolerância de centralização ───────────────────────────────────────────────
CENTER_TOL = 45   # pixels

# ── Foco (variância do Laplacian sobre a ROI da mão) ─────────────────────────
SHARPNESS_MIN = 10   # câmera foco fixo — limiar baixo intencional

# ── Tempo estável antes da captura automática ────────────────────────────────
STABLE_TIME = 1.5   # segundos


# ─────────────────────────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _detect_hand(gray: np.ndarray) -> dict | None:
    """
    Detecta região da mão em imagem IR via threshold Otsu + contornos.
    Retorna dict {centroid, bbox, area} ou None.
    """
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 2000:
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(largest)

    return {"centroid": (cx, cy), "bbox": (x, y, w, h), "area": area}


def _sharpness(gray: np.ndarray, bbox: tuple) -> float:
    """Variância do Laplacian na ROI da mão — proxy de nitidez."""
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    if roi.size == 0:
        return 0.0
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


def _arrow(img, direction, pos, color=(0, 80, 255), size=28):
    """Desenha seta sólida apontando para a direção indicada."""
    cx, cy = pos
    s = size
    pts = {
        "left":  [(cx - s, cy), (cx, cy - s // 2), (cx, cy + s // 2)],
        "right": [(cx + s, cy), (cx, cy - s // 2), (cx, cy + s // 2)],
        "up":    [(cx, cy - s), (cx - s // 2, cy), (cx + s // 2, cy)],
        "down":  [(cx, cy + s), (cx - s // 2, cy), (cx + s // 2, cy)],
    }
    if direction in pts:
        cv2.fillPoly(img, [np.array(pts[direction], np.int32)], color)


def _status_line(img, text, y, ok):
    color = (0, 210, 0) if ok else (30, 100, 255)
    icon = "v" if ok else "x"
    cv2.putText(img, f"[{icon}] {text}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)


def draw_overlay(
    frame_bgr: np.ndarray,
    hand_info: dict | None,
    stable_progress: float = 0.0,
) -> tuple[np.ndarray, bool]:
    """
    Desenha overlay de guia sobre frame BGR.

    Args:
        frame_bgr:       frame já em BGR
        hand_info:       resultado de _detect_hand + campo 'sharpness'
        stable_progress: 0.0–1.0 — progresso até captura automática

    Returns:
        (overlay_bgr, is_ready)
    """
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    tcx, tcy = w // 2, h // 2
    trx = int(w * TARGET_W_RATIO / 2)
    try_ = int(h * TARGET_H_RATIO / 2)

    # ── Sem mão ───────────────────────────────────────────────────────────────
    if hand_info is None:
        cv2.ellipse(overlay, (tcx, tcy), (trx, try_), 0, 0, 360,
                    (0, 140, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, "Posicione a palma na area indicada",
                    (w // 2 - 195, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 140, 255), 2, cv2.LINE_AA)
        return overlay, False

    cx, cy     = hand_info["centroid"]
    area_ratio = hand_info["area"] / (w * h)
    sharp      = hand_info.get("sharpness", 0.0)

    dx = cx - tcx
    dy = cy - tcy

    ok_x        = abs(dx) <= CENTER_TOL
    ok_y        = abs(dy) <= CENTER_TOL
    ok_dist     = AREA_MIN <= area_ratio <= AREA_MAX
    ok_foco     = sharp >= SHARPNESS_MIN

    # Detector treinado (se disponível)
    det_result  = hand_info.get("is_hand", None)   # None = sem modelo
    det_score   = hand_info.get("det_score", 1.0)
    ok_detector = (det_result is None) or bool(det_result)

    is_ready = ok_x and ok_y and ok_dist and ok_foco and ok_detector

    # ── Elipse alvo ───────────────────────────────────────────────────────────
    ellipse_color = (0, 230, 0) if is_ready else (0, 140, 255)
    cv2.ellipse(overlay, (tcx, tcy), (trx, try_), 0, 0, 360,
                ellipse_color, 2, cv2.LINE_AA)

    # Cruz central alvo
    cross = 12
    cv2.line(overlay, (tcx - cross, tcy), (tcx + cross, tcy), ellipse_color, 1)
    cv2.line(overlay, (tcx, tcy - cross), (tcx, tcy + cross), ellipse_color, 1)

    # Ponto centroide da mão
    cv2.circle(overlay, (cx, cy), 5, (255, 230, 0), -1, cv2.LINE_AA)

    # ── Setas de direção ──────────────────────────────────────────────────────
    margin = 50
    if not ok_x:
        # Mover para o lado onde está o alvo (oposto ao deslocamento)
        if dx > 0:   # mão está à direita → mover para esquerda
            _arrow(overlay, "left",  (margin, h // 2))
        else:        # mão está à esquerda → mover para direita
            _arrow(overlay, "right", (w - margin, h // 2))
    if not ok_y:
        if dy > 0:   # mão está abaixo → mover para cima
            _arrow(overlay, "up",   (w // 2, margin))
        else:        # mão está acima → mover para baixo
            _arrow(overlay, "down", (w // 2, h - margin))

    # ── Mensagens de distância / foco / detector ──────────────────────────────
    msgs = []
    if not ok_dist:
        msgs.append("Aproxime a mao" if area_ratio < AREA_MIN else "Afaste a mao")
    if not ok_foco:
        msgs.append("Ajuste a distancia (foco)")
    if not ok_detector:
        msgs.append("Objeto nao reconhecido como mao")

    for i, msg in enumerate(msgs):
        cv2.putText(overlay, msg, (w // 2 - 140, h - 42 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 100, 255), 2, cv2.LINE_AA)

    # ── Painel de status (canto superior esquerdo) ────────────────────────────
    panel_h = 120 if det_result is not None else 100
    roi_panel = overlay[4:4 + panel_h, 4:185]
    dark = np.zeros_like(roi_panel)
    cv2.addWeighted(dark, 0.55, roi_panel, 0.45, 0, roi_panel)
    overlay[4:4 + panel_h, 4:185] = roi_panel

    dist_txt = "OK" if ok_dist else ("Aproxime" if area_ratio < AREA_MIN else "Afaste")
    foco_txt = f"OK ({sharp:.0f})" if ok_foco else f"Ruim ({sharp:.0f})"
    xoff_txt = "OK" if ok_x else (f"<{abs(dx):.0f}px" if dx > 0 else f">{abs(dx):.0f}px")
    yoff_txt = "OK" if ok_y else (f"v{abs(dy):.0f}px" if dy > 0 else f"^{abs(dy):.0f}px")

    _status_line(overlay, f"Hor:  {xoff_txt}",   20, ok_x)
    _status_line(overlay, f"Ver:  {yoff_txt}",   38, ok_y)
    _status_line(overlay, f"Dist: {dist_txt}",   56, ok_dist)
    _status_line(overlay, f"Foco: {foco_txt}",   74, ok_foco)
    if det_result is not None:
        det_txt = f"Mao ({det_score:.2f})" if ok_detector else f"Nao e mao ({det_score:.2f})"
        _status_line(overlay, f"IA:   {det_txt}", 92, ok_detector)

    # ── Barra de progresso ────────────────────────────────────────────────────
    if is_ready:
        bar_x1, bar_x2 = 20, w - 20
        bar_y1, bar_y2 = h - 14, h - 6
        fill = int((bar_x2 - bar_x1) * stable_progress)
        cv2.rectangle(overlay, (bar_x1, bar_y1), (bar_x2, bar_y2),
                      (40, 40, 40), -1)
        cv2.rectangle(overlay, (bar_x1, bar_y1),
                      (bar_x1 + fill, bar_y2), (0, 230, 0), -1)
        cv2.putText(overlay, "Capturando...",
                    (w // 2 - 65, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 0), 1, cv2.LINE_AA)

    return overlay, is_ready


def wait_for_hand(
    picam2,
    stable_time: float = STABLE_TIME,
    timeout: float = 30.0,
    window_name: str = "PalmBiometrics — Posicione a mao",
) -> np.ndarray | None:
    """
    Exibe interface de guia e aguarda mão corretamente posicionada.

    Args:
        picam2:      instância da Picamera2 já iniciada
        stable_time: segundos que a mão deve ficar estável antes de capturar
        timeout:     tempo máximo de espera em segundos (None = infinito)
        window_name: título da janela OpenCV

    Returns:
        Imagem em escala de cinza (uint8) quando pronta, ou None se timeout/cancelado.
    """
    stable_since: float | None = None
    deadline = (time.time() + timeout) if timeout else None
    last_gray = None

    while True:
        if deadline and time.time() > deadline:
            cv2.destroyWindow(window_name)
            return None

        frame = picam2.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
               if len(frame.shape) == 3 else frame.copy()
        last_gray = gray

        hand_info = _detect_hand(gray)
        if hand_info is not None:
            hand_info["sharpness"] = _sharpness(gray, hand_info["bbox"])
            # Detector treinado: confirma se é de fato uma mão
            if detector_disponivel():
                x, y, bw, bh = hand_info["bbox"]
                roi = gray[y:y + bh, x:x + bw]
                eh_mao, score = is_hand(roi)
                hand_info["is_hand"]   = eh_mao
                hand_info["det_score"] = score

        # Avalia se está pronto (sem progresso ainda, só para checar is_ready)
        _, is_ready = draw_overlay(_to_bgr(gray), hand_info, stable_progress=0.0)

        if is_ready:
            if stable_since is None:
                stable_since = time.time()
            progress = min((time.time() - stable_since) / stable_time, 1.0)
        else:
            stable_since = None
            progress = 0.0

        # Desenha overlay com progresso real
        overlay, _ = draw_overlay(_to_bgr(gray), hand_info,
                                  stable_progress=progress)
        cv2.imshow(window_name, overlay)

        if is_ready and progress >= 1.0:
            cv2.destroyWindow(window_name)
            return last_gray

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:   # q ou ESC cancela
            cv2.destroyWindow(window_name)
            return None
