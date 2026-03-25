"""
Palm Biometrics ID — Interface Gráfica
Design: futurista, minimalista, tecnológico
"""

import sys
import os


import cv2
import time
import numpy as np
from datetime import datetime

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QLineEdit, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QScrollArea, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QLinearGradient

from preprocess_veins import preprocess_veins
from biometric_template import (
    extract_template, match_template,
    save_templates, load_templates, MATCH_THRESHOLD
)
from interface import _detect_hand, _sharpness, draw_overlay, _to_bgr
from detector_mao import is_hand, detector_disponivel
from database import init_db, add_user, list_users, list_users_full, get_user_info

# ─── Paleta ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#050b14",
    "card":     "#0d1b2a",
    "card2":    "#091320",
    "accent":   "#00e5ff",
    "accent2":  "#7c3aed",
    "text":     "#e2f4ff",
    "dim":      "#4a6580",
    "success":  "#00ff9d",
    "error":    "#ff1744",
    "warn":     "#ff9800",
    "border":   "#1a3050",
    "borderbr": "#00e5ff",
}

STABLE_TIME = 1.5   # segundos para captura automática
NUM_CAPTURAS = 16

# ─── Stylesheet ───────────────────────────────────────────────────────────────
QSS = f"""
QMainWindow {{ background: {C["bg"]}; }}
QWidget     {{ background: {C["bg"]}; color: {C["text"]};
               font-family: 'Courier New', Courier, monospace; font-size: 13px; }}

/* ── Botões de menu ── */
QPushButton.menu_btn {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    border-left: 3px solid {C["accent"]};
    color: {C["text"]};
    font-size: 14px; font-weight: bold; letter-spacing: 3px;
    padding: 0 24px; text-align: left;
    min-height: 60px;
}}
QPushButton.menu_btn:hover {{
    background: rgba(0,229,255,0.07);
    border: 1px solid {C["accent"]};
    border-left: 3px solid {C["accent"]};
    color: {C["accent"]};
}}
QPushButton.menu_btn:pressed {{ background: rgba(0,229,255,0.14); }}

/* ── Botões primários ── */
QPushButton.primary {{
    background: rgba(0,229,255,0.10);
    border: 1px solid {C["accent"]};
    color: {C["accent"]};
    font-size: 13px; font-weight: bold; letter-spacing: 2px;
    padding: 11px 28px; text-align: center;
}}
QPushButton.primary:hover  {{ background: rgba(0,229,255,0.18); }}
QPushButton.primary:pressed {{ background: rgba(0,229,255,0.28); }}

/* ── Botões secundários ── */
QPushButton.secondary {{
    background: transparent;
    border: 1px solid {C["border"]};
    color: {C["dim"]};
    font-size: 12px; letter-spacing: 2px;
    padding: 11px 24px; text-align: center;
}}
QPushButton.secondary:hover {{ border-color: {C["text"]}; color: {C["text"]}; }}

/* ── Botão perigo ── */
QPushButton.danger {{
    background: transparent;
    border: 1px solid rgba(255,23,68,0.4);
    color: {C["error"]};
    font-size: 12px; letter-spacing: 2px;
    padding: 9px 20px;
}}
QPushButton.danger:hover {{ background: rgba(255,23,68,0.08); border-color: {C["error"]}; }}

/* ── Inputs ── */
QLineEdit {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    color: {C["text"]};
    font-family: 'Courier New', monospace; font-size: 13px;
    padding: 9px 12px;
    selection-background-color: {C["accent"]}; selection-color: {C["bg"]};
}}
QLineEdit:focus {{ border-color: {C["accent"]}; background: #0f2136; }}

/* ── Tabela ── */
QTableWidget {{
    background: {C["card"]};
    border: 1px solid {C["border"]};
    gridline-color: {C["border"]};
    color: {C["text"]};
    font-family: 'Courier New', monospace;
    selection-background-color: rgba(0,229,255,0.12);
    selection-color: {C["accent"]};
    outline: none;
}}
QTableWidget::item {{ padding: 6px 12px; }}
QHeaderView::section {{
    background: {C["card2"]};
    color: {C["accent"]};
    border: none; border-bottom: 1px solid {C["border"]};
    padding: 8px 12px; font-size: 11px; letter-spacing: 2px;
}}
QScrollBar:vertical   {{ background: {C["card"]}; width: 6px; border: none; }}
QScrollBar::handle:vertical {{ background: {C["border"]}; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

/* ── Labels especiais ── */
QLabel#title   {{ color: {C["accent"]}; font-size: 30px; font-weight: bold;
                  letter-spacing: 6px; background: transparent; }}
QLabel#sub     {{ color: {C["dim"]}; font-size: 10px; letter-spacing: 5px;
                  background: transparent; }}
QLabel#section {{ color: {C["accent"]}; font-size: 10px; letter-spacing: 3px;
                  background: transparent; }}
QLabel#card    {{ background: {C["card"]}; border: 1px solid {C["border"]}; padding: 16px; }}
QLabel#success {{ color: {C["success"]}; font-size: 20px; font-weight: bold;
                  letter-spacing: 3px; background: transparent; }}
QLabel#denied  {{ color: {C["error"]}; font-size: 20px; font-weight: bold;
                  letter-spacing: 3px; background: transparent; }}
QLabel#score   {{ color: {C["accent"]}; font-size: 28px; font-weight: bold;
                  background: transparent; }}

QFrame#line {{ background: {C["border"]}; max-height: 1px; min-height: 1px; }}
"""

# ─── Utilitários ──────────────────────────────────────────────────────────────

def bgr_to_pixmap(bgr: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rh, rw = rgb.shape[:2]
    scale = min(w / rw, h / rh)
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(rw * scale), int(rh * scale)))
        rh, rw = rgb.shape[:2]
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, rw, rh, rw * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def sep():
    f = QFrame(); f.setObjectName("line"); f.setFrameShape(QFrame.HLine)
    return f


def lbl(text, obj=None, align=None):
    l = QLabel(text)
    if obj:   l.setObjectName(obj)
    if align: l.setAlignment(align)
    return l


# ─── Thread da câmera ─────────────────────────────────────────────────────────

class CameraThread(QThread):
    frame_ready = pyqtSignal(object, object)   # (gray_ndarray, hand_info|None)

    def __init__(self, picam2, parent=None):
        super().__init__(parent)
        self.picam2 = picam2
        self._active = True

    def run(self):
        while self._active:
            try:
                frame = self.picam2.capture_array()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                       if len(frame.shape) == 3 else frame.copy()
                hand_info = _detect_hand(gray)
                if hand_info is not None:
                    hand_info["sharpness"] = _sharpness(gray, hand_info["bbox"])
                    if detector_disponivel():
                        x, y, bw, bh = hand_info["bbox"]
                        roi = gray[y:y + bh, x:x + bw]
                        eh_mao, score = is_hand(roi)
                        hand_info["is_hand"]   = eh_mao
                        hand_info["det_score"] = score
                self.frame_ready.emit(gray, hand_info)
            except Exception:
                pass
            self.msleep(33)

    def stop(self):
        self._active = False
        self.wait(2000)


# ─── Tela: Menu principal ─────────────────────────────────────────────────────

class MenuScreen(QWidget):
    def __init__(self, app_ref, parent=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self._blink = True
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(900)

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(80, 50, 80, 40)
        root.setSpacing(0)

        # ── Logo / título ──────────────────────────────────────────────────────
        root.addStretch(1)
        root.addWidget(lbl("PALM BIOMETRICS ID", "title", Qt.AlignCenter))
        root.addSpacing(6)
        root.addWidget(lbl("◈  VASCULAR RECOGNITION SYSTEM  ◈", "sub", Qt.AlignCenter))
        root.addSpacing(28)
        root.addWidget(sep())
        root.addSpacing(36)

        # ── Botões de menu ─────────────────────────────────────────────────────
        items = [
            ("1", "AUTENTICAR",    "Verificação biométrica por veias vasculares"),
            ("2", "CADASTRAR",     "Registrar novo usuário com dados e biometria"),
            ("3", "USUÁRIOS",      "Consultar usuários cadastrados no sistema"),
            ("4", "SAIR",          "Encerrar o sistema com segurança"),
        ]
        callbacks = [
            self.app_ref.show_auth,
            self.app_ref.show_register,
            self.app_ref.show_users,
            QApplication.quit,
        ]

        self._btns = []
        for (num, label, desc), cb in zip(items, callbacks):
            row = QHBoxLayout()
            row.setSpacing(0)

            num_lbl = QLabel(f"[{num}]")
            num_lbl.setFixedWidth(52)
            num_lbl.setStyleSheet(f"color:{C['accent']}; font-size:14px; font-weight:bold;"
                                  f" background:transparent; padding-left:4px;")
            num_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)

            btn = QPushButton(f"  {label}")
            btn.setProperty("class", "menu_btn")
            btn.setMinimumHeight(62)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(cb)

            desc_lbl = QLabel(desc)
            desc_lbl.setStyleSheet(f"color:{C['dim']}; font-size:10px; letter-spacing:1px;"
                                   f" background:transparent; padding-right:8px;")
            desc_lbl.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
            desc_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            row.addWidget(btn)
            row.addWidget(desc_lbl)
            root.addLayout(row)
            root.addSpacing(6)
            self._btns.append(btn)

        root.addSpacing(30)
        root.addWidget(sep())
        root.addSpacing(14)

        # ── Status bar ─────────────────────────────────────────────────────────
        self._status = QLabel()
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(f"color:{C['dim']}; font-size:10px; letter-spacing:2px;"
                                   f" background:transparent;")
        root.addWidget(self._status)
        root.addStretch(1)
        self._tick()

    def _tick(self):
        self._blink = not self._blink
        dot = "●" if self._blink else "○"
        n = len(list_users())
        det = "DETECTOR ATIVO" if detector_disponivel() else "SEM DETECTOR"
        now = datetime.now().strftime("%H:%M:%S")
        self._status.setText(
            f"{dot}  SISTEMA ONLINE  ·  {n} USUÁRIO{'S' if n!=1 else ''}  ·  {det}  ·  {now}"
        )

    def keyPressEvent(self, event):
        k = event.key()
        if   k == Qt.Key_1: self.app_ref.show_auth()
        elif k == Qt.Key_2: self.app_ref.show_register()
        elif k == Qt.Key_3: self.app_ref.show_users()
        elif k == Qt.Key_4: QApplication.quit()

    def showEvent(self, e):
        super().showEvent(e)
        self._tick()


# ─── Tela: Cadastro de dados ──────────────────────────────────────────────────

class RegisterScreen(QWidget):
    def __init__(self, app_ref, parent=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(80, 40, 80, 40)
        root.setSpacing(0)

        # Header
        root.addWidget(lbl("NOVO CADASTRO", "section"))
        root.addSpacing(4)
        root.addWidget(sep())
        root.addSpacing(28)

        # Formulário
        form = QGridLayout()
        form.setSpacing(12)
        form.setHorizontalSpacing(20)

        def field_lbl(txt):
            l = QLabel(txt)
            l.setStyleSheet(f"color:{C['dim']}; font-size:10px; letter-spacing:2px;"
                            f" background:transparent;")
            return l

        self._nome  = QLineEdit(); self._nome.setPlaceholderText("Nome completo")
        self._doc   = QLineEdit(); self._doc.setPlaceholderText("CPF / RG")
        self._depto = QLineEdit(); self._depto.setPlaceholderText("Ex: Tecnologia")
        self._cargo = QLineEdit(); self._cargo.setPlaceholderText("Ex: Analista")
        self._email = QLineEdit(); self._email.setPlaceholderText("email@exemplo.com")

        form.addWidget(field_lbl("NOME COMPLETO  *"),  0, 0, 1, 2)
        form.addWidget(self._nome,                     1, 0, 1, 2)
        form.addWidget(field_lbl("DOCUMENTO  (CPF / RG)"), 2, 0)
        form.addWidget(field_lbl("EMAIL"),             2, 1)
        form.addWidget(self._doc,                      3, 0)
        form.addWidget(self._email,                    3, 1)
        form.addWidget(field_lbl("DEPARTAMENTO"),      4, 0)
        form.addWidget(field_lbl("CARGO / FUNÇÃO"),    4, 1)
        form.addWidget(self._depto,                    5, 0)
        form.addWidget(self._cargo,                    5, 1)
        root.addLayout(form)
        root.addSpacing(32)

        # Mensagem de erro
        self._msg = QLabel("")
        self._msg.setAlignment(Qt.AlignCenter)
        self._msg.setStyleSheet(f"color:{C['error']}; font-size:11px; background:transparent;")
        root.addWidget(self._msg)
        root.addSpacing(8)
        root.addStretch()
        root.addWidget(sep())
        root.addSpacing(16)

        # Botões
        row = QHBoxLayout()
        btn_back = QPushButton("← VOLTAR")
        btn_back.setProperty("class", "secondary")
        btn_back.clicked.connect(self.app_ref.show_menu)
        btn_back.setCursor(Qt.PointingHandCursor)

        btn_next = QPushButton("PRÓXIMO  →")
        btn_next.setProperty("class", "primary")
        btn_next.clicked.connect(self._on_next)
        btn_next.setCursor(Qt.PointingHandCursor)

        row.addWidget(btn_back)
        row.addStretch()
        row.addWidget(btn_next)
        root.addLayout(row)

    def _on_next(self):
        nome = self._nome.text().strip()
        if not nome:
            self._msg.setText("● Nome é obrigatório.")
            return
        existing = [n for _, n in list_users()]
        if nome in existing:
            self._msg.setText(f"● Usuário '{nome}' já existe.")
            return
        self._msg.setText("")
        user_data = {
            "nome":        nome,
            "documento":   self._doc.text().strip(),
            "departamento": self._depto.text().strip(),
            "cargo":       self._cargo.text().strip(),
            "email":       self._email.text().strip(),
        }
        self.app_ref.show_biometric(user_data)

    def showEvent(self, e):
        super().showEvent(e)
        self._nome.clear(); self._doc.clear()
        self._depto.clear(); self._cargo.clear()
        self._email.clear(); self._msg.clear()
        self._nome.setFocus()


# ─── Tela: Cadastro biométrico ────────────────────────────────────────────────

class BiometricScreen(QWidget):
    def __init__(self, app_ref, parent=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self._user_data: dict = {}
        self._templates: list = []
        self._stable_since: float | None = None
        self._capturing = False
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 24)
        root.setSpacing(0)

        # Header
        hdr = QHBoxLayout()
        self._title_lbl = QLabel("CADASTRO BIOMÉTRICO")
        self._title_lbl.setObjectName("section")
        self._user_lbl = QLabel("")
        self._user_lbl.setStyleSheet(f"color:{C['accent']}; font-size:13px;"
                                     f" font-weight:bold; background:transparent;")
        hdr.addWidget(self._title_lbl)
        hdr.addStretch()
        hdr.addWidget(self._user_lbl)
        root.addLayout(hdr)
        root.addSpacing(4)
        root.addWidget(sep())
        root.addSpacing(12)

        # Progresso
        prog_row = QHBoxLayout()
        self._prog_lbl = QLabel("0 / 16")
        self._prog_lbl.setStyleSheet(f"color:{C['accent']}; font-size:18px;"
                                     f" font-weight:bold; background:transparent;")
        self._prog_bar = QFrame()
        self._prog_bar.setFixedHeight(4)
        self._prog_bar.setStyleSheet(f"background:{C['border']};")
        self._prog_fill = QFrame(self._prog_bar)
        self._prog_fill.setFixedHeight(4)
        self._prog_fill.setStyleSheet(f"background:{C['accent']};")
        self._prog_fill.setFixedWidth(0)
        prog_row.addWidget(self._prog_lbl)
        prog_row.addWidget(self._prog_bar)
        root.addLayout(prog_row)
        root.addSpacing(10)

        # Camera feed
        self._cam_lbl = QLabel()
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setStyleSheet(f"background:{C['card2']}; border:1px solid {C['border']};")
        self._cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._cam_lbl)
        root.addSpacing(10)

        # Mensagem de status
        self._status_lbl = QLabel("Posicione a palma na área indicada")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setStyleSheet(f"color:{C['dim']}; font-size:11px;"
                                       f" letter-spacing:2px; background:transparent;")
        root.addWidget(self._status_lbl)
        root.addSpacing(12)
        root.addWidget(sep())
        root.addSpacing(12)

        btn_cancel = QPushButton("✕  CANCELAR CADASTRO")
        btn_cancel.setProperty("class", "danger")
        btn_cancel.clicked.connect(self._cancel)
        btn_cancel.setCursor(Qt.PointingHandCursor)
        btn_cancel.setFixedWidth(220)
        root.addWidget(btn_cancel, alignment=Qt.AlignRight)

    def start(self, user_data: dict):
        self._user_data = user_data
        self._templates = []
        self._stable_since = None
        self._capturing = False
        self._user_lbl.setText(user_data["nome"].upper())
        self._update_progress()

    def on_frame(self, gray: np.ndarray, hand_info: dict | None):
        if self._capturing:
            return

        progress = 0.0
        _, is_ready = draw_overlay(_to_bgr(gray), hand_info, 0.0)

        if is_ready:
            if self._stable_since is None:
                self._stable_since = time.time()
            elapsed = time.time() - self._stable_since
            progress = min(elapsed / STABLE_TIME, 1.0)
            if progress >= 1.0:
                self._do_capture(gray)
        else:
            self._stable_since = None

        overlay, _ = draw_overlay(_to_bgr(gray), hand_info, progress)
        pw = self._cam_lbl.width() or 640
        ph = self._cam_lbl.height() or 480
        self._cam_lbl.setPixmap(bgr_to_pixmap(overlay, pw, ph))

    def _do_capture(self, gray: np.ndarray):
        self._capturing = True
        self._stable_since = None
        processed = preprocess_veins(gray)
        template  = extract_template(processed)
        self._templates.append(template)
        n = len(self._templates)
        self._update_progress()
        self._status_lbl.setText(f"✓ Template {n}/{NUM_CAPTURAS} capturado — reposicione a mão")
        self._status_lbl.setStyleSheet(f"color:{C['success']}; font-size:11px;"
                                       f" letter-spacing:2px; background:transparent;")
        QTimer.singleShot(1200, self._reset_status)

        if n >= NUM_CAPTURAS:
            QTimer.singleShot(600, self._finish)
            return
        QTimer.singleShot(400, self._unlock)

    def _unlock(self):
        self._capturing = False

    def _reset_status(self):
        self._status_lbl.setText("Posicione a palma na área indicada")
        self._status_lbl.setStyleSheet(f"color:{C['dim']}; font-size:11px;"
                                       f" letter-spacing:2px; background:transparent;")

    def _update_progress(self):
        n = len(self._templates)
        self._prog_lbl.setText(f"{n} / {NUM_CAPTURAS}")
        # update bar fill width
        QTimer.singleShot(0, lambda: self._resize_bar(n))

    def _resize_bar(self, n):
        total_w = self._prog_bar.width()
        fill_w  = int(total_w * n / NUM_CAPTURAS)
        self._prog_fill.setFixedWidth(fill_w)

    def _finish(self):
        d = self._user_data
        user_dir = os.path.join("USUARIOS", d["nome"])
        os.makedirs(user_dir, exist_ok=True)
        save_templates(self._templates, os.path.join(user_dir, "templates.json"))
        add_user(d["nome"], d.get("documento",""), d.get("departamento",""),
                 d.get("cargo",""), d.get("email",""))
        self._status_lbl.setText(f"✓ {d['nome']} cadastrado com sucesso!")
        self._status_lbl.setStyleSheet(f"color:{C['success']}; font-size:13px;"
                                       f" font-weight:bold; letter-spacing:2px;"
                                       f" background:transparent;")
        QTimer.singleShot(2200, self.app_ref.show_menu)

    def _cancel(self):
        self.app_ref.show_menu()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, lambda: self._resize_bar(len(self._templates)))


# ─── Tela: Autenticação ───────────────────────────────────────────────────────

class AuthScreen(QWidget):
    STATE_SCANNING = "scanning"
    STATE_RESULT   = "result"

    def __init__(self, app_ref, parent=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self._state = self.STATE_SCANNING
        self._stable_since: float | None = None
        self._processing = False
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 24)
        root.setSpacing(0)

        # Header
        root.addWidget(lbl("AUTENTICAÇÃO BIOMÉTRICA", "section"))
        root.addSpacing(4)
        root.addWidget(sep())
        root.addSpacing(12)

        # Stack: câmera | resultado
        self._stack = QStackedWidget()

        # ── Página câmera ──────────────────────────────────────────────────────
        cam_page = QWidget()
        cam_lay = QVBoxLayout(cam_page)
        cam_lay.setContentsMargins(0, 0, 0, 0)
        self._cam_lbl = QLabel()
        self._cam_lbl.setAlignment(Qt.AlignCenter)
        self._cam_lbl.setStyleSheet(f"background:{C['card2']}; border:1px solid {C['border']};")
        self._cam_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cam_lay.addWidget(self._cam_lbl)
        cam_lay.addSpacing(8)
        self._scan_msg = QLabel("Posicione a palma na área indicada")
        self._scan_msg.setAlignment(Qt.AlignCenter)
        self._scan_msg.setStyleSheet(f"color:{C['dim']}; font-size:11px;"
                                     f" letter-spacing:2px; background:transparent;")
        cam_lay.addWidget(self._scan_msg)

        # ── Página resultado ───────────────────────────────────────────────────
        res_page = QWidget()
        res_lay = QVBoxLayout(res_page)
        res_lay.setContentsMargins(20, 20, 20, 20)
        res_lay.setSpacing(0)

        self._res_status = QLabel()
        self._res_status.setObjectName("success")
        self._res_status.setAlignment(Qt.AlignCenter)
        res_lay.addWidget(self._res_status)
        res_lay.addSpacing(20)
        res_lay.addWidget(sep())
        res_lay.addSpacing(20)

        # Card do usuário
        card = QFrame()
        card.setStyleSheet(f"QFrame {{ background:{C['card']}; border:1px solid {C['border']};"
                           f" padding: 0; }}")
        card_lay = QGridLayout(card)
        card_lay.setContentsMargins(24, 20, 24, 20)
        card_lay.setSpacing(10)

        def info_lbl(txt, bold=False, color=None):
            l = QLabel(txt)
            s = f"background:transparent; color:{color or C['text']};"
            if bold: s += " font-weight:bold; font-size:16px;"
            l.setStyleSheet(s)
            return l

        self._r_nome   = info_lbl("", bold=True, color=C["accent"])
        self._r_cargo  = info_lbl("", color=C["dim"])
        self._r_depto  = info_lbl("")
        self._r_doc    = info_lbl("")
        self._r_email  = info_lbl("")
        self._r_data   = info_lbl("")
        self._r_score  = QLabel()
        self._r_score.setObjectName("score")

        # Ícone / avatar placeholder
        avatar = QLabel("◈")
        avatar.setFixedSize(72, 72)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(f"background:{C['card2']}; border:1px solid {C['border']};"
                             f" color:{C['accent']}; font-size:28px;")

        card_lay.addWidget(avatar,        0, 0, 4, 1)
        card_lay.addWidget(self._r_nome,  0, 1, 1, 2)
        card_lay.addWidget(self._r_cargo, 1, 1, 1, 2)
        card_lay.addWidget(self._r_depto, 2, 1)
        card_lay.addWidget(self._r_doc,   2, 2)
        card_lay.addWidget(self._r_email, 3, 1)
        card_lay.addWidget(self._r_data,  3, 2)
        card_lay.setColumnStretch(1, 1)
        card_lay.setColumnStretch(2, 1)
        res_lay.addWidget(card)
        res_lay.addSpacing(16)

        # Score bar
        score_row = QHBoxLayout()
        score_row.addWidget(lbl("SIMILARIDADE:", None))
        score_row.addSpacing(8)
        self._score_bar_bg = QFrame()
        self._score_bar_bg.setFixedHeight(6)
        self._score_bar_bg.setStyleSheet(f"background:{C['border']};")
        self._score_fill = QFrame(self._score_bar_bg)
        self._score_fill.setFixedHeight(6)
        self._score_fill.setStyleSheet(f"background:{C['accent']};")
        self._score_fill.setFixedWidth(0)
        score_row.addWidget(self._score_bar_bg)
        score_row.addSpacing(8)
        score_row.addWidget(self._r_score)
        res_lay.addLayout(score_row)
        res_lay.addStretch()

        # Botões resultado
        res_lay.addWidget(sep())
        res_lay.addSpacing(12)
        btn_row = QHBoxLayout()
        btn_new = QPushButton("↩  NOVA VERIFICAÇÃO")
        btn_new.setProperty("class", "primary")
        btn_new.clicked.connect(self._reset)
        btn_new.setCursor(Qt.PointingHandCursor)
        btn_back = QPushButton("← MENU")
        btn_back.setProperty("class", "secondary")
        btn_back.clicked.connect(self.app_ref.show_menu)
        btn_back.setCursor(Qt.PointingHandCursor)
        btn_row.addWidget(btn_back)
        btn_row.addStretch()
        btn_row.addWidget(btn_new)
        res_lay.addLayout(btn_row)

        self._stack.addWidget(cam_page)   # index 0
        self._stack.addWidget(res_page)   # index 1
        root.addWidget(self._stack)
        root.addSpacing(12)

        # Botão voltar (visível só na câmera)
        self._btn_back = QPushButton("← VOLTAR")
        self._btn_back.setProperty("class", "secondary")
        self._btn_back.setFixedWidth(120)
        self._btn_back.clicked.connect(self.app_ref.show_menu)
        self._btn_back.setCursor(Qt.PointingHandCursor)
        root.addWidget(self._btn_back, alignment=Qt.AlignLeft)

    def _reset(self):
        self._state = self.STATE_SCANNING
        self._stable_since = None
        self._processing = False
        self._stack.setCurrentIndex(0)
        self._btn_back.setVisible(True)
        self._scan_msg.setText("Posicione a palma na área indicada")
        self._scan_msg.setStyleSheet(f"color:{C['dim']}; font-size:11px;"
                                     f" letter-spacing:2px; background:transparent;")

    def on_frame(self, gray: np.ndarray, hand_info: dict | None):
        if self._state != self.STATE_SCANNING or self._processing:
            return

        progress = 0.0
        _, is_ready = draw_overlay(_to_bgr(gray), hand_info, 0.0)

        if is_ready:
            if self._stable_since is None:
                self._stable_since = time.time()
            elapsed = time.time() - self._stable_since
            progress = min(elapsed / STABLE_TIME, 1.0)
            if progress >= 1.0:
                self._do_match(gray)
        else:
            self._stable_since = None

        overlay, _ = draw_overlay(_to_bgr(gray), hand_info, progress)
        pw = self._cam_lbl.width() or 640
        ph = self._cam_lbl.height() or 480
        self._cam_lbl.setPixmap(bgr_to_pixmap(overlay, pw, ph))

    def _do_match(self, gray: np.ndarray):
        self._processing = True
        self._stable_since = None
        self._scan_msg.setText("Processando...")

        processed = preprocess_veins(gray)
        probe     = extract_template(processed)

        usuarios_dir = "USUARIOS"
        if not os.path.isdir(usuarios_dir):
            self._show_denied(0.0)
            return

        best_nome  = None
        best_score = 0.0
        best_match = False

        for u in os.listdir(usuarios_dir):
            tpath = os.path.join(usuarios_dir, u, "templates.json")
            if not os.path.exists(tpath):
                continue
            stored = load_templates(tpath)
            matched, score = match_template(probe, stored)
            if score > best_score:
                best_score = score
                best_nome  = u
                best_match = matched

        if best_match and best_nome:
            info = get_user_info(best_nome) or {}
            self._show_granted(best_nome, best_score, info)
        else:
            self._show_denied(best_score)

    def _show_granted(self, nome: str, score: float, info: dict):
        self._res_status.setObjectName("success")
        self._res_status.setText("●  ACESSO AUTORIZADO")
        self._res_status.setStyleSheet(f"color:{C['success']}; font-size:20px;"
                                       f" font-weight:bold; letter-spacing:3px;"
                                       f" background:transparent;")
        self._r_nome.setText(nome.upper())
        self._r_cargo.setText(info.get("cargo", "—"))
        self._r_depto.setText(f"Depto: {info.get('departamento','—')}")
        self._r_doc.setText(f"Doc: {info.get('documento','—')}")
        self._r_email.setText(info.get("email", ""))
        self._r_data.setText(f"Cadastro: {info.get('data_cadastro','—')}")
        self._r_score.setText(f"{score*100:.1f}%")
        self._r_score.setStyleSheet(f"color:{C['success']}; font-size:22px;"
                                    f" font-weight:bold; background:transparent;")
        self._update_score_bar(score, C["success"])
        self._state = self.STATE_RESULT
        self._btn_back.setVisible(False)
        self._stack.setCurrentIndex(1)

    def _show_denied(self, score: float):
        self._res_status.setText("✕  ACESSO NEGADO")
        self._res_status.setStyleSheet(f"color:{C['error']}; font-size:20px;"
                                       f" font-weight:bold; letter-spacing:3px;"
                                       f" background:transparent;")
        for lbl_w in (self._r_nome, self._r_cargo, self._r_depto,
                      self._r_doc, self._r_email, self._r_data):
            lbl_w.setText("—")
        self._r_nome.setText("USUÁRIO NÃO RECONHECIDO")
        self._r_score.setText(f"{score*100:.1f}%")
        self._r_score.setStyleSheet(f"color:{C['error']}; font-size:22px;"
                                    f" font-weight:bold; background:transparent;")
        self._update_score_bar(score, C["error"])
        self._state = self.STATE_RESULT
        self._btn_back.setVisible(False)
        self._stack.setCurrentIndex(1)

    def _update_score_bar(self, score: float, color: str):
        self._score_fill.setStyleSheet(f"background:{color};")
        QTimer.singleShot(50, lambda: self._score_fill.setFixedWidth(
            int(self._score_bar_bg.width() * min(score, 1.0))
        ))

    def showEvent(self, e):
        super().showEvent(e)
        self._reset()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        QTimer.singleShot(0, lambda: self._update_score_bar(0, C["accent"]))


# ─── Tela: Lista de usuários ──────────────────────────────────────────────────

class UsersScreen(QWidget):
    def __init__(self, app_ref, parent=None):
        super().__init__(parent)
        self.app_ref = app_ref
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(40, 24, 40, 24)
        root.setSpacing(0)

        root.addWidget(lbl("USUÁRIOS CADASTRADOS", "section"))
        root.addSpacing(4)
        root.addWidget(sep())
        root.addSpacing(14)

        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["NOME", "DOCUMENTO", "DEPARTAMENTO", "CARGO", "EMAIL", "CADASTRO"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setShowGrid(True)
        self._table.setAlternatingRowColors(False)
        self._table.setStyleSheet(
            self._table.styleSheet() +
            f"QTableWidget::item:alternate {{ background: {C['card2']}; }}"
        )
        root.addWidget(self._table)
        root.addSpacing(14)

        # Rodapé
        row = QHBoxLayout()
        self._count_lbl = QLabel()
        self._count_lbl.setStyleSheet(f"color:{C['dim']}; font-size:11px;"
                                      f" background:transparent;")
        btn_del = QPushButton("✕  REMOVER SELECIONADO")
        btn_del.setProperty("class", "danger")
        btn_del.clicked.connect(self._remove_selected)
        btn_del.setCursor(Qt.PointingHandCursor)
        btn_del.setFixedWidth(220)
        btn_back = QPushButton("← MENU")
        btn_back.setProperty("class", "secondary")
        btn_back.clicked.connect(self.app_ref.show_menu)
        btn_back.setCursor(Qt.PointingHandCursor)
        row.addWidget(btn_back)
        row.addSpacing(12)
        row.addWidget(self._count_lbl)
        row.addStretch()
        row.addWidget(btn_del)
        root.addWidget(sep())
        root.addSpacing(12)
        root.addLayout(row)

    def _load(self):
        users = list_users_full()
        self._table.setRowCount(len(users))
        for i, u in enumerate(users):
            for j, key in enumerate(["nome","documento","departamento","cargo","email","data_cadastro"]):
                item = QTableWidgetItem(u.get(key) or "")
                item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self._table.setItem(i, j, item)
            # highlight se tem templates
            tpath = os.path.join("USUARIOS", u["nome"], "templates.json")
            color = C["text"] if os.path.exists(tpath) else C["dim"]
            for j in range(6):
                if self._table.item(i, j):
                    self._table.item(i, j).setForeground(QColor(color))
        n = len(users)
        self._count_lbl.setText(f"{n} usuário{'s' if n!=1 else ''} no sistema")

    def _remove_selected(self):
        row = self._table.currentRow()
        if row < 0:
            return
        nome_item = self._table.item(row, 0)
        if not nome_item:
            return
        nome = nome_item.text()
        dlg = QMessageBox(self)
        dlg.setWindowTitle("Confirmar remoção")
        dlg.setText(f"Remover '{nome}' e todos os seus templates?")
        dlg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        dlg.setDefaultButton(QMessageBox.No)
        dlg.setStyleSheet(QSS)
        if dlg.exec_() == QMessageBox.Yes:
            import shutil
            user_dir = os.path.join("USUARIOS", nome)
            if os.path.isdir(user_dir):
                shutil.rmtree(user_dir)
            remove_user(nome)
            self._load()

    def showEvent(self, e):
        super().showEvent(e)
        self._load()


# ─── Janela principal ─────────────────────────────────────────────────────────

class PalmApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Palm Biometrics ID")
        self.setMinimumSize(900, 620)
        self.showFullScreen()

        init_db()
        self._init_leds()
        self._init_camera()

        # Stack de telas
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu     = MenuScreen(self)
        self._register = RegisterScreen(self)
        self._biometric = BiometricScreen(self)
        self._auth     = AuthScreen(self)
        self._users    = UsersScreen(self)

        for w in (self._menu, self._register, self._biometric, self._auth, self._users):
            self._stack.addWidget(w)

        # Câmera → roteador de frames
        if self._cam_thread:
            self._cam_thread.frame_ready.connect(self._route_frame)
            self._cam_thread.start()

        self.show_menu()

    # ── LEDs ──────────────────────────────────────────────────────────────────
    def _init_leds(self):
        try:
            from leds import setup as led_setup, liga_leds
            led_setup()
            liga_leds()
            self._leds_ok = True
        except Exception:
            self._leds_ok = False

    # ── Câmera ────────────────────────────────────────────────────────────────
    def _init_camera(self):
        self._picam2 = None
        self._cam_thread = None
        try:
            from camera import setup_camera
            self._picam2 = setup_camera()
            self._picam2.start()
            self._cam_thread = CameraThread(self._picam2)
        except Exception as e:
            print(f"[aviso] Câmera não disponível: {e}")

    # ── Roteamento de frames para a tela ativa ────────────────────────────────
    def _route_frame(self, gray, hand_info):
        current = self._stack.currentWidget()
        if hasattr(current, "on_frame"):
            current.on_frame(gray, hand_info)

    # ── Navegação ─────────────────────────────────────────────────────────────
    def show_menu(self):
        self._stack.setCurrentWidget(self._menu)

    def show_register(self):
        self._stack.setCurrentWidget(self._register)

    def show_biometric(self, user_data: dict):
        self._biometric.start(user_data)
        self._stack.setCurrentWidget(self._biometric)

    def show_auth(self):
        self._stack.setCurrentWidget(self._auth)

    def show_users(self):
        self._stack.setCurrentWidget(self._users)

    # ── Tecla ESC volta ao menu / sai do fullscreen ───────────────────────────
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self._stack.currentWidget() is self._menu:
                self.close()
            else:
                self.show_menu()
        elif event.key() == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            self._stack.currentWidget().keyPressEvent(event)

    def closeEvent(self, e):
        if self._cam_thread:
            self._cam_thread.stop()
        if self._picam2:
            try:
                self._picam2.stop()
                self._picam2.close()
            except Exception:
                pass
        if self._leds_ok:
            try:
                from leds import desliga_leds, cleanup as led_cleanup
                desliga_leds()
                led_cleanup()
            except Exception:
                pass
        e.accept()


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    app.setApplicationName("Palm Biometrics ID")
    window = PalmApp()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
