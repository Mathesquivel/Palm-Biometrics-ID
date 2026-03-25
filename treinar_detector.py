"""
Treina o detector binário mão / não-mão usando HOG + SVM (OpenCV).

Não requer TensorFlow nem scikit-learn — usa apenas OpenCV (cv2.ml.SVM).

Uso:
    python treinar_detector.py

Entrada:  detector_dataset/mao/     (imagens PNG de palma)
          detector_dataset/nao_mao/ (imagens PNG de não-palma)
Saída:    detector_mao.xml          (modelo SVM treinado)
"""

import os
import cv2
import numpy as np
import random

DATASET_DIR = "detector_dataset"
MODEL_PATH  = "detector_mao.xml"

# Tamanho normalizado para extração HOG
WIN_SIZE = (64, 64)

# Labels: 1 = mão, -1 = não-mão
LABEL_MAO     =  1
LABEL_NAO_MAO = -1


# ─── HOG ──────────────────────────────────────────────────────────────────────

def _make_hog() -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        _winSize    = WIN_SIZE,
        _blockSize  = (16, 16),
        _blockStride= (8, 8),
        _cellSize   = (8, 8),
        _nbins      = 9,
    )


HOG = _make_hog()
# Dimensão do vetor HOG: ((64-16)/8+1)² × (16/8)² × 9 = 49 × 36 = 1764
HOG_DIM = int(HOG.getDescriptorSize())


def _extract_hog(img: np.ndarray) -> np.ndarray:
    """Redimensiona para WIN_SIZE e extrai vetor HOG de 1764 floats."""
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img, WIN_SIZE)
    descriptor = HOG.compute(resized)
    return descriptor.flatten().astype(np.float32)


# ─── Carregamento de dados ────────────────────────────────────────────────────

def _carregar_classe(pasta: str, label: int) -> tuple[list, list]:
    features, labels = [], []
    if not os.path.isdir(pasta):
        return features, labels
    for fname in os.listdir(pasta):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        path = os.path.join(pasta, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [aviso] não consegui ler {path}")
            continue
        feat = _extract_hog(img)
        features.append(feat)
        labels.append(label)
    return features, labels


def _augmentar(img: np.ndarray) -> list[np.ndarray]:
    """Gera variações leves da imagem para aumentar dataset."""
    h, w = img.shape[:2]
    augmented = []

    # Pequenas rotações
    for angle in [-8, -4, 4, 8]:
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h))
        augmented.append(rot)

    # Flip horizontal (palma espelhada)
    augmented.append(cv2.flip(img, 1))

    # Variação de brilho
    for gamma in [0.8, 1.3]:
        table = np.array([min(255, int((i / 255.0) ** gamma * 255))
                          for i in range(256)], dtype=np.uint8)
        augmented.append(cv2.LUT(img, table))

    return augmented


def _carregar_com_aug(pasta: str, label: int,
                      augmentar: bool = True) -> tuple[list, list]:
    features, labels = [], []
    if not os.path.isdir(pasta):
        return features, labels

    arquivos = [f for f in os.listdir(pasta)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    for fname in arquivos:
        img = cv2.imread(os.path.join(pasta, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        features.append(_extract_hog(img))
        labels.append(label)

        if augmentar:
            for aug in _augmentar(img):
                features.append(_extract_hog(aug))
                labels.append(label)

    return features, labels


# ─── Treino ───────────────────────────────────────────────────────────────────

def treinar():
    pasta_mao     = os.path.join(DATASET_DIR, "mao")
    pasta_nao_mao = os.path.join(DATASET_DIR, "nao_mao")

    n_mao     = len([f for f in os.listdir(pasta_mao)
                     if f.endswith(".png")]) if os.path.isdir(pasta_mao) else 0
    n_nao_mao = len([f for f in os.listdir(pasta_nao_mao)
                     if f.endswith(".png")]) if os.path.isdir(pasta_nao_mao) else 0

    print(f"Dataset encontrado:  mao={n_mao}  nao_mao={n_nao_mao}")

    if n_mao < 10 or n_nao_mao < 10:
        print("\nImagens insuficientes. Execute primeiro:")
        print("  python capturar_treino_detector.py")
        print("  (mínimo 30 por classe, recomendado 80+)")
        return

    print("Extraindo features HOG (com data augmentation)...")
    feat_mao,     lab_mao     = _carregar_com_aug(pasta_mao,     LABEL_MAO,     augmentar=True)
    feat_nao_mao, lab_nao_mao = _carregar_com_aug(pasta_nao_mao, LABEL_NAO_MAO, augmentar=True)

    X = np.array(feat_mao + feat_nao_mao, dtype=np.float32)
    y = np.array(lab_mao  + lab_nao_mao,  dtype=np.int32)

    print(f"Amostras totais após augmentation: {len(X)}")
    print(f"  mao={lab_mao.count(LABEL_MAO) if isinstance(lab_mao, list) else sum(1 for l in y if l == LABEL_MAO)}")

    # Embaralhar
    idx = list(range(len(X)))
    random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    # Split treino / validação (80/20)
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\nTreinando SVM (kernel RBF, autotraining)...")
    train_data = cv2.ml.TrainData_create(X_train, cv2.ml.ROW_SAMPLE, y_train)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                          1000, 1e-6))
    svm.trainAuto(train_data, kFold=5)

    # ── Avaliação ──────────────────────────────────────────────────────────
    _, pred_train = svm.predict(X_train)
    acc_train = float(np.mean(pred_train.flatten() == y_train)) * 100

    _, pred_val = svm.predict(X_val)
    acc_val = float(np.mean(pred_val.flatten() == y_val)) * 100

    # Matriz de confusão na validação
    vp = int(np.sum((pred_val.flatten() ==  1) & (y_val ==  1)))  # verdadeiro positivo
    fp = int(np.sum((pred_val.flatten() ==  1) & (y_val == -1)))  # falso positivo
    vn = int(np.sum((pred_val.flatten() == -1) & (y_val == -1)))  # verdadeiro negativo
    fn = int(np.sum((pred_val.flatten() == -1) & (y_val ==  1)))  # falso negativo

    print(f"\n{'─'*40}")
    print(f"  Acurácia treino:    {acc_train:.1f}%")
    print(f"  Acurácia validação: {acc_val:.1f}%")
    print(f"\n  Matriz de confusão (validação):")
    print(f"              Pred MAO  Pred NAO")
    print(f"  Real MAO:     {vp:4d}      {fn:4d}")
    print(f"  Real NAO:     {fp:4d}      {vn:4d}")

    total_val = len(y_val)
    if total_val > 0:
        precisao  = vp / (vp + fp) * 100 if (vp + fp) > 0 else 0
        recall    = vp / (vp + fn) * 100 if (vp + fn) > 0 else 0
        especif   = vn / (vn + fp) * 100 if (vn + fp) > 0 else 0
        print(f"\n  Precisão  (quando diz mão, está certo?): {precisao:.1f}%")
        print(f"  Recall    (detecta todas as mãos?):       {recall:.1f}%")
        print(f"  Especif.  (rejeita não-mãos?):            {especif:.1f}%")

    print(f"{'─'*40}")

    if acc_val < 80:
        print("\n  Aviso: acurácia < 80%. Considere capturar mais imagens.")
    elif acc_val < 90:
        print("\n  Bom resultado. Para melhorar, adicione mais variedade nas imagens.")
    else:
        print("\n  Excelente resultado!")

    svm.save(MODEL_PATH)
    print(f"\nModelo salvo: {MODEL_PATH}")
    print("Para usar: o detector é carregado automaticamente pelo sistema.")


if __name__ == "__main__":
    treinar()
