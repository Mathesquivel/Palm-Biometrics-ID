import cv2
import numpy as np

TARGET_SIZE = (512, 512)


def preprocess_veins(img):
    """
    Pipeline de pré-processamento para realce de veias da palma.

    Etapas:
      1. Conversão para escala de cinza
      2. Redimensionamento para tamanho fixo
      3. Filtro bilateral (remove ruído preservando bordas das veias)
      4. Transformada Black-Hat morfológica (extrai estruturas escuras = veias)
      5. CLAHE (realce de contraste local adaptativo)
      6. Correção gamma (amplifica contraste das veias)
      7. Normalização final
    """
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Tamanho fixo para processamento consistente
        gray = cv2.resize(gray, TARGET_SIZE)

        # Filtro bilateral: remove granulação sem borrar bordas das veias
        smooth = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Black-Hat: closing(img) - img
        # Extrai estruturas escuras (veias) sobre fundo claro (iluminação IR)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
        veins = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)

        # CLAHE para realce de contraste local nas regiões de veias
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(veins)

        # Gamma > 1 escurece pixels escuros → veias mais proeminentes
        gamma = 1.5
        table = np.array(
            [((i / 255.0) ** gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8
        )
        enhanced = cv2.LUT(enhanced, table)

        # Normalização para intervalo completo [0, 255]
        normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)

        return normalized

    except Exception as e:
        print(f"[ERRO - PREPROCESSAMENTO]: {e}")
        return img
