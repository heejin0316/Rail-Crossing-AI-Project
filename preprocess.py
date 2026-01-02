import cv2
import numpy as np

def load_and_preprocess_image(path, size=(256, 256), is_mask=False):
    """
    path: 이미지 파일 경로 (예: 'imagesLevelCrossing/0001_1.jpg')
    size: resize할 최종 크기 (가로, 세로)
    is_mask: rail 마스크일 때 True, 일반 RGB 이미지일 때 False
    """

    # 1) 이미지 읽기
    if is_mask:
        # rail 마스크는 흑백 이미지로 읽기
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)   # shape: (H, W)
    else:
        # 일반 이미지는 컬러로 읽기 (BGR → RGB 변환)
        img = cv2.imread(path, cv2.IMREAD_COLOR)       # shape: (H, W, 3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    # 2) resize
    img = cv2.resize(img, size)  # (세로, 가로) 순이지만 size가 (가로,세로)라 그대로 줘도 됨

    # 3) float으로 변환 후 0~1로 normalize
    img = img.astype("float32") / 255.0

    return img

