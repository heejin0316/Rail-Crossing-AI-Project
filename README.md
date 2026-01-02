# 🚂 철도 건널목 분류 AI (Level Crossing Classification)

이 프로젝트는 철도 건널목 이미지를 분석하여 장애물 유무와 선로 좌표(15개 타겟)를 예측하는 AI 모델을 구축하는 과제입니다.
**ResNet18**을 기반으로 **7-Channel Input (이전 프레임 + 현재 프레임 + 마스크)** 구조를 사용하여 시계열적 특성과 공간적 특성을 동시에 학습하도록 설계했습니다.

---

## 📂 프로젝트 구조 & 파일 설명

### 1. 데이터 전처리 및 무결성 검사 (Preprocessing)
데이터를 학습에 넣기 전, 파일이 실제로 존재하는지 확인하고 정리하는 단계입니다.

* **`scene_summary.py`**
    * 이미지들을 Scene 단위(`0001_1.jpg`, `0001_rail.png` 등)로 묶어서 정리합니다.
    * 결과물: `scene_summary.csv` (각 scene별 파일 개수 및 목록 요약)
* **`check_images.py`**
    * `train.csv`에 적힌 파일명(`image-3sec`, `image`, `segmentedRailImage`)이 실제 폴더에 존재하는지 전수 조사합니다.
    * 누락된 파일이 없음을 확인하여 학습 중 에러를 방지했습니다.
* **`check_columns.py`**
    * CSV 파일의 컬럼명이 코드와 일치하는지 확인하는 유틸리티입니다.
* **`preprocess.py`**
    * **핵심 함수:** `load_and_preprocess_image()`
    * 이미지를 로드하여 `256x256`으로 Resize하고, 픽셀값을 `0~1`로 Normalize 합니다.

### 2. 데이터셋 및 파이프라인 (Dataset & Pipeline)
PyTorch 모델이 학습할 수 있는 형태(Tensor)로 데이터를 변환하여 공급합니다.

* **`my_dataset.py`**
    * `RailDataset` 클래스가 정의되어 있습니다.
    * **입력(Input):** 3초 전 이미지(3ch) + 현재 이미지(3ch) + 레일 마스크(1ch) = **총 7채널**
    * **출력(Target):** 15개의 회귀 값 (장애물 확률, 좌표 x, y, dx, dy 등)
    * 데이터를 모델에 넣기 좋게 `(Batch, 7, 224, 224)` 형태로 변환합니다.

### 3. 모델 아키텍처 (Model Architecture) ⭐
* **`model.py`**
    * **Base Model:** ResNet18 (Pretrained X, 구조만 차용)
    * **Custom Input:** 첫 번째 Conv 레이어를 수정하여 **7채널 입력**을 받을 수 있게 변경했습니다.
    * **Custom Output:** 마지막 FC 레이어를 수정하여 **15개의 값**을 예측하도록 변경했습니다.

### 4. 학습 (Training)
* **`train_runner.py`**
    * 실제 학습 루프(Loop)가 돌아가는 메인 코드입니다.
    * **Loss Function:** `MSELoss` (대회 평가 지표 RMSE 대응)
    * **Optimizer:** `Adam` (Learning Rate: `1e-4`)
    * **Validation:** Train/Val 비율을 8:2로 나누어 과적합(Overfitting)을 감시합니다.
    * **Save:** 검증 손실(Val Loss)이 가장 낮을 때 **`best_model.pth`**를 자동 저장합니다.

### 5. 추론 및 제출 (Inference)
* **`inference.py`**
    * 학습된 `best_model.pth`를 불러와 `test.csv` 데이터에 대한 예측을 수행합니다.
    * 테스트 데이터에는 정답(Target)이 없으므로, 더미(Dummy) 컬럼을 생성해 데이터셋 클래스와 호환되게 처리했습니다.
    * 최종 결과물인 **`my_submission.csv`**를 생성합니다.

---

## 🚀 실행 방법 (How to Run)

필요한 라이브러리 설치 후, 아래 순서대로 실행하면 됩니다.

### 1. 환경 설정
```bash
pip install torch torchvision pandas opencv-python tqdm
```

### 2. 학습 시작
모델 학습을 진행하고 `best_model.pth`를 생성합니다.
```bash
python train_runner.py
```

### 3. 결과 추론 (제출 파일 생성)
학습된 모델로 테스트 데이터를 예측합니다.
```bash
python inference.py
```
> 실행이 완료되면 폴더에 `my_submission.csv` 파일이 생성됩니다. 이 파일을 캐글에 제출하면 됩니다.

---

## 📊 학습 결과 요약
* **Input Shape:** `(Batch_Size, 7, 224, 224)`
* **Output Shape:** `(Batch_Size, 15)`
* **Best Model:** 학습 중 Validation Loss가 최소인 지점에서 자동 저장된 모델 사용.
