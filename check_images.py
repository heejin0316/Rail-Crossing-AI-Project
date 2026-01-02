import os
import pandas as pd

# 이미지 폴더 경로
img_dir = "imagesLevelCrossing"

# train.csv 불러오기
train = pd.read_csv("train.csv")

missing_files = []

# 각 row에 대해 3개 이미지 경로 검사
for idx, row in train.iterrows():
    files = [
        row["image-3sec"],          # 첫 번째 이미지
        row["image"],               # 두 번째 이미지
        row["segmentedRailImage"]   # rail mask
    ]
    
    for fname in files:
        full_path = os.path.join(img_dir, fname)
        if not os.path.exists(full_path):
            missing_files.append(full_path)

# 검사 결과 보고
if not missing_files:
    print("모든 이미지 파일이 정상적으로 존재합니다!")
else:
    print("누락된 파일이 있습니다! (아래는 일부)")
    for f in missing_files[:20]:
        print(f)
