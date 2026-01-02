import os
from collections import defaultdict
import pandas as pd

#이미지 폴더 경로 
img_dir = "imagesLevelCrossing"

# scene_id 별로 묶기
scenes = defaultdict(list)

for fname in os.listdir(img_dir):
    if fname.endswith((".jpg", ".png")):
        scene = fname.split('_')[0]   # "0094_1.jpg" -> "0094"
        scenes[scene].append(fname)

# scene 요약 저장
df = pd.DataFrame({
    "scene_id": list(scenes.keys()),
    "file_count": [len(files) for files in scenes.values()],
    "files": [", ".join(files) for files in scenes.values()]
})

df.to_csv("scene_summary.csv", index=False)
print(df.head())
print("scene_summary.csv 저장 완료")
