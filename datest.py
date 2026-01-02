import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# 너가 만든 전처리 함수 가져오기
from preprocess import load_and_preprocess_image

class RailDataset(Dataset):
    def __init__(self, csv_path, img_dir, size=(256, 256)):
        """
        csv_path: train.csv 경로
        img_dir: imagesLevelCrossing 폴더 경로
        size: 이미지 resize 크기
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.size = size

        # 타겟 15개 컬럼 이름
        self.target_cols = [
            "probaObstacle1","x1","dx1","y1","dy1",
            "probaObstacle2","x2","dx2","y2","dy2",
            "probaObstacle3","x3","dx3","y3","dy3"
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1) 이미지 파일 경로 설정
        img1_path = os.path.join(self.img_dir, row["image-3sec"])
        img2_path = os.path.join(self.img_dir, row["image"])
        rail_path = os.path.join(self.img_dir, row["segmentedRailImage"])

        # 2) 이미지 불러오기 + 전처리
        img1 = load_and_preprocess_image(img1_path, size=self.size, is_mask=False)
        img2 = load_and_preprocess_image(img2_path, size=self.size, is_mask=False)
        rail = load_and_preprocess_image(rail_path, size=self.size, is_mask=True)

        # 3) rail은 1채널 → 채널 차원 추가
        rail = np.expand_dims(rail, axis=-1)

        # 4) target 15개 불러오기
        target = row[self.target_cols].values.astype("float32")

        # 5) numpy → torch tensor (HWC → CHW)
        img1 = torch.tensor(img1).permute(2, 0, 1)
        img2 = torch.tensor(img2).permute(2, 0, 1)
        rail = torch.tensor(rail).permute(2, 0, 1)
        target = torch.tensor(target)

        return img1, img2, rail, target

if __name__ == "__main__":
    dataset = RailDataset("train.csv", "imagesLevelCrossing")

    img1, img2, rail, target = dataset[0]

    print("img1:", img1.shape)
    print("img2:", img2.shape)
    print("rail:", rail.shape)
    print("target:", target)

    
