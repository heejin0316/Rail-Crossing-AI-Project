import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class LevelCrossingDataset(Dataset):
    def __init__(self, dataframe, root_dir='./imagesLevelCrossing', transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 1. CSV 정보 가져오기
        row = self.dataframe.iloc[idx]
        
        # 2. 파일 이름 (ID, img1, img2, mask 순서)
        img1_name = str(row.iloc[1])
        img2_name = str(row.iloc[2])
        mask_name = str(row.iloc[3])
        
        # 3. 경로 결합
        img1_path = os.path.join(self.root_dir, img1_name)
        img2_path = os.path.join(self.root_dir, img2_name)
        mask_path = os.path.join(self.root_dir, mask_name)
        
        # 4. 이미지 열기
        try:
            image1 = Image.open(img1_path).convert("RGB")
            image2 = Image.open(img2_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
        except (FileNotFoundError, OSError):
            image1 = Image.new('RGB', (224, 224))
            image2 = Image.new('RGB', (224, 224))
            mask = Image.new('L', (224, 224))

        # 5. 전처리
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
            mask = self.transform(mask)
        
        # 6. 합치기
        input_tensor = torch.cat([image1, image2, mask], dim=0)

        # 7. 정답 데이터
        targets = row.iloc[4:].values.astype('float32')
        targets = torch.tensor(targets)

        return input_tensor, targets
