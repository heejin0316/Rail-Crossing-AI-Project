# 파일명: train.py
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms

# 방금 수정한 dataset 파일 불러오기
from my_dataset import LevelCrossingDataset 

def main():
    # 1. 설정값
    CSV_PATH = './train.csv'          
    IMG_DIR = './imagesLevelCrossing' 
    BATCH_SIZE = 16
    
    # 2. CSV 읽기
    try:
        # header=0 : 첫 줄(ID, image...)을 제목으로 인식하고 데이터에서 제외함
        df = pd.read_csv(CSV_PATH, header=0) 
        print(f"CSV 파일 로드 성공! 총 데이터 수: {len(df)}개")
    except FileNotFoundError:
        print("에러: train.csv 파일을 찾을 수 없습니다.")
        return

    # 3. 데이터 나누기 (Train 80% : Val 20%)
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    print(f"학습용: {len(train_df)}개, 검증용: {len(val_df)}개")

    # 4. 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
    ])

    # 5. 데이터셋 만들기
    train_dataset = LevelCrossingDataset(train_df, IMG_DIR, transform=transform)
    val_dataset = LevelCrossingDataset(val_df, IMG_DIR, transform=transform)

    # 6. 데이터 로더 만들기
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 테스트 코드 ---
    print("\n[데이터 로더 테스트 중...]")
    try:
        images, targets = next(iter(train_loader))
        print(f"입력 데이터 모양: {images.shape}")
        print(f"정답 데이터 모양: {targets.shape}")
        
        if images.shape[1] == 7:
            print("성공! 7채널(이미지1+이미지2+마스크)이 잘 합쳐졌습니다.")
        else:
            print(f"주의: 채널 수가 {images.shape[1]}개 입니다.")
    except Exception as e:
        print("로딩 중 에러 발생:", e)

if __name__ == "__main__":
    main()
