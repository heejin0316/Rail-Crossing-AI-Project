import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import os
from tqdm import tqdm # 진행상황 바 

# 우리가 만든 파일들 불러오기
from my_dataset import LevelCrossingDataset
from model import LevelCrossingModel

def main():
    # --- 1. 환경 및 경로 설정 ---
    BATCH_SIZE = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {DEVICE}")

    # [수정 포인트] 사용자 폴더 환경에 맞게 경로 설정
    TEST_CSV_PATH = './test.csv'               
    SUBMISSION_PATH = './sample_submission.csv' 
    IMAGE_ROOT = './imagesLevelCrossing'       
    OUTPUT_NAME = 'my_submission.csv'

    # --- 2. 데이터 준비 (KeyError 방지 처리) ---
    print("데이터 불러오는 중...")
    
    # (1) 파일 읽기
    test_df = pd.read_csv(TEST_CSV_PATH)
    sample_submission = pd.read_csv(SUBMISSION_PATH)
    
    # (2) ★test.csv에는 정답 컬럼이 없으므로, 
    # submission 파일의 헤더(컬럼명)를 보고 가짜 컬럼(0)을 만든다 
    target_cols = sample_submission.columns[1:] # 첫번째 ID 컬럼 제외한 나머지
    for col in target_cols:
        test_df[col] = 0.0 # 가짜 값 채우기

    # (3) 데이터셋 생성
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = LevelCrossingDataset(test_df, IMAGE_ROOT, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 3. 모델 로드 ---
    print("학습된 모델(best_model.pth) 로드 중...")
    model = LevelCrossingModel().to(DEVICE)
    
    # 저장된 가중치 불러오기
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
        print("-> 모델 로드 성공!")
    else:
        print("-> [에러] best_model.pth 파일이 없습니다. 학습이 완료되었는지 확인하세요.")
        return

    model.eval() # 평가 모드 (필수)

    # --- 4. 추론 (Inference) ---
    print("추론 시작 (정답 예측 중)...")
    all_predictions = []

    with torch.no_grad():
        for inputs, _ in tqdm(test_loader): # 정답(_)은 버림
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            
            # 결과를 CPU로 가져와서 리스트에 추가
            all_predictions.extend(outputs.cpu().numpy())

    # --- 5. 제출 파일 저장 ---
    print("제출 파일 생성 중...")
    
    # 예측값을 제출 양식에 덮어쓰기
    sample_submission.iloc[:, 1:] = all_predictions
    
    # CSV 저장
    sample_submission.to_csv(OUTPUT_NAME, index=False)
    print(f"\n완료! '{OUTPUT_NAME}' 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()
