import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# 우리가 만든 파일들 불러오기
from my_dataset import LevelCrossingDataset
from model import LevelCrossingModel

def main():
    # --- [4단계 관련 설정] ---
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001  # 1e-4 
    EPOCHS = 10 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {DEVICE}")

    # --- 데이터 준비 ---
    print("데이터 준비 중...")
    df = pd.read_csv('./train.csv', header=0)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = LevelCrossingDataset(train_df, './imagesLevelCrossing', transform=transform)
    val_dataset = LevelCrossingDataset(val_df, './imagesLevelCrossing', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 모델 준비 ---
    model = LevelCrossingModel().to(DEVICE)

    # --- [4단계 핵심 코드] 손실 함수 및 최적화 설정 ---
    # 1. Loss: MSELoss (RMSE 평가 기준 대응)
    criterion = nn.MSELoss() 
    
    # 2. Optimizer: Adam (Learning Rate 1e-4)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 5단계: 학습 루프 (Training Loop) ---
    print("\n[학습 시작!]")
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train() # 공부 모드
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()       # 초기화
            outputs = model(inputs)     # 예측
            loss = criterion(outputs, targets) # 채점 (MSE Loss)
            loss.backward()             # 오답 분석
            optimizer.step()            # 공부(가중치 수정)

            running_loss += loss.item()
            
            if (i+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # 검증 (시험)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"=== Epoch {epoch+1} 끝 | 평균 검증 Loss: {avg_val_loss:.4f} ===")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("최고 기록 갱신! 모델 저장됨 (best_model.pth)")

    print("\n모든 학습 완료!")

if __name__ == "__main__":
    main()
