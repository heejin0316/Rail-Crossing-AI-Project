import torch
import torch.nn as nn
import torchvision.models as models

class LevelCrossingModel(nn.Module):
    def __init__(self, num_classes=15):
        """
        ResNet18 모델을 가져와서 우리 대회에 맞게 수정하는 클래스
        - 입력: 7채널 (이미지1 RGB + 이미지2 RGB + 마스크 1)
        - 출력: 15개 (좌표 및 확률값)
        """
        super(LevelCrossingModel, self).__init__()
        
        # 1. ImageNet으로 미리 학습된 ResNet18 불러오기
        # (weights=models.ResNet18_Weights.DEFAULT 옵션이 최신 방식)
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 2. [입력층 수정]
        # 원래 ResNet은 3채널(RGB)을 받지만, 7채널을 넣어야 함.
        # 기존 conv1 레이어: nn.Conv2d(3, 64, kernel_size=7, ...)
        original_first_layer = self.backbone.conv1
        
        # 7채널을 받는 새로운 레이어로 교체
        self.backbone.conv1 = nn.Conv2d(
            in_channels=7,  # (3 -> 7)
            out_channels=original_first_layer.out_channels,
            kernel_size=original_first_layer.kernel_size,
            stride=original_first_layer.stride,
            padding=original_first_layer.padding,
            bias=original_first_layer.bias
        )
        
        # 주의: 새로 만든 레이어는 학습된 가중치가 없으므로 초기화가 필요하지만,
        # PyTorch가 기본적으로 랜덤 초기화를 해줌 

        # 3. [출력층 수정]
        # 원래 ResNet의 마지막 층(fc)은 1000개를 출력함.
        # 이걸 15개(좌표, 확률)만 출력하도록 교체.
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        # 모델이 데이터를 받아서 예측값을 내놓는 과정
        return self.backbone(x)

# --- [테스트 코드] 
if __name__ == "__main__":
    # 가짜 데이터 (Batch=2, Channel=7, Height=224, Width=224)
    dummy_input = torch.randn(2, 7, 224, 224)
    
    # 모델 생성
    model = LevelCrossingModel()
    
    # 예측해보기
    output = model(dummy_input)
    
    print("\n[모델 구조 테스트]")
    print(f"입력 데이터 모양: {dummy_input.shape}")
    print(f"출력 데이터 모양: {output.shape}")
    
    if output.shape == (2, 15):
        print("✅ 성공! 모델이 7채널 입력을 받아 15개 출력을 잘 뱉어냅니다.")
    else:
        print("❌ 실패! 출력 모양이 이상합니다.")
