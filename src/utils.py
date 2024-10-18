import torch.nn as nn

# 각 중간 feature에 대응하는 1x1 Conv 레이어를 생성합니다.
conv_layers = nn.ModuleDict()
for layer_name in mapping_layers_stu:
    # 중간 feature의 채널 수에 맞게 입력 및 출력 채널 수를 설정합니다.
    # 예를 들어, 채널 수가 256이라면:
    in_channels = acts_stu[layer_name].shape[1]  # forward pass 후에 채널 수를 알 수 있음
    conv_layers[layer_name] = nn.Conv2d(in_channels, in_channels, kernel_size=1).to(device)
