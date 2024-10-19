import torch
import torch.nn as nn

class MultiConv1x1(nn.Module):
    def __init__(self, in_channels_list, out_channels_list):
        """
        여러 개의 입력 및 출력 채널에 맞는 1x1 Conv 레이어들을 정의하는 클래스.
        
        Args:
            in_channels_list (list of int): 각 입력에 대한 채널 리스트.
            out_channels_list (list of int): 각 출력에 대한 채널 리스트.
        """
        super(MultiConv1x1, self).__init__()
        assert len(in_channels_list) == len(out_channels_list), "Input and output channel lists must have the same length."
        
        # 여러 개의 1x1 conv 레이어를 nn.ModuleList로 정의
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)
            for in_ch, out_ch in zip(in_channels_list, out_channels_list)
        ])
        
    def forward(self, x_list):
        """
        입력으로 들어온 여러 텐서에 대해 1x1 Conv 레이어들을 적용
        
        Args:
            x_list (list of tensors): 각 입력 텐서의 리스트.
            
        Returns:
            list of tensors: 각 입력 텐서에 대해 변환된 결과 텐서 리스트.
        """
        assert len(x_list) == len(self.convs), "Number of inputs must match the number of convolution layers."
        
        # 각 텐서에 대응되는 conv 레이어 적용
        output_list = [conv(x) for conv, x in zip(self.convs, x_list)]
        return output_list
    
    
def get_layer_output_channels(model, mapping_layers):
    channels_list = []
    for layer_name in mapping_layers:
        layer = dict(model.named_modules())[layer_name]  # 해당 레이어 가져오기
        if isinstance(layer, nn.Conv2d):  # Conv 레이어일 경우
            channels_list.append(layer.out_channels)
        elif isinstance(layer, nn.BatchNorm2d):  # BatchNorm일 경우, num_features는 채널 크기
            channels_list.append(layer.num_features)
        elif hasattr(layer, 'out_channels'):  # 혹시 다른 커스텀 레이어일 경우
            channels_list.append(layer.out_channels)
        else:
            raise ValueError(f"Layer {layer_name} does not have 'out_channels' or similar property.")
    return channels_list