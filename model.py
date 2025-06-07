import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from thop import profile
from KAN import KAN


class HybridCNNLSTMFFT(nn.Module):
    def __init__(self, input_size=36, cnn_channels=256, lstm_hidden_size=128, lstm_num_layers=3, output_size=7):
    # def __init__(self, input_size, cnn_channels, lstm_hidden_size, lstm_num_layers, output_size):
        super(HybridCNNLSTMFFT, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),
            nn.Conv1d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),
            nn.Conv1d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.KAN = KAN(layers_hidden=[lstm_hidden_size * 2, 64, output_size])
        self.adaptive_filter = True
        self.threshold_param = nn.Parameter(torch.rand(1))
        self.complex_weight = nn.Parameter(torch.randn(input_size, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_high = nn.Parameter(torch.randn(input_size, 2, dtype=torch.float32) * 0.02)
        self.lstm = nn.LSTM(input_size=cnn_channels * 4, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, output_size)
        self.fc_dft = nn.Linear(input_size, output_size)

    def forward(self, x):
        if x.dim() == 2:
            # If input is 2D, assume shape is [batch_size, seq_len], reshape to [batch_size, 1, seq_len]
            x = x.unsqueeze(1)
        # x = x[:, :, :, 0]
        if x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D, but got {x.dim()}D tensor instead.")
        # 分出两个分支
        # 傅里叶变换分支
        x_fft = x  # 96,1,36
        B, N, C = x_fft.shape
        dtype = x_fft.dtype
        # Apply FFT along the time dimension
        x_fft = torch.fft.rfft(x_fft, dim=1, norm='ortho')  # 96,1,9
        # print(x_fft.shape)
        weight = torch.view_as_complex(self.complex_weight)
        x_weighted = x_fft * weight
        if self.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high

            x_weighted += x_weighted2
        # Apply Inverse FFT
        x_fft = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')
        x_fft = x_fft.to(dtype)
        x_fft = x_fft.view(B, N, C)
        x_fft_weight = x_fft
        print(x_fft_weight.shape)
        # x_fft = self.fc_dft(x_fft)  # 96,1,3

        # CNN+LSTM+KAN分支
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        # print(x.shape) [96, 1, 32]

        # Change shape to [batch_size, input_size, seq_len] for CNNs
        # x = x.permute(0, 2, 1)  # (64,9,1)
        # print(x.shape)  # [64, 10, 10])

        # Change shape to [batch_size, seq_len, cnn_channels * 2] for LSTM
        # BiLSTM
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        # print(x.shape)  # [96, 1, 32]
        # print(x[:, -1, :].shape)  # [96, 64]
        # 64,16,32
        # FAN
        # x = self.fan_layer(x)
        # x = self.fc(x[:, -1, :])  # Use the last LSTM output for classification
        # x = x[:, -1, :]  # 只取LSTM输出中的最后一个时间步
        x = self.KAN(x[:, -1, :])
        torch.set_printoptions(threshold=float('inf'))
        # print(x)

        #  没有LSTM时
        # x = x.permute(0, 2, 1)
        # x = self.KAN(x)
        # print(x.shape)  # 96,3
        # x = self.fc(x[:, -1, :])
        # 特征融合：直接加和、结果融合
        # x = x_fft.squeeze() + x
        # 附加权重融合
        # x = x[:,0,:]
        # a = torch.sum(x_fft_weight * weight, -1)
        # print(a.shape)  # 96,1  x [96,1,3]
        # print(torch.sum(x_fft_weight * weight, -1).shape)  # 96,1
        x = (1 - 0.86 ** 10) * x + (0.86 ** 10) * torch.sum(
            x_fft_weight * weight, -1)
        # print(x.shape)
        # x = self.fc(x[:, -1, :])  # 通过一个全连接层
        return x

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape

        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)

        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)

        adaptive_mask = ((
                                 normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask


if __name__ == "__main__":
    model = HybridCNNLSTMFFT()
    # torchstat.stat(model, (96, 36))
    # torchsummary.summary(model, input_size=[(96, 36)], device="cpu")
    flops, params = profile(model, inputs=(torch.randn(1, 96, 36)))
    print('Flops: % .4fG' % (flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))
