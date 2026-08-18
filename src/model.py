import torch
import torch.nn as nn

class MiniEEGNet(nn.Module):
    def __init__(self, num_channels=8, timepoints=500):
        super(MiniEEGNet, self).__init__()
        
        # Temporal Convolution (1D)
        self.temp_conv = nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=32, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(16)
        
        # Spatial Depthwise Convolution
        self.depth_conv = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=1, groups=16)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(p=0.5)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)
        
        # Calculate size after pooling
        pooled_size = timepoints // 4
        self.fc = nn.Linear(32 * pooled_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, channels, timepoints)
        x = self.temp_conv(x)
        x = self.batchnorm1(x)
        x = torch.relu(x)
        
        x = self.depth_conv(x)
        x = self.batchnorm2(x)
        x = torch.relu(x)
        
        x = self.pool(x)
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
