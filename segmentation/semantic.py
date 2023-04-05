import torch
import torch.nn as nn
import torch.nn.functional as F

class EncModule(nn.Module):
    def __init__(self, in_channels: int, kf=2, kernel_size=3):
        super(EncModule, self).__init__()
        out_channels = in_channels * kf
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        self.out_value = x
        return self.pool(x)

class NeckModule(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(NeckModule, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class DecModule(nn.Module):
    def __init__(self, channels: int, kernel_size=3):
        super(DecModule, self).__init__()
        padding = kernel_size // 2
        self.upconv = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(channels * 2, channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels, channels // 2, kernel_size, padding=padding)

    def forward(self, x, enc_value):
        x = self.upconv(x)
        pads = []
        for i in range(3, 1, -1):
            diff = enc_value.shape[i] - x.shape[i]
            left = diff // 2
            right = diff - left
            pads.extend([left, right])
        x = F.pad(x, pads, 'constant', 0)
        x = torch.hstack([enc_value, x])
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Unet(nn.Module):
    path = 'unet.txt'
    device = torch.device('cuda:0')

    def __init__(self, in_channels=3, out_channels=3, num_modules=4, basic_kf=64):
        super(Unet, self).__init__()
        self.num_modules = num_modules
        self.enc_modules = []
        self.dec_modules = []
        in_ch = in_channels
        kf = basic_kf
        for i in range(self.num_modules):
            self.enc_modules.append(EncModule(in_ch, kf=kf).to(self.device))
            in_ch *= kf
            self.dec_modules.append(DecModule(in_ch).to(self.device))
            kf = 2
        
        self.neck = NeckModule(in_ch).to(self.device)
        self.final_conv = nn.Conv2d(in_channels * basic_kf // 2, out_channels, 3, padding=1).to(self.device)

    def forward(self, x):
        x = x.to(torch.float).to(self.device)
        for i in range(self.num_modules):
            x = self.enc_modules[i].forward(x)
        x = self.neck.forward(x)
        for i in range(self.num_modules - 1, -1, -1):
            x = self.dec_modules[i].forward(x, self.enc_modules[i].out_value)
        x = F.relu(self.final_conv(x))
        return x

    def fit(self, loader, epoches=100, lr=0.01):
        criterion = nn.MSELoss()
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epoches):
            total_loss = 0
            for batch in loader:
                optim.zero_grad()
                y_pred = self.forward(batch['img']).to(self.device)
                y = batch['sem'].to(torch.float).to(self.device)
                loss = criterion(y_pred, y)
                loss.backward()
                optim.step()
                total_loss += loss
            print("Epoch {}: loss {}".format(epoch, total_loss))

    def evaluate(self, loader):
        mse = nn.MSELoss()
        mse_loss = 0
        for batch in loader:
            y_pred = self.forward(batch['img'])
            y = batch['sem'].to(self.device)
            mse_loss += mse(y_pred, y)
        print("MSE loss: {}".format(mse_loss))

    def store(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))