import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import itertools
import random
import numpy as np


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        
        self.down1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(256) for _ in range(9)]
        )
        
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=7, padding=3, padding_mode='reflect'),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.res_blocks(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv2(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(channels)
        )
        
    def forward(self, x):
        return x + self.block(x)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
            
        self.model = nn.Sequential(
            *discriminator_block(3, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1)
        )
        
    def forward(self, x):
        return self.model(x)


class Horse2ZebraDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        self.transform = transform
        self.files_A = sorted(os.listdir(os.path.join(root_dir, f'{mode}A')))
        self.files_B = sorted(os.listdir(os.path.join(root_dir, f'{mode}B')))
        self.root_A = os.path.join(root_dir, f'{mode}A')
        self.root_B = os.path.join(root_dir, f'{mode}B')
        
    def __getitem__(self, index):
        
        index_A = index % len(self.files_A)
        index_B = index % len(self.files_B)
        
        item_A = Image.open(os.path.join(self.root_A, self.files_A[index_A])).convert('RGB')
        item_B = Image.open(os.path.join(self.root_B, self.files_B[index_B])).convert('RGB')
        
        if self.transform:
            item_A = self.transform(item_A)
            item_B = self.transform(item_B)
            
        return {'A': item_A, 'B': item_B}
    
    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class CycleGANLoss:
    def __init__(self, device):
        self.device = device
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
    def gan_loss(self, pred, target_is_real):
        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        return self.mse_loss(pred, target)
        
    def cycle_loss(self, real, reconstructed):
        return self.l1_loss(real, reconstructed)


def train_cyclegan(data_root, epochs=200, batch_size=1, lr=0.0002, beta1=0.5, lambda_cycle=10.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    G_AB = Generator().to(device)
    G_BA = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)
    
    
    optimizer_G = optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=lr, betas=(beta1, 0.999)
    )
    optimizer_D_A = optim.Adam(D_A.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D_B = optim.Adam(D_B.parameters(), lr=lr, betas=(beta1, 0.999))
    
    
    criterion = CycleGANLoss(device)
    
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = Horse2ZebraDataset(data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    
    for epoch in range(epochs):
        for i, batch in enumerate(dataloader):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            
            
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            
            
            rec_A = G_BA(fake_B)
            rec_B = G_AB(fake_A)
            
            
            optimizer_G.zero_grad()
            
            
            loss_id_A = criterion.cycle_loss(real_A, G_BA(real_A)) * 0.5 * lambda_cycle
            loss_id_B = criterion.cycle_loss(real_B, G_AB(real_B)) * 0.5 * lambda_cycle
            
            
            loss_gan_AB = criterion.gan_loss(D_B(fake_B), True)
            loss_gan_BA = criterion.gan_loss(D_A(fake_A), True)
            
            
            loss_cycle_A = criterion.cycle_loss(real_A, rec_A) * lambda_cycle
            loss_cycle_B = criterion.cycle_loss(real_B, rec_B) * lambda_cycle
            
            
            loss_G = loss_gan_AB + loss_gan_BA + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B
            loss_G.backward()
            optimizer_G.step()
            
            
            optimizer_D_A.zero_grad()
            loss_real_A = criterion.gan_loss(D_A(real_A), True)
            loss_fake_A = criterion.gan_loss(D_A(fake_A.detach()), False)
            loss_D_A = (loss_real_A + loss_fake_A) * 0.5
            loss_D_A.backward()
            optimizer_D_A.step()
            
            
            optimizer_D_B.zero_grad()
            loss_real_B = criterion.gan_loss(D_B(real_B), True)
            loss_fake_B = criterion.gan_loss(D_B(fake_B.detach()), False)
            loss_D_B = (loss_real_B + loss_fake_B) * 0.5
            loss_D_B.backward()
            optimizer_D_B.step()
            
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss_D: {(loss_D_A + loss_D_B).item():.4f} '
                      f'Loss_G: {loss_G.item():.4f} '
                      f'Loss_cycle: {(loss_cycle_A + loss_cycle_B).item():.4f}')
        
        
        if (epoch + 1) % 50 == 0:
            torch.save(G_AB.state_dict(), f'G_AB_epoch_{epoch+1}.pth')
            torch.save(G_BA.state_dict(), f'G_BA_epoch_{epoch+1}.pth')
            torch.save(D_A.state_dict(), f'D_A_epoch_{epoch+1}.pth')
            torch.save(D_B.state_dict(), f'D_B_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    data_root = '.'  
    train_cyclegan(data_root)
