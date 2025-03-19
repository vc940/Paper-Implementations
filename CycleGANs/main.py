import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset

class ImageLoader(Dataset):
    def __init__(self):
        super().__init__()
        pass

class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, X):
        return X + self.model(X)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.Downsample = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3, bias=False),  
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Residual blocks
        self.resnet = nn.Sequential(
            ResnetBlock(256),
            ResnetBlock(256)
        )

        # Upsampling layers
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1, bias=False), 
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1, bias=False), 
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, bias=False), 
            nn.Tanh()
        )

    def forward(self, X):
        X = self.Downsample(X)
        X = self.resnet(X)
        X = self.upsample(X)
        return X



class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(256, 512,kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(512, 1,kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.model(x)
    
dataset = ImageLoader()
dataloader = DataLoader()



G = Generator()
F = Generator()
Dx = Discriminator()
Dy = Discriminator()
Epochs = 100
lr = 1e-4
optimizer_G = torch.optim.Adam(G.parameters(),lr  = lr)
optimizer_F = torch.optim.Adam(F.parameters(),lr  = lr)
optimizer_Dx = torch.optim.Adam(Dx.parameters(),lr  = lr)
optimizer_Dy = torch.optim.Adam(Dy.parameters(),lr  = lr)
adversarial_loss = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()
lambda_cycle = 100
for epoch in range(Epochs):

    for image1, image2 in dataloader:
    
        optimizer_Dx.zero_grad()
        optimizer_Dy.zero_grad()
        
        real_G = image1
        real_F = image2  
        
        fake_F = G(real_G)  
        fake_G = F(real_F) 
        
    
        labels_real_X = Dx(real_G)  
        labels_fake_X = Dx(fake_F.detach())  
        labels_real_Y = Dy(real_F) 
        labels_fake_Y = Dy(fake_G.detach())  
        
        loss_Dx = adversarial_loss(labels_real_X, torch.ones_like(labels_real_X)) +adversarial_loss(labels_fake_X, torch.zeros_like(labels_fake_X))
        
        loss_Dy = adversarial_loss(labels_real_Y, torch.ones_like(labels_real_Y)) + adversarial_loss(labels_fake_Y, torch.zeros_like(labels_fake_Y))
        
        discriminator_loss = loss_Dx + loss_Dy
        discriminator_loss.backward()
        
        optimizer_Dx.step()
        optimizer_Dy.step()

        optimizer_G.zero_grad()
        optimizer_F.zero_grad()
        
        labels_fake_X = Dx(fake_F)  
        labels_fake_Y = Dy(fake_G) 
        
        
        loss_Gx = adversarial_loss(labels_fake_X, torch.ones_like(labels_fake_X))  
        loss_Gy = adversarial_loss(labels_fake_Y, torch.ones_like(labels_fake_Y))  
        
        cycle_loss = lambda_cycle * (
            l1_loss(real_G, F(fake_F)) +  
            l1_loss(real_F, G(fake_G))   
        )
        
        generator_loss = cycle_loss + loss_Gx + loss_Gy
        generator_loss.backward()
        
        optimizer_G.step()
        optimizer_F.step()

        
        
        

