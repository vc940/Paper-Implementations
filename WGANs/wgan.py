import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision .transforms import transforms
import  torch.optim as optim 

dataset = CIFAR10(root="./data", train=True, transform=transforms.Compose([
        transforms.Resize(64),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]), download=False)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)


def critic_loss(True_,Generated):
        loss = -torch.mean(True_)+torch.mean(Generated)
        penalty = torch.abs(torch.std(True_) - torch.std(Generated))
        return loss - 0.1 *penalty



def generator_loss(fake_output):
        loss = -torch.mean(fake_output)
        return loss



class Critic(nn.Module):
        def __init__(self):
            super().__init__()  
            self.cn1 = nn.Conv2d(in_channels= 3, out_channels= 64,stride =2,padding=1,kernel_size=(4,4))
            self.cn2 = nn.Conv2d(in_channels=64,out_channels=128,stride= 2, padding = 1,kernel_size=(4,4))
            self.cn3 = nn.Conv2d(in_channels=128,out_channels=256,stride=2 ,padding =1,kernel_size=(4,4))
            self.cn4 = nn.Conv2d(in_channels=256,out_channels=512, stride = 2 , padding =1,kernel_size=(4,4))
            self.relu = nn.LeakyReLU(0.2)
            self.flatten  = nn.Flatten()
            self.linear = nn.Linear(in_features=8192,out_features=1)


        def forward(self,X):
            X = self.cn1(X)
            X = self.relu(X)
            X = self.cn2(X)
            X = self.relu(X)
            X = self.cn3(X)
            X = self.relu(X)
            X = self.cn4(X)
            X = self.relu(X)
            X = self.flatten(X)
            X = self.linear(X)
            return X
        
class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(in_features=100,out_features=4*4*1024)
            self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=5,stride=2,padding=2,output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=512,out_channels=256,kernel_size=5,stride=2,padding = 2,output_padding=1 ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=5,stride=2,padding = 2,output_padding=1 ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128,out_channels=3,kernel_size=5,stride=2,padding = 2,output_padding=1 ),
            nn.Tanh()
            )
        
        def forward(self,X):
            batch_size =X.shape[0]
            X = self.fc1(X)
            X = X.reshape(batch_size,1024,4,4)
            return self.model(X)


device = torch.device('cuda')
generator = Generator().to(device)
critic = Critic().to(device)    


lr = 5e-5
optimizer_G = optim.RMSprop(generator.parameters(),lr = lr)
optimizer_D =optim.RMSprop((critic.parameters()),lr =lr)
num_epochs =100
batch_size = 64
latent_dims = 100

for epochs in range(num_epochs):
        for real_images,_ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            for _ in range(5):
                optimizer_D.zero_grad()
                real_output = critic(real_images)
                noise = torch.randn(batch_size,latent_dims).to(device)
                fake_images = generator(noise)
                fake_output = critic(fake_images)
                critic_l = critic_loss(real_output,fake_output)
                critic_l.backward()
                optimizer_D.step()
                for p in critic.parameters():
                    p.data.clamp_(-0.1, 0.1)

            optimizer_G.zero_grad()
            noise = torch.randn(batch_size,latent_dims).to(device)
            fake_images =(generator(noise))
            fake_output = critic(fake_images)
            g_loss = generator_loss(fake_output)
            g_loss.backward()
            optimizer_G.step()
        print(f"epochs = {epochs}/{num_epochs},loss_g = {g_loss.item()},loss_D = {critic_l.item()}")


















