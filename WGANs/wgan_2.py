import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torchvision.utils as vutils
from tqdm import tqdm
import os

os.makedirs("generated_images", exist_ok=True)

dataset = CIFAR10(root="./data", train=True, transform=transforms.Compose([
    transforms.Resize(64),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]), download=True)

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def critic_loss(real, fake, gp, lambda_gp=10):
    return -torch.mean(real) + torch.mean(fake) + lambda_gp * gp

def generator_loss(fake_output):
    return -torch.mean(fake_output)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(8192, 1)
        )
    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 4 * 4 * 1024)
        self.model = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 5, 2, 2, 1), nn.Tanh()
        )
    def forward(self, x):
        x = self.fc1(x).view(-1, 1024, 4, 4)
        return self.model(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
generator = Generator().to(device)
critic = Critic().to(device)

lr = 5e-5
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
optimizer_D = optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))

num_epochs = 100
latent_dims = 100

for epoch in range(num_epochs):
    batch_count = 0
    with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
        for real_images, _ in pbar:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            for _ in range(5):
                optimizer_D.zero_grad()
                real_output = critic(real_images)
                noise = torch.randn(batch_size, latent_dims, device=device)
                fake_images = generator(noise)
                fake_output = critic(fake_images.detach())
                gp = compute_gradient_penalty(critic, real_images, fake_images, device)
                loss_D = critic_loss(real_output, fake_output, gp)
                loss_D.backward()
                optimizer_D.step()
            
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, latent_dims, device=device)
            fake_images = generator(noise)
            fake_output = critic(fake_images)
            loss_G = generator_loss(fake_output)
            loss_G.backward()
            optimizer_G.step()
            
            batch_count += 1
            pbar.set_postfix(loss_G=loss_G.item(), loss_D=loss_D.item(), batches=batch_count)
    
    with torch.no_grad():
        fixed_noise = torch.randn(16, latent_dims, device=device)
        fake_samples = generator(fixed_noise)
        vutils.save_image(fake_samples, f"generated_images/epoch_{epoch+1}.png", normalize=True)
      
torch.save(generator.state_dict(), "generator.pth")
