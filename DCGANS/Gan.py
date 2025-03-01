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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()  
        self.cn1 = nn.Conv2d(in_channels= 3, out_channels= 64,stride =2,padding=1,kernel_size=(4,4))
        self.cn2 = nn.Conv2d(in_channels=64,out_channels=128,stride= 2, padding = 1,kernel_size=(4,4))
        self.cn3 = nn.Conv2d(in_channels=128,out_channels=256,stride=2 ,padding =1,kernel_size=(4,4))
        self.cn4 = nn.Conv2d(in_channels=256,out_channels=512, stride = 2 , padding =1,kernel_size=(4,4))
        self.relu = nn.LeakyReLU(0.2)
        self.flatten  = nn.Flatten()
        self.linear = nn.Linear(in_features=8192,out_features=1)
        self.sigmoid = nn.Sigmoid()


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
        X = self.sigmoid(X)
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


device = torch.device('cpu')
generator = Generator().to(device)
discriminator = Discriminator().to(device)    

loss_fn = nn.BCELoss()
lr = 0.0001
optimizer_G = optim.Adam(generator.parameters(),lr = lr)
optimizer_D =optim.Adam((discriminator.parameters()),lr =lr)
num_epochs =100
batch_size = 64
latent_dims = 100

for epochs in range(num_epochs):
    for real_images,_ in dataloader:
        batch_size = real_images.size(0)
        real_labels  = torch.ones(batch_size,1).to(device)
        fake_labels  = torch.ones(batch_size,1).to(device)
        optimizer_D.zero_grad()
        real_images = real_images.to(device)
        real_output = discriminator(real_images)
        real_loss = loss_fn(real_output,real_labels)

        noise = torch.randn(batch_size,latent_dims).to(device)
        fake_images = generator(noise)
        fake_output = discriminator(fake_images)
        fake_loss = loss_fn(fake_output,fake_labels)

        d_loss = fake_loss +real_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        noise = torch.randn(batch_size,latent_dims).to(device)
        fake_images =(generator(noise))
        fake_output = discriminator(fake_images)
        g_loss = loss_fn(fake_output,real_labels)
        g_loss.backward()
        optimizer_G.step()
    print(f"epochs = {epochs}/{num_epochs},loss_g = {g_loss.item()},loss_D = {d_loss.item()}")























# class Generator(nn.Module):
#     def __init__(self,in_channels,out_channels):
#         super().__init__()
#         def discriminator_block(in_features,out_features,batch_normalize = True):
#             block = [
#                 nn.Conv2d(in_features,out_features,3,2,1),
#                 nn.LeakyReLU(0.2,inplace=True),
#                 nn.Dropout(0.25)
#             ]
#             if not batch_normalize:
#                 block.append(nn.BatchNorm2d(out_features,0.8))

#         self.model = nn.Sequential(
#             discriminator_block(in_features=3,out_features= 16,batch_normalize=False),
#             discriminator_block(16,32),
#             discriminator_block(32,64),
#             discriminator_block(64,128),

#         )

        


# gan = Gan()
# print(gan.discriminator(X=torch.randn((1,3,64,64))))
    

        