import torch
import torch.nn as nn

class ConvVAE(nn.Module):

    def __init__(
        self,
        input_shape=(4, 256, 256),
        n_classes=10,
        device=torch.device('cpu')):
        
        super().__init__()
        self.device = device
        self.relu = nn.ReLU()

        # Encoder
        self.conv1 = nn.Conv2d(input_shape[0], 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.fc1 = nn.Linear(64 * 64 * 64, 1024)
        self.fc21 = nn.Linear(1024, 256)
        self.fc22 = nn.Linear(1024, 256)

        # Decoder
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 64 * 64 * 64)
        self.deconv1 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(32, n_classes, 3, 2, 1, 1)

    def encode(self, x):
        h1 = self.relu(self.conv1(x))
        h1 = self.relu(self.conv2(h1))
        h1 = h1.reshape(-1, 64 * 64 * 64)
        h1 = self.relu(self.fc1(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h1 = self.relu(self.fc3(z))
        h1 = self.relu(self.fc4(h1))
        h1 = h1.reshape(-1, 64, 64, 64)
        h1 = self.relu(self.deconv1(h1))
        return self.deconv2(h1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
