import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time  

def generate_sine_wave_data(num_samples=1000):
    x = np.linspace(-3.14, 3.14, num_samples)
    y = np.sin(x) 
    data = np.stack((x,y), axis=1)
    return torch.tensor(data, dtype=torch.float32)

data = generate_sine_wave_data()

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

latent_dim = 10
data_dim = 2

generator = Generator(latent_dim, data_dim)
discriminator = Discriminator(data_dim)

optimizer_G = optim.Adam(generator.parameters(), lr=0.001)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)

criterion = nn.BCELoss()
num_epochs = 1000
batch_size = 64

for epoch in range(num_epochs):
    start_time = time.time()  # Start timing

    # Train Discriminator
    for _ in range(batch_size):
        real_data = data[torch.randint(0, len(data), (batch_size,))]
        real_labels = torch.ones(batch_size, 1)

        noise = torch.randn(batch_size, latent_dim)
        fake_data = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_data), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

    # Train Generator
    optimizer_G.zero_grad()
    noise = torch.randn(batch_size, latent_dim)
    fake_data = generator(noise)
    fake_labels = torch.ones(batch_size, 1)
    g_loss = criterion(discriminator(fake_data), fake_labels)
    g_loss.backward()
    optimizer_G.step()

    end_time = time.time()  # End timing
    epoch_time = end_time - start_time  # Calculate epoch duration

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Time: {epoch_time:.4f} seconds')


noise = torch.randn(100, latent_dim)
generated_data = generator(noise).detach().numpy()

plt.scatter(data[:, 0], data[:, 1], label='Real Data')
plt.scatter(generated_data[:, 0], generated_data[:, 1], label='Generated Data')
plt.legend()
plt.show()