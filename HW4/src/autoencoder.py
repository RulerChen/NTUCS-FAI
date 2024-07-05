import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

"""
Implementation of Autoencoder
"""


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int) -> None:
        """
        Modify the model architecture here for comparison
        """
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Linear(encoding_dim, encoding_dim//2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim//2, encoding_dim),
            nn.Linear(encoding_dim, input_dim),
        )
        self.training_curve = []

    def forward(self, x):
        # TODO: 5%

        return self.decoder(self.encoder(x))

    def fit(self, X, epochs=10, batch_size=32, learning_rate=1e-4, opti=torch.optim.AdamW):
        # TODO: 5%

        criterion = nn.MSELoss()
        optimizer = opti(self.parameters(), lr=learning_rate)
        data = DataLoader(X, batch_size=batch_size)

        self.train()

        for epoch in tqdm(range(epochs)):
            loss_list = []
            for batch in data:
                batch = batch.to(torch.float32)
                optimizer.zero_grad()
                output = self(batch)
                loss = criterion(output, batch)
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
            self.training_curve.append(sum(loss_list)/len(loss_list))

    def transform(self, X):
        # TODO: 2%

        X = torch.tensor(X, dtype=torch.float32)
        return self.encoder(X).detach().numpy()

    def reconstruct(self, X):
        # TODO: 2%

        X = torch.tensor(X, dtype=torch.float32)
        return self(X).detach().numpy()


"""
Implementation of DenoisingAutoencoder
"""


class DenoisingAutoencoder(Autoencoder):
    def __init__(self, input_dim, encoding_dim, noise_factor=0.2):
        super(DenoisingAutoencoder, self).__init__(input_dim, encoding_dim)
        self.noise_factor = noise_factor

    def add_noise(self, x):
        # TODO: 3%

        return x + torch.normal(mean=0, std=self.noise_factor, size=x.shape)

    def fit(self, X, epochs=10, batch_size=32, learning_rate=1e-4, opti=torch.optim.AdamW):
        # TODO: 4%

        data = DataLoader(X, batch_size=batch_size)
        criterion = nn.MSELoss()
        optimizer = opti(self.parameters(), lr=learning_rate)

        self.train()

        for epoch in tqdm(range(epochs)):
            loss_list = []
            for batch in data:
                batch = batch.to(torch.float32)
                batch_noisy = self.add_noise(batch)
                optimizer.zero_grad()
                output = self(batch_noisy)
                loss = criterion(output, batch)
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
            self.training_curve.append(sum(loss_list)/len(loss_list))
