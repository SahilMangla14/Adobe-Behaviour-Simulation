import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.device = device

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            for inputs, targets in self.train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                print(f"Batch Loss : {loss}")
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')