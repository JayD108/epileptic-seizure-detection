import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from .model import MiniEEGNet, get_device
from .data_generator import generate_synthetic_eeg

def train_model(epochs=10, lr=0.001, batch_size=32, num_samples_per_class=100):
    device = get_device()
    print(f"Training on device: {device}")
    
    # Generate data
    X, y = generate_synthetic_eeg(num_samples_per_class=num_samples_per_class)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MiniEEGNet(num_channels=X.shape[1], timepoints=X.shape[2]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'loss': [], 'accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            
            # Handle single batch item case
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            predictions = (outputs >= 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
            
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(accuracy)
        
        yield epoch + 1, avg_loss, accuracy, model

    return history, model
