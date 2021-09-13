import torch
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def train(model, train_loader, epochs, optimizer, loss_fn):
    """Train model.
        
    Parameters
    ----------
    model : torch.nn.Model
        model to train
    train_loader: 
        
    epochs: int
        number of epochs
    optimizer:
        
    loss_fn:
        loss function
    """
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_X)
            loss = loss_fn(output, batch_y)
            
            # Backward pass and calculate gradients
            optimizer.zero_grad()
            loss.backward()
            # Clip gradient in case it explodes
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update parameters
            optimizer.step()
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))