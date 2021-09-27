import argparse
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data

from model import LSTMClassifier


def _get_train_data_loader(batch_size: int, training_dir: str):
    """Load data.
        
    Parameters
    ----------
    batch_size : int
        size of each batch
    training_dir: str
        directory of train data
    """
    # Read data
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    # Build dataset
    train_ds = torch.utils.data.TensorDataset(train_X, train_y)
    print("Loaded and prepared dataset of {} records and {} features.".format(train_data.shape[0], train_data.shape[1] - 1))
    
    # Organize dataset into batches
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn):
    """Train model.
        
    Parameters
    ----------
    model : torch.nn.Model
        model to train
    train_loader: torch.utils.data.DataLoader
        batch data loader
    epochs: int
        number of epochs
    optimizer:
        optimizer
    loss_fn:
        loss function
    """
    # Determine which device CPU/GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    print("Using device {}.".format(device))
    model.to(device)

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


if __name__ =='__main__':

    # ============================ #
    # ======== Parse Args ======== #
    # ============================ #

    parser = argparse.ArgumentParser()

    # Model Parameters
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--embedding_dim', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=100)
    
    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1000)
    # parser.add_argument('--learning-rate', type=float, default=0.05)

    # SageMaker Parameters
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

    # Parse args
    args = parser.parse_args()
    
    # ============================= #
    # ======== Train Model ======== #
    # ============================= #

    # Set seed
    torch.manual_seed(args.seed)
    
    # Load training data
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)  # args.train

    # Build model
    model = LSTMClassifier(args.vocab_size, args.embedding_dim, args.hidden_dim)
    print("Model loaded with vocab_size {}, embedding_dim {}, and hidden_dim {}.".format(args.vocab_size, args.embedding_dim, args.hidden_dim))

    # Define optimizer and loss and train model
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.BCELoss()

    # Train model
    train(model, train_loader, args.epochs, optimizer, loss_fn)
    
    # ============================ #
    # ======== Save Model ======== #
    # ============================ #
    
    # Save model parameters
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'vocab_size': args.vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
        }
        torch.save(model_info, f)

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
