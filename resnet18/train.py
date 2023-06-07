import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import random
from ResNet18.models.resnet18 import ResNet, BasicBlock
from ResNet18.infrastructure.training_utils import train, validate, resnet, save_model
from ResNet18.infrastructure.utils import save_plots, get_data
from torchvision.models import ResNet18_Weights


# Create command line interface
parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', default='scratch',
    help='choose model built from scratch or the Torchvision model',
    choices=['scratch', 'torchvision']
)

args = vars(parser.parse_args())

# Set seed for reproducibility.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
np.random.seed(seed)
random.seed(seed)

# The learning and training parameters.
epochs = 20
batch_size = 1024
learning_rate = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

train_loader, valid_loader = get_data(batch_size=batch_size)

# Define model based on the argument parser string.
if args['model'] == 'scratch':
    print('[INFO]: Training ResNet18 built from scratch...')
    
    model = resnet(
        weights=ResNet18_Weights.DEFAULT,
        img_channels=3, 
        num_layers=18,
        block=BasicBlock,
        num_classes=10, 
        device=device
        )
    
    plot_name = 'resnet_scratch'
# print(model)

# Total parameters and trainable parameters.
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")

# Optimizer.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Loss function.
criterion = nn.CrossEntropyLoss()

if __name__ == '__main__':
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        valid_epoch_loss, valid_epoch_acc = validate(
            model,
            valid_loader,
            criterion,
            device
        )

        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_loss.append(valid_epoch_loss)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss: .3f}, training acc: {train_epoch_acc: .3f}")
        print(f"Validation loss: {valid_epoch_loss: .3f}, validation acc: {valid_epoch_acc: .3f}")
        print('-'*50)

    save_model(model, "CIFAIR10_trained_ResNet.pt")

    # Save the loss and accuracy plots.
    save_plots(
        train_acc,
        valid_acc,
        train_loss,
        valid_loss,
        name=plot_name
    )



    print("TRAINING COMPLETE")



