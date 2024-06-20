import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_multiple_losses(train_losses, step_interval):
    """
    Plots the contrastive, classification, reconstruction, and total losses over the training steps.
    """
    
    step_interval = step_interval

    points = len(train_losses['contrast_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    contrast_losses = [loss.detach().numpy() for loss in train_losses['contrast_losses']]
    
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].plot(steps, contrast_losses, label='Contrastive Loss')
    ax[0, 0].set_title('Contrastive Loss')
    ax[0, 0].set_xlabel('Steps')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    points = len(train_losses['classification_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    classification_losses = [loss.detach().numpy() for loss in train_losses['classification_losses']]

    ax[0, 1].plot(steps, classification_losses, label='Classification Loss')
    ax[0, 1].set_title('Classification Loss')
    ax[0, 1].set_xlabel('Steps')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].legend()

    points = len(train_losses['reconstruction_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    reconstruction_losses = [loss.detach().numpy() for loss in train_losses['reconstruction_losses']]

    ax[1, 0].plot(steps, reconstruction_losses, label='Reconstruction Loss')
    ax[1, 0].set_title('Reconstruction Loss')
    ax[1, 0].set_xlabel('Steps')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend()

    points = len(train_losses['total_losses'])
    steps = np.arange(0, points * step_interval, step_interval)
    total_losses = [loss.detach().numpy() for loss in train_losses['total_losses']]

    ax[1, 1].plot(steps, total_losses, label='Total Loss')
    ax[1, 1].set_title('Total Loss')
    ax[1, 1].set_xlabel('Steps')
    ax[1, 1].set_ylabel('Loss')
    ax[1, 1].legend()

    plt.tight_layout()
    plt.savefig('losses_plot.png')
    plt.show()