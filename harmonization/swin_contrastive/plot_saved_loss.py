import json
import numpy as np
import matplotlib.pyplot as plt

def load_losses(file_path):
    with open(file_path, 'r') as f:
        losses = json.load(f)
    return losses

def plot_loaded_losses(name=''):
    step_interval = 1  # Adjust this value if needed

    contrast_losses = np.array(load_losses(f"{name}_contrast_losses.json")['contrast_losses'])
    classification_losses = np.array(load_losses(f"{name}_classification_losses.json")['classification_losses'])
    total_losses = np.array(load_losses(f"{name}_total_losses.json")['total_losses'])
    reconstruction_losses = np.array(load_losses(f"{name}_reconstruction_losses.json")['reconstruction_losses'])
    #self.train_losses['orthogonality_losses'].append(self.losses_dict['orthogonality_loss'])
    orthogonality_losses = np.array(load_losses(f"{name}_orthogonality_losses.json")['orthogonality_losses'])
    dice_losses = np.array(load_losses(f"{name}_dice_losses.json")['dice_losses']) 

    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    points = len(contrast_losses)
    steps = np.arange(0, points * step_interval, step_interval)
    ax[0, 0].plot(steps, contrast_losses, label='Contrastive Loss')
    ax[0, 0].set_title('Contrastive Loss')
    ax[0, 0].set_xlabel('Steps')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].legend()

    points = len(dice_losses)
    steps = np.arange(0, points * step_interval, step_interval)
    ax[0, 1].plot(steps, dice_losses, label='dice losses')
    ax[0, 1].set_title('dice_losses')
    ax[0, 1].set_xlabel('Steps')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].legend()

    points = len(orthogonality_losses)
    steps = np.arange(0, points * step_interval, step_interval)
    ax[1, 0].plot(steps, orthogonality_losses, label='Orthogonality Loss')
    ax[1, 0].set_title('Orthogonality Loss')
    ax[1, 0].set_xlabel('Steps')
    ax[1, 0].set_ylabel('Loss')
    ax[1, 0].legend()
    

    points = len(total_losses)
    steps = np.arange(0, points * step_interval, step_interval)
    ax[1, 1].plot(steps, total_losses, label='Total Loss')
    ax[1, 1].set_title('Total Loss')
    ax[1, 1].set_xlabel('Steps')
    ax[1, 1].set_ylabel('Loss')
    ax[1, 1].legend()
    
    

    plt.tight_layout()
    plt.savefig('loaded_losses_plot.png')
    plt.show()

if __name__ == '__main__':
    #parsing name
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()
    plot_loaded_losses(name=args.name)
