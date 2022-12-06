# python packages
import logging
import os
import einops
import numpy as np
# packages data steps
import torch
import albumentations as A  # image augmentation library
import torch.nn.functional as F
# packages modelling
import torch.nn as nn
from tqdm import tqdm
import torchvision.transforms as transforms
# Viz
import matplotlib.pyplot as plt

# custom packages
import src.commons.dataset as ds
import src.commons.constants as cons

# Define the logging level
logging.getLogger().setLevel(logging.INFO)


# U-Net implementation from https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SegmentationNetwork(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SegmentationNetwork, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
        

def evaluate(model, data_loader, **kwargs):
    # arguments
    loss_fn = kwargs.get("loss_fn", nn.MSELoss())
    device = kwargs.get("device", torch.device("cpu"))
    resize_dim = kwargs.get("resize_dim", 200)
    
    model.eval() # set model to evaluation mode
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    avg_loss = 0.
    for i, batch in pbar:

        resize = transforms.Resize(resize_dim)
        inputs = resize(batch) / 255.0
        inputs = inputs.to(device)
        
        model.zero_grad() # initialize gradients to zero
        with torch.no_grad():
            recons = model(inputs)
            loss = loss_fn(recons, inputs)
        avg_loss += loss.item()
        pbar.set_description(f"loss = {loss:.3f}")
    avg_loss /= len(data_loader)
    return avg_loss

def fit(model, train_loader, val_loader, optimizer, **kwargs):
    loss_fn = kwargs.get("loss_fn", nn.MSELoss())
    device = kwargs.get("device", torch.device("cpu"))
    num_epochs = kwargs.get("num_epochs", 100)
    checkpoint_path = os.path.join(kwargs.get("checkpoint_path", None), 'autoencoder_models')
    resize_dim = kwargs.get("resize_dim", 200)
    category = kwargs.get("category", '')

    train_loss_hist, val_loss_hist = [], []

    for epoch in range(num_epochs):
        # Checkpoint if validation loss improves  
        if epoch > 2 and val_loss_hist[-2] > val_loss_hist[-1] and checkpoint_path is not None:
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss_hist[-1],
                    }, os.path.join(checkpoint_path, f"autoencoder_{category}_resizedim_{resize_dim}.pt"))

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train() # set model to training mode
        train_loss = 0.
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            
            resize = transforms.Resize(resize_dim) # resize for memory saving
            inputs = resize(batch) / 255.0
            inputs = inputs.to(device)

            logging.debug(f"Inputs: shape - {inputs.shape}, min/max: {inputs.min(), inputs.max()}")
               
            model.zero_grad() # initialize gradients to zero
            recons = model(inputs) # forward pass

            logging.debug(f"Recons: shape - {recons.shape}, min/max: {recons.min(), recons.max()}")

            loss = loss_fn(recons, inputs) # loss computation
            loss.backward() # computing gradients (backward pass)
            
            optimizer.step() # updating the parameters of the model
                
            # pop computational graph
            train_loss += loss.item()
            pbar.set_description(f"loss = {loss:.3f}")
        

        train_loss /= len(train_loader)
        print(f"Train loss: {train_loss:.3f}")
        train_loss_hist.append(train_loss)
        
        val_loss = evaluate(model, val_loader, loss_fn=loss_fn, device=device, resize_dim=resize_dim)
        print(f"Validation loss: {val_loss:.3f}")
        val_loss_hist.append(val_loss)
        
    return train_loss_hist, val_loss_hist


def predict(model, data_loader, **kwargs):
    device = kwargs.get("device", torch.device("cpu"))
    resize_dim = kwargs.get("resize_dim", 200)
    
    input_imgs, recon_images, targets,  = [], [], []

    model.eval() # set model to evaluation mode
    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in pbar:

        logging.debug(f"Inputs : {batch['test'].shape}, {batch['test'].dtype}")

        resize = transforms.Resize(resize_dim)
        inputs = resize(batch["test"]) / 255.0
        inputs = inputs.to(device)
        
        logging.debug(f"Inputs : {inputs.shape}, {inputs.dtype}")


        model.zero_grad() # initialize gradients to zero
        with torch.no_grad():
            recons = model(inputs)
        
        input_imgs.append(inputs)
        recon_images.append(recons)
        targets.append(resize(batch["ground_truth"]) / 255.0)

    return torch.stack(input_imgs[:-1]), torch.stack(recon_images[:-1]), torch.stack(targets[:-1])


def train_model(data, category, **kwargs):
    num_classes = kwargs.get("num_classes", 3)
    device = kwargs.get("device", torch.device("cpu"))
    resize_dim = kwargs.get("resize_dim", 200)
    learning_rate = kwargs.get("learning_rate", 1e-5)
    checkpoint_path = kwargs.get("checkpoint_path", ".." + cons.DIR_SEP + os.path.join("results"))
    num_epochs = kwargs.get("num_epochs", 20)

    # Instantiate model and optimizer
    model = SegmentationNetwork(n_channels=3, n_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) :.2e}")
    
    # Send to device
    model = model.to(device)

    # Training
    train_loss, val_loss = fit(model, data['train'], data['val'], optimizer,
                               loss_fn=criterion, num_epochs=num_epochs, device=device,
                               checkpoint_path=checkpoint_path, resize_dim=resize_dim,
                               category=category)
    return train_loss, val_loss, model


def print_loss(train_loss, val_loss):
    # Plot loss and accuracy
    with plt.rc_context(rc={'font.size': 8}):
        plt.figure(figsize=(4,4))
        plt.title("Loss")
        plt.plot(train_loss, lw=2.0, c="b", label="Train")
        plt.plot(val_loss, lw=2.0, c="r", label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()


def load_training_and_validation_data(category, **kwargs):
    batch_size = kwargs.get("batch_size", 4)

    train_dataset = ds.MVTECTrainDataset(os.path.join(ds.current_dir(),'../', cons.DATA_PATH), category)
    print(len(train_dataset), len(train_dataset.train_data))
    
    train_data, val_data = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
    
    # Define DataLoaders for batching input
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return {'dataset': train_dataset, 'train': train_dataloader, 'val': val_dataloader}


def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def get_plot_training_data(train_dataset, n_images):
    fig, axs = plt.subplots(1, n_images, figsize=(12, 12))
    for idx in range(n_images):
        img = train_dataset[idx]
        axs[idx].imshow(torch.permute(img, (1, 2, 0)))
    fig.tight_layout()
    plt.show()


def load_test_data(category, **kwargs):
    batch_size = kwargs.get("batch_size", 4)
    # Load test dataset
    test_data = ds.MVTECTestDataset(os.path.join(ds.current_dir(),'../', cons.DATA_PATH), category)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return {'dataset': test_data, 'test': test_dataloader}


def get_plot(inputs, recons, targets, idx, threshold=0.5):
    fig, axs = plt.subplots(1, 3, figsize=(8, 8))
    scores = np.sqrt(((inputs[idx] - recons[idx])**2).sum(-1)) # Compute anomaly mask
    scores = (scores - scores.min())/(scores.max() - scores.min()) # Normalize scores
    scores = np.where(scores > threshold, 1, 0) # Threshold scores

    axs[0].imshow(rgb2gray(inputs[idx]), cmap="Greys_r")
    axs[0].set_title('Raw image')
    axs[1].imshow(scores, cmap="viridis", alpha=0.7)
    axs[1].imshow(recons[idx])
    axs[1].set_title('Generated image + score map')
    axs[2].imshow(targets[idx])
    axs[2].imshow(scores, cmap="viridis", alpha=0.7)
    axs[2].set_title('Target mask + score map')

    fig.tight_layout()
    plt.show()


def plot_in_row(inputs, recons, targets, num_images, threshold=0.5):
    fig, axs = plt.subplots(1, num_images, figsize=(8, 8))

    for idx in range(num_images):
        scores = np.sqrt(((inputs[idx] - recons[idx])**2).sum(-1)) # Compute anomaly mask
        scores = (scores - scores.min())/(scores.max() - scores.min()) # Normalize scores
        scores = np.where(scores > threshold, 1, 0) # Threshold scores
        axs[idx].imshow(rgb2gray(inputs[idx]), cmap="Greys_r")
        axs[idx].imshow(targets[idx], alpha=0.5),
        axs[idx].imshow(scores, cmap="viridis", alpha=0.5)
    fig.tight_layout()
    plt.title('Original + Target mask + score map')
    plt.show()


def run_predictions_and_plot_results(model, test_dataloader, **kwargs):
    device = kwargs.get("device", torch.device("cpu"))
    resize_dim = kwargs.get("resize_dim", 200)
    # prediction
    inputs, recons, targets = predict(model, test_dataloader, device=device, resize_dim=resize_dim)
    inputs, recons, targets = inputs.cpu().detach().numpy(), recons.cpu().detach().numpy(), targets.cpu().detach().numpy()
    # reshape
    inputs = einops.rearrange(inputs, "n b c h w -> (n b) h w c")
    recons = einops.rearrange(recons, "n b c h w -> (n b) h w c")
    targets = einops.rearrange(targets, "n b c h w -> (n b) h w c")
    
    plot_in_row(inputs, recons, targets, 4, threshold=0.5)
