import torch
import albumentations as a
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCH = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "data/train/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val/"
VAL_MASK_DIR = "data/val_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = a.Compose(
        [
            a.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            a.Rotate(limit=35, p=1.0),
            a.HorizontalFlip(p=0.5),
            a.VerticalFlip(p=0.1),
            a.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )
    val_tranform = a.Compose(
        [
            a.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            a.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)  # if more then one class change outchannel to num of classes
    loss_fn = nn.BCEWithLogitsLoss()  # For more than one class use cross entropy loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_tranform,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCH):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }

        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some example to a folder
        save_predictions(val_loader, model, folder="saved_images/", device=DEVICE)


if __name__ == "__main__":
    main()
