import os
import time
import torch
import numpy as np
import torch.nn as nn
import cv2
from torchinfo import summary
from torch.utils.data import DataLoader
import source.logger as logger
from source.model import FusionNet, UNet
from source.dataset.dataset import NucleiCellDataset
import source.utils as utils
import source.arguments as arguments


def main(m_args):
    # Name model
    model_name = utils.get_model_name(m_args)

    # Tensorboard
    logger_tb = logger.Logger(log_dir=model_name)

    # Get dataset
    train_dataset = NucleiCellDataset(m_args.train_data,
                                      phase="train",
                                      transform=m_args.transform,
                                      image_size=m_args.image_size)
    validation_dataset = NucleiCellDataset(m_args.train_data,
                                           phase="validation",
                                           transform=m_args.transform,
                                           image_size=m_args.image_size)

    # Create dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=m_args.batch_size,
                                  shuffle=True,
                                  num_workers=m_args.num_workers,
                                  pin_memory=True)

    val_dataloader = DataLoader(validation_dataset,
                                batch_size=m_args.batch_size,
                                shuffle=False,
                                num_workers=m_args.num_workers,
                                pin_memory=True)

    # Device
    device = torch.device("cuda:" + m_args.gpu_ids) \
        if torch.cuda.is_available() else "cpu"

    # Model
    if m_args.model == "fusion":
        model = FusionNet(m_args, train_dataset.dim)
    else:
        model = UNet(m_args.num_kernel, m_args.kernel_size, train_dataset.dim,
                     train_dataset.target_dim)

    summary(model)
    print(list(model.parameters())[0].shape)
    print("total number of training examples", str(len(train_dataset)))
    print("total number of validation examples", str(len(validation_dataset)))
    print("length of train data loader", str(len(train_dataloader)))
    print("length of validation data loader", str(len(val_dataloader)))
    model = model.to(device)
    dataiter = iter(train_dataloader)
    imgs, _, _ = dataiter.next()
    imgs = imgs.float().to(device)
    print(imgs.shape)
    # logger_tb.update_graph(model, imgs)

    # Optimizer
    parameters = model.parameters()
    if m_args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, m_args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, m_args.lr)

    # Loss
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    count = 0
    try:
        cp_p = os.path.join("output/", m_args.experiment_name,
                            model_name + ".pth.tar")
        count = utils.load_checkpoint(torch.load(cp_p), model, optimizer)
        print("Train from a previous checkpoint...")
    except FileNotFoundError:
        print("No checkpoint found, start training from step 0...")
        pass
    # Train model
    model.train()
    best_valid_loss = float("Inf")
    total_time_min, total_time_sec = 0.0, 0.0
    global_steps_list = []
    for epoch in range(m_args.epoch):
        start_time = time.time()
        total_loss = []
        for i, (x_train, y_nuclei, y_cell) in enumerate(train_dataloader):

            optimizer.zero_grad()

            if m_args.target_type == "nuclei":
                y_train = y_nuclei
            else:
                y_train = y_cell

            # Send data and label to device
            x = x_train.to(device)
            # Input should be between 0 and 1
            x = torch.div(x, 255)

            y = y_train.to(device)

            # Predict segmentation
            pred = model(x).squeeze(1)

            # Calculate loss
            loss = criterion(pred, y.long())
            total_loss.append(loss.item())

            # Get the class with the highest probability
            _, pred = torch.max(pred, dim=1)

            # Back prop
            loss.backward()
            optimizer.step()

            # Log loss, dice and iou
            avg_loss = np.mean(total_loss)
            count += 1
            logger_tb.update_value("steps vs train loss", avg_loss, count)
            global_steps_list.append(count)

            # Display segmentation on tensorboard
            if i == 0:
                original = x_train[i].detach().cpu().numpy()
                truth = y[i].squeeze().detach().cpu().numpy()
                seg = pred[i].squeeze().detach().cpu().numpy()

                logger_tb.update_image("original", original, count)

                seg = cv2.normalize(seg, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX)
                seg = np.expand_dims(seg, axis=0)
                seg = seg.astype(np.uint8)
                logger_tb.update_image("segmentation", seg, count)

                truth = cv2.normalize(truth, None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX)
                truth = np.expand_dims(truth, axis=0)
                truth = truth.astype(np.uint8)
                logger_tb.update_image("truth", truth, count)

            if count % 5 == 0:
                avg_loss_val, dice, valid_loss = validate(m_args, criterion,
                                                          device, model,
                                                          val_dataloader)
                print("Epoch [{}/{}], Step [{}/{}] || "
                      "Train Loss: {:.4f}, Valid Loss: {:.4f}"
                      .format(epoch + 1, m_args.epoch, count,
                              m_args.epoch * len(train_dataloader),
                              avg_loss, avg_loss_val))
                logger_tb.update_value("steps vs validation loss",
                                       avg_loss_val,
                                       count)
                logger_tb.update_value("steps vs validation dice",
                                       dice,
                                       count)

                if best_valid_loss > avg_loss_val:
                    best_valid_loss = avg_loss_val
                    utils.create_checkpoint(model_name,
                                            count,
                                            global_steps_list,
                                            model,
                                            optimizer,
                                            total_loss,
                                            valid_loss,
                                            m_args.experiment_name)
                model.train()

        ep_loss_val, epoch_dice, epoch_val_loss = validate(m_args,
                                                           criterion,
                                                           device,
                                                           model,
                                                           val_dataloader)
        end_time = time.time()
        epoch_mins, epoch_secs = utils.epoch_time(start_time, end_time)
        total_time_min += epoch_mins
        total_time_sec += epoch_secs

        logger_tb.update_value("epoch vs validation loss", ep_loss_val, epoch)
        logger_tb.update_value("epoch vs validation dice", epoch_dice, epoch)
        logger_tb.update_value("epoch vs time", total_time_min, epoch)
        logger_tb.update_value("steps vs time", total_time_min, count)


def validate(v_args, criterion, device, model, validation_dataloader):
    model.eval()
    valid_loss = []
    intersections, totals = 0, 0
    with torch.no_grad():
        for i_val, (x_val, y_nuclei_val, y_cell_val) in enumerate(
                validation_dataloader):
            if v_args.target_type == "nuclei":
                y_train = y_nuclei_val
            else:
                y_train = y_cell_val

            # Send data and label to device
            x = x_val.to(device)
            # Input should be between 0 and 1
            x = torch.div(x, 255)
            y = y_train.to(device)

            # Predict segmentation
            pred = model(x).squeeze(1)

            # Calculate loss
            loss = criterion(pred, y.long())

            # Get the class with the highest probability
            _, pred = torch.max(pred, dim=1)

            inputs = pred.view(-1)
            targets = y.view(-1)
            intersection = (inputs * targets).sum()
            total = inputs.sum() + targets.sum()

            # intersection is equivalent to True Positive count
            intersections += intersection
            # union is the mutually inclusive area of all labels & predictions
            totals += total
            valid_loss.append(loss.item())
    dice = (2. * intersections) / totals
    avg_loss_val = np.mean(valid_loss)
    return avg_loss_val, dice, valid_loss


if __name__ == "__main__":
    args = arguments.get_arguments()
    main(args)
