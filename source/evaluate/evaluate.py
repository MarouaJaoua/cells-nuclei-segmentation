import os
import torch
from torchinfo import summary
from torch.utils.data import DataLoader
import source.utils as utils
import source.arguments as arguments
from source.model import FusionNet, UNet
from source.dataset.dataset import NucleiCellDataset


def main(m_args):

    # For reproducibility
    torch.manual_seed(123)

    # Get model name
    model_name = utils.get_model_name(m_args)

    # Device
    device = torch.device("cuda:" + m_args.gpu_ids) \
        if torch.cuda.is_available() else "cpu"

    # Model
    if m_args.model == "fusion":
        model = FusionNet(m_args, 1)
    else:
        model = UNet(m_args.num_kernel, m_args.kernel_size, 1, 2)
    print(list(model.parameters())[0].shape)
    summary(model)
    model = model.to(device)

    # Optimizer
    parameters = model.parameters()
    if m_args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, m_args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, m_args.lr)

    # Load model
    if m_args.device == "cpu":
        utils.load_checkpoint(
            torch.load(os.path.join("output/", m_args.experiment_name,
                                    model_name + ".pth.tar"),
                       map_location=torch.device("cpu")), model, optimizer)
    else:
        utils.load_checkpoint(
            torch.load(os.path.join("output/", m_args.experiment_name,
                                    model_name + ".pth.tar")),
            model, optimizer)

    # Load data
    test_dataset = NucleiCellDataset(m_args.test_data,
                                     phase="test",
                                     transform=m_args.transform,
                                     image_size=m_args.image_size)
    validation_dataset = NucleiCellDataset(m_args.train_data,
                                           phase="validation",
                                           transform=m_args.transform,
                                           image_size=m_args.image_size)
    validation_dataloader = DataLoader(validation_dataset,
                                       batch_size=m_args.batch_size,
                                       shuffle=False,
                                       num_workers=m_args.num_workers,
                                       pin_memory=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=m_args.batch_size,
                                 shuffle=False,
                                 num_workers=m_args.num_workers,
                                 pin_memory=True)

    print("Total number of test examples", str(len(test_dataset)))
    print("Total number of validation examples", str(len(validation_dataset)))
    # Calculate dice and ious
    print("---- Validation metrics ----")
    dice_val = calculate_metrics(m_args, device, model, validation_dataloader)
    print("---- Test metrics ----")
    dice_test = calculate_metrics(m_args, device, model, test_dataloader)
    print("Total number of parameters")
    params = sum(dict((p.data_ptr(), p.numel())
                      for p in model.parameters()).values())
    print(params)

    with open(os.path.join("output/results.csv"), "a") as file:
        file.write("{},{},{},{},{},{},{},{},{}\n"
                   .format(model_name,
                           str(m_args.target_type),
                           str(m_args.num_kernel),
                           str(m_args.image_size),
                           str(m_args.batch_size),
                           str(m_args.lr),
                           str(dice_val),
                           str(dice_test),
                           str(params)))


def calculate_metrics(f_args, device, model, loader):
    intersections, totals = 0, 0
    model.eval()
    with torch.no_grad():
        for i_val, (x_val, y_nuclei_val, y_cell_val) in enumerate(loader):
            if f_args.target_type == "nuclei":
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
        dice = (2. * intersections) / totals
    print("dice: ", dice.item())
    return dice.item()


if __name__ == "__main__":
    args = arguments.get_arguments()

    main(args)
