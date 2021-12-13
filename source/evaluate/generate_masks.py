import os
import cv2
import numpy as np
import torch
import tifffile as tiff
from nd2reader import ND2Reader
import source.arguments as arguments
import source.utils as utils
from source.model import FusionNet, UNet


def main(m_args):
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

    model = model.to(device)

    # Optimizer
    parameters = model.parameters()
    if m_args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, m_args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, m_args.lr)

    # Load model
    cp_path = os.path.join("output/", m_args.experiment_name,
                           model_name + ".pth.tar")
    if m_args.device == "cpu":
        utils.load_checkpoint(torch.load(cp_path,
                                         map_location=torch.device("cpu")),
                              model,
                              optimizer)
    else:
        utils.load_checkpoint(torch.load(cp_path), model, optimizer)

    # Folders to save generated files
    if not os.path.exists("data/generate/original"):
        os.mkdir("data/generate/original")
    if not os.path.exists("data/generate/masks/cell"):
        os.mkdir("data/generate/masks/cell")
    if not os.path.exists("data/generate/masks/nuclei"):
        os.mkdir("data/generate/masks/nuclei")

    with torch.no_grad():
        model.eval()
        for file_name in os.listdir(m_args.generate_folder):
            file_name_path = os.path.join(m_args.generate_folder, file_name)
            if file_name.endswith(".tif"):
                image = tiff.imread(file_name_path)
                file_name = file_name.replace(".tif", "")
                ext = ".tif"
            elif file_name.endswith(".nd2"):
                image = ND2Reader(file_name_path)
                image = [image[i] for i in
                         range(image.metadata["total_images_per_channel"])]
                file_name = file_name.replace(".nd2", "")
                ext = ".nd2"
            elif file_name.endswith(".png") or file_name.endswith(".jpeg") \
                    or file_name.endswith(".jpg"):
                image = cv2.imread(file_name_path, 0)
                image = np.expand_dims(image, axis=0)
                file_name = file_name.replace(".png", "")
                ext = ".png"
            else:
                continue

            originals = []
            segments = []

            for i, x in enumerate(image):
                if ext in [".tif", ".nd2"]:
                    # This is because .tif and .nd2 provided images need
                    # to be converted to 8 bit
                    x = cv2.convertScaleAbs(x, alpha=0.05, beta=0)
                else:
                    x = cv2.normalize(x, None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX)
                x = cv2.resize(x, (m_args.image_size, m_args.image_size))
                originals.append(x)
                orig_p = os.path.join(m_args.generate_folder, "original",
                                      file_name + "_" + str(i) + ".png")
                cv2.imwrite(orig_p, x)
                x = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
                x = x.astype(np.float32)
                x = x.T
                x = np.expand_dims(x, axis=0)
                x = torch.Tensor(x).to(device)

                x = x / 255
                x = x.narrow(1, 0, 1)
                pred = model(x).squeeze(1)
                _, pred = torch.max(pred, dim=1)

                seg = pred.squeeze().detach().cpu().numpy()
                seg = cv2.normalize(seg, None, alpha=0, beta=255,
                                    norm_type=cv2.NORM_MINMAX)
                seg = np.expand_dims(seg, axis=0)
                seg = seg.astype(np.uint8)
                seg = seg.T
                segments.append(seg)
                mask_p = os.path.join(m_args.generate_folder, "masks",
                                      m_args.target_type,
                                      file_name + "_" + str(i) + ".png")
                cv2.imwrite(mask_p, seg)

            """orig_p = os.path.join(args.generate_folder, "original",
                                  file_name + "_org.tif")
            tiff.imsave(orig_p, originals)
            mask_p = os.path.join(args.generate_folder, "masks",
                                  args.target_type, file_name + "_mask.tif")
            tiff.imsave(mask_p, segments)"""


if __name__ == "__main__":
    args = arguments.get_arguments()
    main(args)
