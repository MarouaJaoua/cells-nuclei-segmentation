import os
import glob
import time
import h5py
import tqdm
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import source.arguments as arguments


def load_filenames(input_dir):
    filenames = glob.glob(input_dir)
    return [filename[:-4] for filename in filenames]


def process(rgb_files, cell_mask_files, nuclei_mask_files, output_dir,
            input_dir, image_size, file_name):
    hdf5_rgb_name = os.path.join(output_dir, file_name + "_"
                                 + str(image_size) + ".hdf5")
    hdf5_cell_name = os.path.join(output_dir, file_name + "_" + str(image_size)
                                  + "_cell_mask.hdf5")
    hdf5_nuclei_name = os.path.join(output_dir, file_name + "_"
                                    + str(image_size) + "_nuclei_mask.hdf5")
    remove_hdf5(hdf5_cell_name, hdf5_nuclei_name, hdf5_rgb_name)

    hdf5_rgb = h5py.File(hdf5_rgb_name, "a")
    hdf5_cell_mask = h5py.File(hdf5_cell_name, "a")
    hdf5_nuclei_mask = h5py.File(hdf5_nuclei_name, "a")

    print("Creating dataset for ", file_name)
    count = 0
    for file in tqdm.tqdm(rgb_files):
        rgb_file = file + ".png"
        file_name = file.split("/")[-1]
        cell_file = input_dir + "/hpa_cell_mask/" + file_name
        nuclei_file = input_dir + "/hpa_nuclei_mask/" + file_name

        if cell_file in cell_mask_files and nuclei_file in nuclei_mask_files:
            cell_file += ".npz"
            nuclei_file += ".npz"

            # Convert rgb to gray-scale and resize original rgb image
            rgb = cv2.imread(rgb_file, 0)
            data_rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            data_rgb = cv2.resize(data_rgb, (image_size, image_size))

            # Resize masks
            cell_mask = cv2.resize(np.load(cell_file)["arr_0"],
                                   (image_size, image_size))
            nuclei_mask = cv2.resize(np.load(nuclei_file)["arr_0"],
                                     (image_size, image_size))
            cell_mask = np.where(cell_mask > 0, 1, 0)
            nuclei_mask = np.where(nuclei_mask > 0, 1, 0)

            # Create hdf5
            hdf5_rgb.create_dataset(str(file_name), data=data_rgb,
                                    dtype=np.uint8, chunks=True)
            hdf5_cell_mask.create_dataset(str(file_name), data=cell_mask,
                                          dtype=np.uint8, chunks=True)
            hdf5_nuclei_mask.create_dataset(str(file_name), data=nuclei_mask,
                                            dtype=np.uint8, chunks=True)
            count += 1

    print("There are " + str(count) + " samples.")

    hdf5_rgb.close()
    hdf5_cell_mask.close()
    hdf5_nuclei_mask.close()


def remove_hdf5(hdf5_cell_name, hdf5_nuclei_name, hdf5_rgb_name):
    for current_file in [hdf5_rgb_name, hdf5_cell_name, hdf5_nuclei_name]:
        if os.path.isfile(current_file):
            print(current_file, " deleted.")
            os.remove(current_file)
            time.sleep(5)  # Wait until it is really deleted


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    cell_mask_files = load_filenames(args.input_dir
                                     + "/hpa_cell_mask/*.npz")
    nuclei_mask_files = load_filenames(args.input_dir
                                       + "/hpa_nuclei_mask/*.npz")

    rgb_files = load_filenames(args.input_dir + "/rgb/*.png")

    # Create hdf5 for train and validation data
    rgb_files_train, rgb_files_test = train_test_split(rgb_files,
                                                       train_size=0.9,
                                                       random_state=1)
    rgb_files_train, rgb_files_validation = train_test_split(rgb_files_train,
                                                             train_size=0.9,
                                                             random_state=1)
    process(rgb_files_test, cell_mask_files, nuclei_mask_files,
            args.output_dir, args.input_dir, args.image_size,
            file_name="test")
    process(rgb_files_train, cell_mask_files, nuclei_mask_files,
            args.output_dir, args.input_dir, args.image_size,
            file_name="train")
    process(rgb_files_validation, cell_mask_files, nuclei_mask_files,
            args.output_dir, args.input_dir, args.image_size,
            file_name="validation")
    print("Preprocessing done!")


if __name__ == "__main__":
    arguments = arguments.get_arguments()

    main(arguments)
