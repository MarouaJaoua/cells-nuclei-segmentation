import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_kernel",
                        type=int,
                        default=2)
    parser.add_argument("--kernel_size",
                        type=int,
                        default=3)
    parser.add_argument("--lr",
                        type=float,
                        default=0.1,
                        help="a float for the learning rate")
    parser.add_argument("--epoch",
                        type=int,
                        default=200,
                        help="an integer for the number of epochs")
    parser.add_argument("--train_data",
                        type=str,
                        default="data/preprocessed/",
                        help="path of the train data")
    parser.add_argument("--test_data",
                        type=str,
                        default="data/preprocessed/",
                        help="path of the test data")
    parser.add_argument("--save_dir",
                        type=str,
                        default="output/",
                        help="path of the trained models")
    parser.add_argument("--dataset",
                        type=str,
                        default="hpa",
                        help="a string for dataset name")
    parser.add_argument("--device",
                        type=str,
                        default="cpu",
                        help="train on cpu or gpu (cuda)")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adam",
                        help="a string for the optimizer to use")
    parser.add_argument("--model",
                        type=str,
                        default="unet",
                        help="a string for the algorithm to train")
    parser.add_argument("--batch_size",
                        type=int,
                        default="2",
                        help="an integer for the size of the batch")
    parser.add_argument("--shuffle",
                        type=bool,
                        default=False)
    parser.add_argument("--gpu_ids",
                        type=str,
                        default="1",
                        help="a string for the id of the gpu in the cluster")
    parser.add_argument("--num_workers",
                        type=int,
                        default="1")
    parser.add_argument("--experiment_name",
                        type=str,
                        default="1",
                        help="A string to give the experiment a name")
    parser.add_argument("--target_type",
                        type=str,
                        default="nuclei",
                        help="training nulcei or cells model")
    parser.add_argument("--generate_folder",
                        type=str,
                        default="data/generate/",
                        help="path of the generated data")
    parser.add_argument("--image_size",
                        type=int,
                        default=100,
                        help="an integer for the size of the image")
    parser.add_argument("--input_dir",
                        default="data/original",
                        type=str,
                        help="path of the original data")
    parser.add_argument("--output_dir",
                        default="data/preprocessed",
                        type=str,
                        help="path of the preprocessed data")

    # Augmentation
    def boolean_string(s):
        if s not in {"False", "True"}:
            raise ValueError("Not a valid boolean string")
        return s == "True"

    parser.add_argument("--transform", type=boolean_string, default="False")

    return parser.parse_args()
