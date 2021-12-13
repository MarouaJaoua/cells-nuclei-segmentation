import os
import torch


def create_checkpoint(model_name, global_step, global_steps_list, model,
                      optimizer, train_loss_list, valid_loss_list,
                      experiment_name):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": global_step
    }
    last_state_dict = {
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "global_steps_list": global_steps_list
    }
    if not os.path.exists("output/"):
        os.mkdir("output/")
    if not os.path.exists("output/" + experiment_name):
        os.mkdir("output/" + experiment_name)

    cp_path = os.path.join("output/", experiment_name, model_name + ".pth.tar")
    save_checkpoint(checkpoint, filename=cp_path)
    metrics_path = os.path.join("output/", experiment_name, model_name + ".pt")
    save_metrics(save_path=metrics_path, metrics_dict=last_state_dict)


def save_metrics(save_path, metrics_dict):
    torch.save(metrics_dict, save_path)
    print(f"Model saved to ==> {save_path}")


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = elapsed_time / 60
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step


def get_model_name(args):
    model_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}"\
        .format(args.experiment_name,
                args.target_type,
                args.model,
                args.batch_size,
                args.optimizer,
                args.num_kernel,
                args.kernel_size,
                args.lr,
                args.image_size)
    return model_name
