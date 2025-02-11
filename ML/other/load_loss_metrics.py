import os
import csv
from pprint import pprint
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import shutil


def extract_from_tensorboard(tfevents_file):
    """Extracts scalar values from a TensorBoard tfevents file and saves to CSV."""
    data = {}

    for event in tf.compat.v1.train.summary_iterator(tfevents_file):
        for value in event.summary.value:
            tag = value.tag  # Name of the scalar
            step = event.step
            scalar_value = value.simple_value if value.HasField(
                'simple_value') else None

            if scalar_value is not None:
                if tag not in data:
                    data[tag] = []
                data[tag].append((step, scalar_value))

    return data


def load_losses(experiment_name: str, arhitectura: str):
    folder_path = Path(
        "<path_to_experiments>")
    folder_path = folder_path / arhitectura / experiment_name
    folder_path = folder_path / "tensorboard"
    folder_train_loss = folder_path / "rank_0_epoch_total_loss_train"
    folder_val_loss = folder_path / "rank_0_epoch_total_loss_eval"
    file_train_loss = folder_train_loss / os.listdir(folder_train_loss)[0]
    file_val_loss = folder_val_loss / os.listdir(folder_val_loss)[0]
    data_train_loss = extract_from_tensorboard(str(file_train_loss))[
        "rank_0/epoch/total_loss"]
    data_val_loss = extract_from_tensorboard(str(file_val_loss))[
        "rank_0/epoch/total_loss"]
    if len(data_train_loss) != len(data_val_loss):
        data_val_loss = data_val_loss[:-1]
    clean_dict = {}
    clean_dict["train_loss"] = data_train_loss
    clean_dict["val_loss"] = data_val_loss
    return clean_dict


def load_metrics(experiment_name: str, arhitectura: str):
    folder_path = Path(
        "<path_to_experiments>")
    folder_path = folder_path / arhitectura / experiment_name
    folder_path = folder_path / "tensorboard"
    event_files = [x for x in os.listdir(
        folder_path) if x.startswith("events.") and x.endswith(".2")]
    event_files.sort()
    file_metric = folder_path / event_files[0]
    data_metrics = extract_from_tensorboard(str(file_metric))
    # print(data_train_loss)
    just_map = {k: v for k, v in data_metrics.items() if k in ('rank_0/metrics/Training/map',
                                                               'rank_0/metrics/Training/map_50', 'rank_0/metrics/Validation/map', 'rank_0/metrics/Validation/map_50')}
    clean_dict = {}
    clean_dict["train_map"] = just_map["rank_0/metrics/Training/map"]
    clean_dict["train_map_50"] = just_map["rank_0/metrics/Training/map_50"]
    clean_dict["val_map"] = just_map["rank_0/metrics/Validation/map"]
    clean_dict["val_map_50"] = just_map["rank_0/metrics/Validation/map_50"]
    if len(clean_dict["val_map"]) != len(clean_dict["train_map"]):
        clean_dict["train_map"] = clean_dict["train_map"][:-1]
    if len(clean_dict["val_map_50"]) != len(clean_dict["train_map_50"]):
        clean_dict["train_map_50"] = clean_dict["train_map_50"][:-1]
    return clean_dict


def plot_losses(loss_dict, directory=None):
    plt.style.use("default")
    epochs, loss_validation_values = zip(*(loss_dict["val_loss"]))

    # Extract epochs and loss values
    epochs, loss_values = zip(*(loss_dict["train_loss"]))

    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, loss_values, linestyle='-',
             label="Training Loss", color="blue")
    plt.plot(epochs, loss_validation_values,
             linestyle='-', label="Validation Loss", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    if directory is not None:
        plt.savefig(f"{directory}/loss_plot.eps", format="eps")

    plt.show()


def plot_metrics(metrics_dict, directory):
    plt.style.use("default")
    epochs, train_map = zip(*(metrics_dict["train_map"]))
    epochs, val_map = zip(*(metrics_dict["val_map"]))
    epochs, train_map_50 = zip(*(metrics_dict["train_map_50"]))
    epochs, val_map_50 = zip(*(metrics_dict["val_map_50"]))

    # Create the plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_map, linestyle='-',
             label="Training mAP", color="blue")
    plt.plot(epochs, val_map, linestyle='-',
             label="Validation mAP", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("mAP")
    plt.title("Training and Validation mAP Over Epochs")
    plt.legend()
    plt.grid(True)

    if directory is not None:
        plt.savefig(f"{directory}/map_plot.eps", format="eps")

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_map_50, linestyle='-',
             label="Training mAP@50", color="blue")
    plt.plot(epochs, val_map_50, linestyle='-',
             label="Validation mAP@50", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("mAP@50")
    plt.title("Training and Validation mAP@50 Over Epochs")
    plt.legend()
    plt.grid(True)

    if directory is not None:
        plt.savefig(f"{directory}/map_50_plot.eps", format="eps")

    plt.show()


def create_all(experiment_name, arhitectura):
    all_losses = load_losses(experiment_name, arhitectura)
    all_metrics = load_metrics(experiment_name, arhitectura)
    director_nou = f"{arhitectura}--{experiment_name}"
    if os.path.exists(director_nou):
        shutil.rmtree(director_nou)
    os.mkdir(director_nou)
    plot_metrics(all_metrics, director_nou)
    plot_losses(all_losses, director_nou)
    print("Plotted everything successfully!")
