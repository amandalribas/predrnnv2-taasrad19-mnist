import os
import random
from time import perf_counter
from argsparser import create_train_parser, validate_train_args
import pandas as pd
import torch
from tqdm import tqdm
from codecarbon import EmissionsTracker
from torchinfo import summary
from datasets.taasrad19 import TAASRAD19
from datasets.moving_mnist import MovingMNIST  
from datasets.data_utils import prepare_for_predrnn, reshape_patch
from core.models.predrnn_v2 import RNN
from predRNNconfig import PredRNNConfig


class Trainer:
    def __init__(self, config):
        super(Trainer, self).__init__()
        self.config = config
        self._setup_cuda_device()
        self._preparation()


    def _setup_cuda_device(self):
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    def _preparation(self):
        self._init_random_seed()
        self._get_dataset_params()
        self._get_data()
        self._get_epoch()
        self._build_model()
        self._select_optimizer()
        self._training_directory_setup()


    def _init_random_seed(self):
        if self.config.use_seed:
            print(f"[INFO] Using {self.config.seed} for random seed...")
            torch.manual_seed(self.config.seed)


    def _get_dataset_params(self):
        print(f"[INFO] Dataset used: {self.config.dataset}")
        self.input_seq_len = self.config.input_length
        self.output_seq_len = self.config.output_length
        self.total_length = self.config.total_length
        self.input_size = self.config.img_width


    def _get_data(self):
        dataset_loader = {
            "taasrad19": TAASRAD19.load,
            "moving_mnist": MovingMNIST.load,
        }

        loader_function = dataset_loader.get(self.config.dataset)

        if loader_function is None:
            raise ValueError(f"Unsupported dataset: {self.config.dataset}")

        loader_args = {
            "dataset_mode": "train",
            "dataset_size": self.config.dataset_size,
            "normalize_values": self.config.normalize_values,
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "shuffle": True,
        }

        if self.config.dataset == "moving_mnist":
            loader_args["split_validation"] = self.config.split_validation

        self.train_loader, self.max_value = loader_function(**loader_args)
        self.train_dataset = self.train_loader.dataset
        self.train_len = len(self.train_loader)


    def _get_epoch(self):
        self.total_epochs = self.config.num_epochs
        self.train_len = len(self.train_loader)
        self.start_epoch = 1

    def _build_model(self):
        num_hidden = [int(value) for value in self.config.num_hidden.split(",")]

        self.model = RNN(
            num_layers=len(num_hidden),
            num_hidden=num_hidden,
            configs=self.config,
        )

        self.model.to(self.config.device)
        print(f"[INFO] PredRNN-V2 device: {self.config.device}")


    def _select_optimizer(self):
        self.optimizer = torch.optim.Adam( #update weights
            self.model.parameters(),
            lr=self.config.lr,
            betas=(self.config.lr_beta1, self.config.lr_beta2),
            eps=self.config.optimizer_eps,
        )


    def _training_directory_setup(self):
        folder_name = self.config.dataset

        if self.config.result_folder_suffix:
            folder_name = (
                f"{folder_name}_{self.config.result_folder_suffix}"
            )

        self.train_dir = os.path.join(
            self.config.working_dir,
            "results",
            folder_name,
        )

        self.weights_dir = os.path.join(self.train_dir, "weights")
        self.metrics_dir = os.path.join(self.train_dir, "metrics")
        self.emissions_dir = os.path.join(self.train_dir, "emissions")

        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.emissions_dir, exist_ok=True)

        print(f"[INFO] Experiment output directory: {self.train_dir}")


    def _create_mask_true(self, batch_size):
        patch_height = self.config.img_width // self.config.patch_size
        patch_width = self.config.img_width // self.config.patch_size
        frame_channel = self.config.patch_size * self.config.patch_size * self.config.img_channel

        if self.config.reverse_scheduled_sampling:
            mask_true = torch.ones(
                (batch_size, self.config.total_length - 2, patch_height, patch_width, frame_channel),
                device=self.config.device,
            )
        else:
            mask_true = torch.ones(
                (
                    batch_size,
                    self.config.total_length - self.config.input_length - 1,
                    patch_height,
                    patch_width,
                    frame_channel,
                ),
                device=self.config.device,
            )

        return mask_true


    def _prepare_batch(self, train_imgs):
        train_imgs = train_imgs.to(self.config.device, dtype=torch.float32, non_blocking=True)
        train_imgs = prepare_for_predrnn(train_imgs)
        train_imgs = reshape_patch(train_imgs, self.config.patch_size)
        return train_imgs


    def train(self):
        self.model.train()

        min_step_loss = 1_000_000
        min_epoch_loss = getattr(self, "previous_min_epoch_loss", 1_000_000)
        min_epoch = getattr(self, "previous_min_epoch", 0)

        training_step_metrics = []
        training_epoch_metrics = []
        step_itr = 1

        with EmissionsTracker(
            project_name=f"PredRNN-V2-{self.config.dataset}-train",
            log_level="error",
            output_dir=self.emissions_dir,
        ) as tracker:

            start = perf_counter() #training time

            for epoch in range(self.start_epoch - 1, self.total_epochs):
                pbar = tqdm(
                    total=len(self.train_loader),
                    desc=f"Epoch {epoch + 1}/{self.total_epochs}",
                    postfix=dict,
                    mininterval=0.3,
                )

                epoch_loss = 0.0

                for _, train_imgs in enumerate(self.train_loader):
                    train_imgs = self._prepare_batch(train_imgs)
                    batch_size = train_imgs.shape[0]
                    mask_true = self._create_mask_true(batch_size)

                    self.optimizer.zero_grad()

                    _, loss = self.model(train_imgs, mask_true)

                    loss.backward()
                    self.optimizer.step()

                    loss_value = loss.item()
                    epoch_loss += loss_value

                    pbar.set_postfix(loss=loss_value)
                    pbar.update(1)

                    if min_step_loss > loss_value:
                        min_step_loss = loss_value

                    training_step_metrics.append({
                        "epoch": epoch + 1,
                        "step": step_itr,
                        "loss": loss_value,
                    })

                    step_itr += 1

                pbar.close()

                epoch_loss /= len(self.train_loader)

                if min_epoch_loss > epoch_loss:
                    min_epoch_loss = epoch_loss
                    min_epoch = epoch + 1

                training_epoch_metrics.append({
                    "epoch": epoch + 1,
                    "loss": epoch_loss,
                    "min_step_loss": min_step_loss,
                    "min_until_epoch_loss": min_epoch_loss,
                })

                print(f"[INFO] Epoch {epoch + 1}/{self.total_epochs} - Loss: {epoch_loss:.6f}")

                if epoch + 1 >= self.config.start_epoch_saving:
                    checkpoint_path = os.path.join(
                        self.weights_dir,
                        f"{self.config.dataset}_predrnn_v2_{epoch + 1}.pth",
                    )

                    torch.save(self.model.state_dict(), checkpoint_path)

                    print(f"[INFO] Model weights saved to {checkpoint_path}")


            end = perf_counter()

        print(
            "Training Time Consumption: {:.0f}min {:.1f}s".format(
                (end - start) // 60, (end - start) % 60,
            )
        )

        print(f"Training minimum epoch loss: {min_epoch_loss}")
        print(f"Training minimum epoch: {min_epoch}")

        emissions = tracker.final_emissions
        print(f"Training emissions: {emissions}")

        training_execution_metrics = [{
            "total_training_time_sec": (end - start),
            "total_training_time_part_hour":(end - start) // 3600,
            "total_training_time_part_min": ((end - start) % 3600) // 60,
            "total_training_time_part_sec": (end - start) % 60,
            "average_epoch_time_sec": (end - start) / self.config.num_epochs,
            "min_epoch_loss": min_epoch_loss,
            "min_epoch": min_epoch,
            "emissions": emissions,
        }]

        pd.DataFrame(training_execution_metrics).to_csv(
            os.path.join(self.metrics_dir, "execution_metrics_training.csv"),
            index=False,
        )

        pd.DataFrame(training_step_metrics).to_csv(
            os.path.join(self.metrics_dir, "loss_step.csv"),
            index=False,
        )

        pd.DataFrame(training_epoch_metrics).to_csv(
            os.path.join(self.metrics_dir, "loss_epoch.csv"),
            index=False,
        )


if __name__ == "__main__":
    parser = create_train_parser()
    args = validate_train_args(parser.parse_args())

    config = PredRNNConfig()

    for name, value in vars(args).items():
        setattr(config, name, value)

    trainer = Trainer(config)

    print(
        ">>>>>>>>>>>>>>>>>>>>>>>>>>>Start of training<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
    )
    trainer.train()