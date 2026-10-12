import os
from time import perf_counter
from pathlib import Path
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from tqdm import tqdm
from argsparser import create_predict_parser
from core.models.predrnn_v2 import RNN
from datasets.data_utils import prepare_for_predrnn,reshape_patch, reshape_patch_back
from datasets.moving_mnist import MovingMNIST
from datasets.taasrad19 import TAASRAD19
from metrics import Evaluator
from predRNNconfig import PredRNNConfig

class Predictor:
    def __init__(self, args, config):
        super().__init__()

        self.args = args
        self.config = config
        self.device = torch.device(config.device)

        self._setup_cuda_device()
        self._preparation()

    def _setup_cuda_device(self):
        """Configure the device used for inference."""
        print(f"[INFO] Device: {self.device}")

    def _preparation(self):
        """Prepare the dataset, output directories, evaluator and model."""
        self._get_dataset_params()
        self._get_data()
        self._predict_directory_setup()
        self._get_evaluator()
        self._build_model()

    def _get_dataset_params(self):
        """Load the sequence parameters from the model configuration."""
        self.input_seq_len = self.config.input_length
        self.output_seq_len = self.config.output_length
        self.total_length = self.input_seq_len + self.output_seq_len
        self.input_size = self.config.img_width
        self.patch_size = self.config.patch_size

        print(f"[INFO] Dataset used: {self.args.dataset}")
        print(f"[INFO] Input sequence length: {self.input_seq_len}")
        print(f"[INFO] Output sequence length: {self.output_seq_len}")

    def _predict_directory_setup(self):
        """Create directories for prediction outputs."""
        dataset_name = self.args.dataset
        suffix = getattr(self.args, "result_folder_suffix", "")

        folder_name = f"{dataset_name}_predrnn_v2"
        if suffix:
            folder_name += f"_{suffix}"

        self.predict_dir = (
            Path(self.args.working_dir) / "results" / folder_name
        )

        self.predict_dirs = {
            "root": self.predict_dir,
            "metrics": self.predict_dir / "metrics",
            "emissions": self.predict_dir / "emissions",
            "weights": self.predict_dir / "weights",
            "npy_files": {
                "real": self.predict_dir / "npy_files" / "real",
                "generated": self.predict_dir / "npy_files" / "generated",
            },
            "images": {
                "real": self.predict_dir / "images" / "real",
                "generated": self.predict_dir / "images" / "generated",
            },
        }

        for directory in (
            self.predict_dirs["metrics"],
            self.predict_dirs["emissions"],
        ):
            directory.mkdir(parents=True, exist_ok=True)

        if getattr(self.args, "save_sequence_npy_files", False):
            for directory in self.predict_dirs["npy_files"].values():
                directory.mkdir(parents=True, exist_ok=True)

        if getattr(self.args, "save_sequence_images", False):
            for directory in self.predict_dirs["images"].values():
                directory.mkdir(parents=True, exist_ok=True)

    def _get_data(self):
        """Load the selected dataset without shuffling."""
        dataset_loaders = {
            "taasrad19": TAASRAD19.load,
            "moving_mnist": MovingMNIST.load,
        }

        if self.args.dataset not in dataset_loaders:
            raise ValueError(
                f"Unsupported dataset: {self.args.dataset}"
            )

        loader = dataset_loaders[self.args.dataset]

        self.test_loader, self.dataset_max_value = loader(
            dataset_mode=self.args.dataset_mode,
            dataset_size=self.args.dataset_size,
            normalize_values=self.args.normalize_values,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
        )

    def _get_evaluator(self):
        """Initialize the evaluator."""

        self.evaluator = Evaluator(
            model_name="predrnn_v2",
            dataset=self.args.dataset,
            output_seq_len=self.output_seq_len,
            metric_dir=str(self.predict_dirs["metrics"]),
            pixel_thresholds=self.args.pixel_thresholds,
            pixel_balancing_weights=self.args.pixel_balancing_weights,
            dataset_max_value=self.dataset_max_value,
        )

    def _build_model(self):
        """Build the PredRNN-V2 model and load its trained weights."""
        self.gen = RNN(
            num_layers=len(self.config.num_hidden.split(",")),
            num_hidden=[
                int(value)
                for value in self.config.num_hidden.split(",")
            ],
            configs=self.config,
        ).to(self.device)

        state_dict = torch.load(
            self.args.model_weight_file_path,
            map_location=self.device,
            weights_only=True,
        )
        self.gen.load_state_dict(state_dict)
        self.gen.eval()

    def _prepare_batch(self, batch):
        """Convert a batch to the tensor layout expected by PredRNN-V2."""
        batch = batch.to(
            device=self.device,
            dtype=torch.float32,
        )

        batch = prepare_for_predrnn(batch)
        batch = reshape_patch(batch, self.patch_size)

        return batch

   
    def _create_mask(self, batch_size):
        """Create the inference mask expected by PredRNN-V2."""

        patch_width = self.input_size // self.patch_size
        patch_channels = (
            self.patch_size
            * self.patch_size
            * self.config.img_channel
        )

        if self.config.reverse_scheduled_sampling == 1:
            mask_length = self.total_length - 2
        else:
            mask_length = (
                self.total_length - self.input_seq_len - 1
            )

        mask = torch.zeros(
            (
                batch_size,
                mask_length,
                patch_width,
                patch_width,
                patch_channels,
            ),
            device=self.device,
            dtype=torch.float32,
        )
        if self.config.reverse_scheduled_sampling == 1:
            mask[:, :self.input_seq_len - 1] = 1.0

        return mask


    def _predict_batch(self, batch):
        """Generate a sequence using the PredRNN-V2 forward pass."""
        batch = self._prepare_batch(batch)
        mask = self._create_mask(batch.shape[0])

        with torch.no_grad():
            generated, _ = self.gen(batch, mask)

        # The exact temporal indexing must match RNN.forward().
        generated = generated[:, -self.output_seq_len:]

        generated = reshape_patch_back(generated, self.patch_size, self.config.img_channel)

        # Convert [B,T,H,W,C] back to [B,T,C,H,W].
        generated = generated.permute(0, 1, 4, 2, 3).contiguous()

        return generated

    def predict(self):
        """Run inference and save execution metrics."""
        self.gen.eval()

        predictions = []
        groundtruth = []

        with EmissionsTracker(
            project_name=(
                f"PredRNN-V2-{self.args.dataset}-predict"
            ),
            log_level="error",
            output_dir=str(self.predict_dirs["emissions"]),
        ) as tracker:
            start = perf_counter()

            for step, batch in enumerate(
                tqdm(self.test_loader, desc="Prediction")
            ):
                batch = batch.to(
                    device=self.device,
                    dtype=torch.float32,
                )

                input_frames = batch[:, :self.input_seq_len]
                target_frames = batch[
                    :,
                    self.input_seq_len:self.total_length,
                ]

                predicted_frames = self._predict_batch(batch)

                groundtruth.append(target_frames.cpu())
                predictions.append(predicted_frames.cpu())

                # Saving sequences can be added here after validating
                # the expected input format of the saving utilities.

            end = perf_counter()

        groundtruth = torch.cat(groundtruth, dim=0)
        predictions = torch.cat(predictions, dim=0)

        if not self.args.skip_run_evaluation:
            if self.args.normalize_values:
                groundtruth = groundtruth * self.dataset_max_value
                predictions = predictions * self.dataset_max_value

            self.evaluator.save_image_metrics(
                groundtruth,
                predictions,
            )
            self.evaluator.save_score_csv(
                groundtruth,
                predictions,
            )

            end_with_evaluation = perf_counter()
            print("[INFO] Evaluation metrics saved.")
        else:
            end_with_evaluation = end

        elapsed = end - start

        print(
            "Forecasting Time Consumption: "
            f"{elapsed // 60:.0f}min {elapsed % 60:.1f}s"
        )

        execution_metrics = pd.DataFrame([
            {
                "total_predict_time_sec": elapsed,
                "total_predict_time_with_evaluation_sec": (
                    end_with_evaluation - start
                ),
                "total_predict_time_part_hour": elapsed // 3600,
                "total_predict_time_part_min": (
                    elapsed % 3600
                ) // 60,
                "total_predict_time_part_sec": elapsed % 60,
            }
        ])

        execution_metrics.to_csv(
            self.predict_dirs["metrics"]
            / f"execution_metrics_predict_predrnn_v2_"
              f"{self.args.dataset}.csv",
            index=False,
        )


if __name__ == "__main__":
    parser = create_predict_parser()
    args = parser.parse_args()

    
    config = PredRNNConfig()
    config.dataset = args.dataset

    predictor = Predictor(args, config)

    print(">>>>>>>>>>>>>>>> Start of forecasting <<<<<<<<<<<<<<<<")
    predictor.predict()
