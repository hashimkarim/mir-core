"""
PyTorch Lightning modules for beat tracking and genre classification training.

Modules:
    BeatTrackingModule     — BCE-loss module for BockTCN (1-class beat output).
    BeatNetModule          — CE-loss module for BeatNet (3-class: beat / downbeat / non-beat).
    GenreClassifierModule  — CE-loss module for genre classification.
"""

from typing import Dict, Any, Optional, List, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Handle both lightning and pytorch_lightning imports
try:
    import lightning as L
except ImportError:
    import pytorch_lightning as L

from torch.utils.data import DataLoader

import mir_eval

from mir_core.beats.schema import BEAT_CHANNEL, FrameClass


class BeatTrackingModule(L.LightningModule):
    """
    PyTorch Lightning module for beat tracking.

    Wraps any beat tracking model and provides training, validation,
    and test logic with optional wandb/tensorboard logging.

    Args:
        model: Beat tracking model (BockTCN, BeatNetCRNN, etc.)
        learning_rate: Learning rate for optimizer
        loss_fn: Loss function ('bce', 'masked_bce', 'focal')
        optimizer: Optimizer type ('radam', 'adam', 'sgd')
        scheduler_patience: Patience for ReduceLROnPlateau
        include_downbeats: Whether model outputs downbeats
        include_tempo: Whether model outputs tempo
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.005,
        loss_fn: str = "bce",
        optimizer: str = "radam",
        scheduler_patience: int = 10,
        include_downbeats: bool = False,
        include_tempo: bool = False,
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn_name = loss_fn
        self.optimizer_name = optimizer
        self.scheduler_patience = scheduler_patience
        self.include_downbeats = include_downbeats
        self.include_tempo = include_tempo

        self.loss_fn = self._get_loss_fn(loss_fn)

        # Metrics storage
        self.test_fmeasures = []
        self.test_results = []

        self.save_hyperparameters(ignore=['model'])

    def _get_loss_fn(self, loss_fn: str) -> Callable:
        """Get loss function by name."""
        if loss_fn == "bce":
            return F.binary_cross_entropy
        elif loss_fn == "masked_bce":
            return self._masked_bce
        elif loss_fn == "focal":
            return self._focal_loss
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}")

    def _masked_bce(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask_value: float = -1.0
    ) -> torch.Tensor:
        """Binary cross entropy with masking for invalid targets."""
        mask = (target != mask_value).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)

        loss = F.binary_cross_entropy(pred, target * mask, reduction='none')
        return (loss * mask).sum() / mask.sum()

    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """Focal loss for handling class imbalance."""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = alpha * (1 - pt) ** gamma * bce
        return focal.mean()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x = batch["x"]
        beats_target = batch["beats"]

        output = self(x)
        beats_pred = output["beats"].squeeze(-1)

        loss = self.loss_fn(beats_pred, beats_target)

        # Log training loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Optional: downbeat and tempo losses
        if self.include_downbeats and "downbeats" in batch:
            db_pred = output["downbeats"].squeeze(-1)
            db_target = batch["downbeats"]
            db_loss = self.loss_fn(db_pred, db_target)
            loss = loss + db_loss
            self.log("train_downbeat_loss", db_loss)

        if self.include_tempo and "tempo" in batch:
            tempo_pred = output["tempo"]
            tempo_target = batch["tempo"]
            tempo_loss = self.loss_fn(tempo_pred, tempo_target)
            loss = loss + tempo_loss
            self.log("train_tempo_loss", tempo_loss)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        x = batch["x"]
        beats_target = batch["beats"]

        output = self(x)
        beats_pred = output["beats"].squeeze(-1)

        loss = self.loss_fn(beats_pred, beats_target)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

        return loss

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Test step with beat detection and evaluation."""
        import madmom

        x = batch["x"]
        sr = batch["sr"].detach().cpu().item()
        beats_target = batch["beats_ann"].squeeze().detach().cpu().numpy()

        output = self(x)
        beats_act = output["beats"].squeeze().detach().cpu().numpy()

        # Post-process with DBN
        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0,
            max_bpm=215.0,
            fps=100,
            transition_lambda=100,
            online=False
        )

        if beats_act.size > 1:
            beats_pred = beat_dbn(beats_act)
        else:
            beats_pred = np.array([])

        # Evaluate with mir_eval
        fmeasure = mir_eval.beat.f_measure(beats_target, beats_pred)
        self.test_fmeasures.append(fmeasure)

        # Store detailed results
        if "track_id" in batch:
            track_id = batch["track_id"][0] if isinstance(batch["track_id"], list) else batch["track_id"]
        else:
            track_id = f"track_{batch_idx}"

        self.test_results.append({
            "track_id": track_id,
            "fmeasure": fmeasure,
            "beats_target": beats_target,
            "beats_pred": beats_pred,
        })

        return {"fmeasure": fmeasure}

    def on_test_epoch_end(self):
        """Aggregate test metrics."""
        if self.test_fmeasures:
            avg_fmeasure = np.mean(self.test_fmeasures)
            std_fmeasure = np.std(self.test_fmeasures)

            self.log("avg_fmeasure", avg_fmeasure)
            self.log("std_fmeasure", std_fmeasure)

            print(f"\nTest Results:")
            print(f"  Average F-measure: {avg_fmeasure:.4f} +/- {std_fmeasure:.4f}")
            print(f"  Num tracks: {len(self.test_fmeasures)}")

        # Note: Don't reset here - let caller read results first
        # Caller should reset manually if needed: lit_model.test_fmeasures = []

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(),
                lr=self.learning_rate
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.2,
                patience=self.scheduler_patience,
                threshold=1e-3,
                cooldown=0,
                min_lr=1e-7,
            ),
            "monitor": "val_loss",
        }

        return [optimizer], [scheduler]


class BeatNetModule(L.LightningModule):
    """
    PyTorch Lightning module for BeatNet training.

    Uses cross-entropy loss for the official BeatNet 3-class order:
    - Class 0: Beat
    - Class 1: Downbeat
    - Class 2: Non-beat

    Args:
        model: BeatNet model (BeatNetCRNN or BeatNetBatch)
        learning_rate: Learning rate for optimizer
        class_weights: Optional weights for class imbalance (default [1.0, 1.0, 0.1])
        optimizer: Optimizer type ('adam', 'radam', 'sgd')
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.0001,
        class_weights: Optional[List[float]] = None,
        optimizer: str = "adam",
    ):
        super().__init__()

        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer

        # Default class weights for imbalanced data
        if class_weights is None:
            class_weights = [1.0, 1.0, 0.1]  # Down-weight non-beat class
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float32)
        )

        # Metrics storage
        self.test_fmeasures = []
        self.test_results = []
        self.val_step_outputs = []
        self._current_split = "test"

        self.save_hyperparameters(ignore=['model'])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step with cross-entropy loss."""
        x = batch["x"]  # (batch, time, 272)
        targets = batch["targets"]  # (batch, time) int64

        output = self(x)

        # Get logits (before softmax).
        # BeatNetBatch.forward() applies softmax internally, so we recover
        # log-probabilities here.  This is numerically acceptable for training
        # because F.cross_entropy(log_probs, targets) = F.nll_loss + const.
        # TODO: consider adding a raw-logits output path to BeatNetBatch to
        # avoid the log(softmax(...)) round-trip.
        if isinstance(output, dict) and "activations" in output:
            probs = output["activations"]  # (batch, time, 3)
            logits = torch.log(probs + 1e-8)
        else:
            # Forward returns raw output tensor, need to reshape
            # BeatNetCRNN returns (batch, 3, time), transpose to (batch, time, 3)
            logits = output.transpose(1, 2) if output.dim() == 3 else output

        # Reshape for cross-entropy: (batch * time, 3)
        logits_flat = logits.reshape(-1, 3)
        targets_flat = targets.reshape(-1)

        # Weighted cross-entropy loss
        loss = F.cross_entropy(
            logits_flat, targets_flat, weight=self.class_weights
        )

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Validation step with cross-entropy loss and optional fast F1 collection."""
        x = batch["x"]
        targets = batch["targets"]

        output = self(x)

        # Same log-probability recovery as training_step (see comment there).
        if isinstance(output, dict) and "activations" in output:
            probs = output["activations"]
            logits = torch.log(probs + 1e-8)
        else:
            # BeatNetCRNN returns (batch, 3, time), transpose to (batch, time, 3)
            logits = output.transpose(1, 2) if output.dim() == 3 else output

        logits_flat = logits.reshape(-1, 3)
        targets_flat = targets.reshape(-1)

        loss = F.cross_entropy(
            logits_flat, targets_flat, weight=self.class_weights
        )

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

        # Collect activations for per-epoch val F1
        if "beats_ann" in batch:
            with torch.no_grad():
                if isinstance(output, torch.Tensor):
                    probs = F.softmax(output, dim=1)  # (batch, 3, time)
                    beat_act = (
                        probs[:, int(FrameClass.beat), :]
                        + probs[:, int(FrameClass.downbeat), :]
                    ).squeeze().detach().cpu().numpy()
                elif (
                    isinstance(output, dict)
                    and "event_activation_data" in output
                ):
                    beat_act = (
                        output["event_activation_data"]
                        .all_beats.squeeze()
                        .detach()
                        .cpu()
                        .numpy()
                    )
                elif isinstance(output, dict) and "event_activations" in output:
                    p = output["event_activations"]
                    beat_act = p[:, :, BEAT_CHANNEL].squeeze().detach().cpu().numpy()
                elif isinstance(output, dict) and "activations" in output:
                    p = output["activations"]
                    beat_act = (
                        p[:, :, int(FrameClass.beat)]
                        + p[:, :, int(FrameClass.downbeat)]
                    ).squeeze().detach().cpu().numpy()
                else:
                    beat_act = None
                if beat_act is not None:
                    beats_ann = batch["beats_ann"].squeeze().detach().cpu().numpy()
                    self.val_step_outputs.append((beat_act, beats_ann))

        return loss

    def on_validation_epoch_end(self):
        """Compute and log fast val F1 using scipy peak picking (no DBN)."""
        if self.current_epoch == 0 or not self.val_step_outputs:
            self.val_step_outputs = []
            return
        try:
            from scipy.signal import find_peaks
            import mir_eval
            fps = 50
            fmeasures = []
            for beat_act, beats_ann in self.val_step_outputs:
                if beat_act.ndim == 0 or beat_act.size < 2:
                    continue
                peaks, _ = find_peaks(beat_act, height=0.3, distance=int(fps * 0.3))
                beats_pred = peaks / fps
                f = mir_eval.beat.f_measure(beats_ann, beats_pred)
                fmeasures.append(f)
            if fmeasures:
                val_f1 = float(np.mean(fmeasures))
                self.log("val_fmeasure", val_f1, prog_bar=True, on_epoch=True)
        except Exception:
            pass
        finally:
            self.val_step_outputs = []

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """Test step with beat detection and evaluation."""
        import madmom

        x = batch["x"]
        beats_target = batch["beats_ann"].squeeze().detach().cpu().numpy()

        output = self(x)

        # Get canonical all-beat probabilities (including downbeats).
        if isinstance(output, dict) and "event_activation_data" in output:
            beats_act = (
                output["event_activation_data"]
                .all_beats.squeeze()
                .detach()
                .cpu()
                .numpy()
            )
        elif isinstance(output, dict) and "beats" in output:
            beats_act = output["beats"].squeeze().detach().cpu().numpy()
        elif isinstance(output, dict) and "event_activations" in output:
            probs = output["event_activations"]  # (batch, time, 2)
            beats = probs[:, :, BEAT_CHANNEL]
            beats_act = beats.squeeze().detach().cpu().numpy()
        elif isinstance(output, dict) and "activations" in output:
            probs = output["activations"]  # (batch, time, 3)
            beats = (
                probs[:, :, int(FrameClass.beat)]
                + probs[:, :, int(FrameClass.downbeat)]
            )
            beats_act = beats.squeeze().detach().cpu().numpy()
        elif isinstance(output, torch.Tensor):
            # BeatNetCRNN returns (batch, 3, time) logits
            # Apply softmax and form the canonical all-beat probability.
            probs = F.softmax(output, dim=1)  # (batch, 3, time)
            beats = (
                probs[:, int(FrameClass.beat), :]
                + probs[:, int(FrameClass.downbeat), :]
            )
            beats_act = beats.squeeze().detach().cpu().numpy()
        else:
            return {}

        # Post-process with DBN (at 50 FPS for BeatNet)
        beat_dbn = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=55.0,
            max_bpm=215.0,
            fps=50,  # BeatNet uses 50 FPS
            transition_lambda=100,
            online=False
        )

        if beats_act.size > 1:
            beats_pred = beat_dbn(beats_act)
        else:
            beats_pred = np.array([])

        # Evaluate with mir_eval
        fmeasure = mir_eval.beat.f_measure(beats_target, beats_pred)
        self.test_fmeasures.append(fmeasure)

        # Store detailed results
        if "track_id" in batch:
            track_id = batch["track_id"][0] if isinstance(batch["track_id"], list) else batch["track_id"]
        else:
            track_id = f"track_{batch_idx}"

        self.test_results.append({
            "track_id": track_id,
            "fmeasure": fmeasure,
            "beats_target": beats_target,
            "beats_pred": beats_pred,
        })

        self.log("test_fmeasure", fmeasure, prog_bar=True, batch_size=1)

        return {"fmeasure": fmeasure}

    def on_test_epoch_end(self):
        """Calculate and log average metrics."""
        if self.test_fmeasures:
            avg_fmeasure = np.mean(self.test_fmeasures)
            self.log("avg_test_fmeasure", avg_fmeasure)
            print(f"\nAverage F-measure: {avg_fmeasure:.4f}")

            # Log per-track table to W&B
            try:
                import wandb
                if wandb.run is not None:
                    table = wandb.Table(columns=["track_id", "fmeasure", "split"])
                    for r in self.test_results:
                        split = getattr(self, "_current_split", "test")
                        table.add_data(r["track_id"], r["fmeasure"], split)
                    wandb.log({f"per_track/{getattr(self, '_current_split', 'test')}": table})
            except Exception:
                pass

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(), lr=self.learning_rate
            )
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        }


class GenreClassifierModule(L.LightningModule):
    """PyTorch Lightning module for genre classifier training.

    Uses cross-entropy loss with optional class weights for imbalanced datasets.
    Logs accuracy and macro-F1 per epoch; prints confusion matrix on test end.

    Args:
        model: A GenreClassifier (or any nn.Module producing logits).
        num_classes: Number of genre classes.
        learning_rate: Learning rate for optimizer.
        class_weights: Optional per-class weights for CE loss.
        optimizer: Optimizer type ('adam', 'radam', 'sgd').
        genre_labels: List of genre label strings for display.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        class_weights: Optional[List[float]] = None,
        optimizer: str = "adam",
        genre_labels: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.genre_labels = genre_labels or [str(i) for i in range(num_classes)]

        if class_weights is not None:
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.class_weights = None

        # Collect predictions for epoch-level metrics
        self._val_preds: List[int] = []
        self._val_targets: List[int] = []
        self._test_preds: List[int] = []
        self._test_targets: List[int] = []

        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = batch["mel"]
        targets = batch["label"]
        if isinstance(targets, list):
            targets = torch.tensor(targets, device=x.device)
        logits = self(x)
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        preds = logits.argmax(dim=-1)
        return loss, preds, targets

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch)
        acc = (preds == targets).float().mean()
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=len(preds))
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, batch_size=len(preds))
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch)
        acc = (preds == targets).float().mean()
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=len(preds))
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=len(preds))
        self._val_preds.extend(preds.cpu().tolist())
        self._val_targets.extend(targets.cpu().tolist())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        from sklearn.metrics import f1_score
        f1 = f1_score(self._val_targets, self._val_preds, average="macro", zero_division=0)
        self.log("val_f1_macro", f1, prog_bar=True)
        self._val_preds.clear()
        self._val_targets.clear()

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch)
        acc = (preds == targets).float().mean()
        self.log("test_loss", loss, batch_size=len(preds))
        self.log("test_acc", acc, batch_size=len(preds))
        self._test_preds.extend(preds.cpu().tolist())
        self._test_targets.extend(targets.cpu().tolist())

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return
        from sklearn.metrics import (
            accuracy_score, f1_score, classification_report, confusion_matrix,
        )
        y_true = self._test_targets
        y_pred = self._test_preds
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        self.log("test_acc_epoch", acc)
        self.log("test_f1_macro_epoch", f1)

        print(f"\n{'='*60}")
        print(f"Genre Classification Results")
        print(f"{'='*60}")
        print(f"Accuracy: {acc:.4f}  |  Macro-F1: {f1:.4f}")
        print(f"\n{classification_report(y_true, y_pred, target_names=self.genre_labels, zero_division=0)}")
        cm = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix:")
        # Header
        header = "        " + " ".join(f"{g[:6]:>6}" for g in self.genre_labels)
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join(f"{v:6d}" for v in row)
            print(f"{self.genre_labels[i][:6]:>6}  {row_str}")
        print(f"{'='*60}\n")

        self._test_preds.clear()
        self._test_targets.clear()

    def configure_optimizers(self):
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9,
            )
        elif self.optimizer_name == "radam":
            optimizer = torch.optim.RAdam(
                self.parameters(), lr=self.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.learning_rate,
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
