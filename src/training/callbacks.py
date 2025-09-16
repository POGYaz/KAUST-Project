"""
Training callbacks for monitoring and controlling the training process.

This module provides various callback implementations including early stopping,
model checkpointing, learning rate scheduling, and training monitoring.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

from ..utils.io import write_json
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Callback:
    """
    Base callback class for training monitoring and control.
    
    Callbacks provide hooks into the training process to implement
    custom logic for monitoring, logging, early stopping, etc.
    """
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from the epoch.
            
        Returns:
            Whether to stop training (True) or continue (False).
        """
        return False
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass


class CallbackManager:
    """
    Manager for handling multiple callbacks during training.
    
    Coordinates the execution of multiple callbacks and handles
    their return values to control training flow.
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks: List of callbacks to manage.
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callback) -> None:
        """Remove a callback from the manager."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(logs)
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(logs)
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Call on_epoch_end for all callbacks.
        
        Returns:
            True if any callback requests to stop training.
        """
        stop_training = False
        for callback in self.callbacks:
            if callback.on_epoch_end(epoch, logs):
                stop_training = True
        return stop_training
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


class EarlyStopping(Callback):
    """
    Early stopping callback to prevent overfitting.
    
    Monitors a specified metric and stops training when it stops
    improving for a given number of epochs (patience).
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize early stopping callback.
        
        Args:
            monitor: Metric to monitor for early stopping.
            patience: Number of epochs with no improvement to wait.
            min_delta: Minimum change to qualify as an improvement.
            mode: 'min' for minimizing metric, 'max' for maximizing.
            restore_best_weights: Whether to restore best weights on stop.
            verbose: Whether to log early stopping messages.
        """
        super().__init__()
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        # Internal state
        self.best_score = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        
        # Comparison function based on mode
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Reset early stopping state at training start."""
        self.best_score = None
        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check for early stopping condition.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from the epoch.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if logs is None:
            return False
        
        current_score = logs.get(self.monitor)
        
        if current_score is None:
            if self.verbose:
                logger.warning(f"Early stopping metric '{self.monitor}' not found in logs")
            return False
        
        # Initialize best score on first epoch
        if self.best_score is None:
            self.best_score = current_score
            if self.restore_best_weights:
                # Store current weights as best (assuming model is accessible)
                # This would need to be set by the trainer
                pass
            return False
        
        # Check for improvement
        if self.is_better(current_score, self.best_score):
            # Improvement found
            self.best_score = current_score
            self.wait = 0
            
            if self.restore_best_weights:
                # Store current weights as best
                # This would need to be set by the trainer
                pass
            
            if self.verbose:
                logger.info(f"Improvement in {self.monitor}: {current_score:.6f}")
        
        else:
            # No improvement
            self.wait += 1
            
            if self.verbose:
                logger.info(
                    f"No improvement in {self.monitor} for {self.wait}/{self.patience} epochs"
                )
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                
                if self.verbose:
                    logger.info(
                        f"Early stopping triggered after {self.patience} epochs "
                        f"without improvement in {self.monitor}"
                    )
                
                return True
        
        return False
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log early stopping results at training end."""
        if self.stopped_epoch > 0 and self.verbose:
            logger.info(f"Training stopped early at epoch {self.stopped_epoch}")
            logger.info(f"Best {self.monitor}: {self.best_score:.6f}")


class ModelCheckpoint(Callback):
    """
    Model checkpointing callback for saving best models.
    
    Monitors a specified metric and saves the model when it improves.
    Also supports saving checkpoints at regular intervals.
    """
    
    def __init__(
        self,
        filepath: Union[str, Path],
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        save_weights_only: bool = False,
        period: int = 1,
        verbose: bool = True,
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path template for saving checkpoints.
            monitor: Metric to monitor for saving best model.
            mode: 'min' for minimizing metric, 'max' for maximizing.
            save_best_only: Whether to only save when metric improves.
            save_weights_only: Whether to save only weights or full model.
            period: Frequency (in epochs) for saving checkpoints.
            verbose: Whether to log checkpoint messages.
        """
        super().__init__()
        
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.verbose = verbose
        
        # Internal state
        self.best_score = None
        self.epochs_since_last_save = 0
        
        # Comparison function
        if mode == 'min':
            self.is_better = lambda current, best: current < best
        elif mode == 'max':
            self.is_better = lambda current, best: current > best
        else:
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Store reference to model (set by trainer)
        self.model = None
    
    def set_model(self, model: nn.Module) -> None:
        """Set the model reference for checkpointing."""
        self.model = model
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if model should be saved.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from the epoch.
            
        Returns:
            Always returns False (doesn't stop training).
        """
        if self.model is None:
            logger.warning("Model not set for checkpointing")
            return False
        
        self.epochs_since_last_save += 1
        
        should_save = False
        
        if self.save_best_only:
            # Save only if metric improved
            if logs is not None:
                current_score = logs.get(self.monitor)
                
                if current_score is not None:
                    if self.best_score is None or self.is_better(current_score, self.best_score):
                        self.best_score = current_score
                        should_save = True
                        
                        if self.verbose:
                            logger.info(
                                f"Saving model with improved {self.monitor}: {current_score:.6f}"
                            )
        else:
            # Save periodically
            if self.epochs_since_last_save >= self.period:
                should_save = True
        
        if should_save:
            self._save_model(epoch, logs)
            self.epochs_since_last_save = 0
        
        return False
    
    def _save_model(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save the model to disk."""
        try:
            # Format filepath with epoch and metrics
            filepath_str = str(self.filepath)
            
            if logs is not None:
                # Replace placeholders in filepath
                filepath_str = filepath_str.format(epoch=epoch, **logs)
            
            filepath = Path(filepath_str)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            if self.save_weights_only:
                # Save only state dict
                torch.save(self.model.state_dict(), filepath)
            else:
                # Save full model
                torch.save(self.model, filepath)
            
            if self.verbose:
                logger.info(f"Model checkpoint saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Failed to save model checkpoint: {e}")


class LearningRateLogger(Callback):
    """
    Callback for logging learning rate during training.
    
    Tracks and logs the current learning rate, which is useful
    for monitoring learning rate scheduling effects.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize learning rate logger.
        
        Args:
            verbose: Whether to log learning rate changes.
        """
        super().__init__()
        self.verbose = verbose
        self.learning_rates = []
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log current learning rate.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from the epoch.
            
        Returns:
            Always returns False (doesn't stop training).
        """
        if logs is not None:
            lr = logs.get('learning_rate')
            if lr is not None:
                self.learning_rates.append(lr)
                
                if self.verbose:
                    logger.info(f"Epoch {epoch}: Learning rate = {lr:.2e}")
        
        return False


class MetricsLogger(Callback):
    """
    Callback for logging and saving training metrics.
    
    Accumulates metrics throughout training and optionally
    saves them to disk for later analysis.
    """
    
    def __init__(
        self,
        save_path: Optional[Union[str, Path]] = None,
        save_frequency: int = 10,
    ):
        """
        Initialize metrics logger.
        
        Args:
            save_path: Optional path to save metrics JSON file.
            save_frequency: Frequency (in epochs) for saving metrics.
        """
        super().__init__()
        self.save_path = Path(save_path) if save_path is not None else None
        self.save_frequency = save_frequency
        
        # Metrics storage
        self.metrics_history = {}
        self.epochs_since_save = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """
        Log metrics for the epoch.
        
        Args:
            epoch: Current epoch number.
            logs: Dictionary of metrics from the epoch.
            
        Returns:
            Always returns False (doesn't stop training).
        """
        if logs is not None:
            # Store metrics
            for key, value in logs.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
        
        # Save periodically
        self.epochs_since_save += 1
        if (
            self.save_path is not None
            and self.epochs_since_save >= self.save_frequency
        ):
            self._save_metrics()
            self.epochs_since_save = 0
        
        return False
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save final metrics at training end."""
        if self.save_path is not None:
            self._save_metrics()
    
    def _save_metrics(self) -> None:
        """Save metrics to disk."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.metrics_history, self.save_path)
            logger.info(f"Metrics saved to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")


class TimerCallback(Callback):
    """
    Callback for timing training epochs and overall training.
    
    Tracks timing information for performance monitoring
    and optimization.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize timer callback.
        
        Args:
            verbose: Whether to log timing information.
        """
        super().__init__()
        self.verbose = verbose
        
        # Timing state
        self.train_start_time = None
        self.epoch_start_time = None
        self.epoch_times = []
    
    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Record training start time."""
        self.train_start_time = time.time()
        if self.verbose:
            logger.info("Training started")
    
    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log total training time."""
        if self.train_start_time is not None:
            total_time = time.time() - self.train_start_time
            
            if self.verbose:
                logger.info(f"Training completed in {total_time:.2f} seconds")
                
                if self.epoch_times:
                    avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
                    logger.info(f"Average epoch time: {avg_epoch_time:.2f} seconds")
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Record epoch start time."""
        self.epoch_start_time = time.time()
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> bool:
        """Log epoch time."""
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)
            
            if self.verbose:
                logger.info(f"Epoch {epoch} completed in {epoch_time:.2f} seconds")
        
        return False
