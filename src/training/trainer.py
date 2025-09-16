"""
Training utilities and trainer classes for recommendation models.

This module provides comprehensive training functionality including
optimizers, schedulers, early stopping, and model checkpointing
for both retrieval and ranking models.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..models.retriever.losses import create_loss_function
from ..utils.io import write_json
from ..utils.logging import get_logger
from .callbacks import CallbackManager, EarlyStopping, ModelCheckpoint

logger = get_logger(__name__)


class BaseTrainer:
    """
    Base trainer class with common training functionality.
    
    Provides core training loop, validation, logging, and callback
    management functionality that can be extended for specific models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        device: Union[str, torch.device] = 'cpu',
        scheduler: Optional[_LRScheduler] = None,
        gradient_clip_value: Optional[float] = None,
        accumulation_steps: int = 1,
        mixed_precision: bool = False,
    ):
        """
        Initialize the base trainer.
        
        Args:
            model: Model to train.
            optimizer: Optimizer for training.
            loss_function: Loss function to use.
            device: Device for training.
            scheduler: Optional learning rate scheduler.
            gradient_clip_value: Optional gradient clipping value.
            accumulation_steps: Number of steps for gradient accumulation.
            mixed_precision: Whether to use mixed precision training.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.device = torch.device(device)
        self.scheduler = scheduler
        self.gradient_clip_value = gradient_clip_value
        self.accumulation_steps = accumulation_steps
        self.mixed_precision = mixed_precision
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize mixed precision scaler (CUDA only)
        use_amp = bool(mixed_precision and self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        
        # Initialize callback manager
        self.callback_manager = CallbackManager()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }
    
    def add_callback(self, callback) -> None:
        """Add a callback to the trainer."""
        self.callback_manager.add_callback(callback)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader.
            validation_loader: Optional validation data loader.
            
        Returns:
            Dictionary with training metrics.
        """
        self.model.train()
        
        # Initialize metrics
        epoch_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            loss = self._forward_pass(batch)
            
            # Backward pass
            self._backward_pass(loss)
            
            # Update metrics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 100 == 0:
                logger.debug(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        # Compute average loss
        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # Validation
        avg_val_loss = None
        if validation_loader is not None:
            avg_val_loss = self.validate(validation_loader)
        
        # Update learning rate
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric = avg_val_loss if avg_val_loss is not None else avg_train_loss
                self.scheduler.step(metric)
            else:
                self.scheduler.step()
        
        # Compute epoch time
        epoch_time = time.time() - start_time
        
        # Prepare metrics
        metrics = {
            'train_loss': avg_train_loss,
            'epoch_time': epoch_time,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        if avg_val_loss is not None:
            metrics['val_loss'] = avg_val_loss
        
        return metrics
    
    def validate(self, validation_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            validation_loader: Validation data loader.
            
        Returns:
            Average validation loss.
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in validation_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                loss = self._forward_pass(batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        save_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: Training data loader.
            validation_loader: Optional validation data loader.
            epochs: Number of epochs to train.
            save_dir: Optional directory to save checkpoints.
            
        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        # Initialize save directory
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Callback: on_epoch_begin
            self.callback_manager.on_epoch_begin(epoch, logs={})
            
            # Train for one epoch
            metrics = self.train_epoch(train_loader, validation_loader)
            
            # Update training history
            for key, value in metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
            
            # Log epoch results
            log_message = f"Epoch {epoch + 1}/{epochs}"
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_message += f" | {key}: {value:.4f}"
            logger.info(log_message)
            
            # Callback: on_epoch_end
            stop_training = self.callback_manager.on_epoch_end(epoch, logs=metrics)
            
            if stop_training:
                logger.info("Early stopping triggered")
                break
        
        # Save final training history
        if save_dir is not None:
            history_path = save_dir / 'training_history.json'
            write_json(self.training_history, history_path)
            logger.info(f"Training history saved to {history_path}")
        
        logger.info("Training completed")
        return self.training_history
    
    def _move_batch_to_device(self, batch: Any) -> Any:
        """Move batch data to the training device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {key: self._move_batch_to_device(value) for key, value in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [self._move_batch_to_device(item) for item in batch]
        else:
            return batch
    
    def _forward_pass(self, batch: Any) -> torch.Tensor:
        """
        Perform forward pass and compute loss.
        
        This method should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement _forward_pass")
    
    def _backward_pass(self, loss: torch.Tensor) -> None:
        """Perform backward pass with gradient accumulation and clipping."""
        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps
        if (self.global_step + 1) % self.accumulation_steps == 0:
            # Gradient clipping
            if self.gradient_clip_value is not None:
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_value
                )
            
            # Optimizer step
            if self.mixed_precision and self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Zero gradients
            self.optimizer.zero_grad()
        
        self.global_step += 1


class RetrievalTrainer(BaseTrainer):
    """
    Trainer specifically designed for retrieval models.
    
    Handles Two-Tower models with InfoNCE loss and other
    contrastive learning objectives.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: nn.Module,
        temperature: float = 0.07,
        **kwargs
    ):
        """
        Initialize retrieval trainer.
        
        Args:
            model: Two-Tower or similar retrieval model.
            optimizer: Optimizer for training.
            loss_function: Loss function (e.g., InfoNCE).
            temperature: Temperature parameter for similarity scaling.
            **kwargs: Additional arguments for BaseTrainer.
        """
        super().__init__(model, optimizer, loss_function, **kwargs)
        self.temperature = temperature
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for retrieval model.
        
        Args:
            batch: Batch dictionary with users, positives, negatives.
            
        Returns:
            Computed loss.
        """
        users = batch['users']
        positives = batch['positives']
        negatives = batch['negatives']
        
        # Forward pass through model
        use_amp = bool(self.mixed_precision and self.device.type == 'cuda')
        if use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                # Get embeddings
                user_embeddings = self.model.user_tower(users)
                positive_embeddings = self.model.item_tower(positives)
                
                # Reshape negatives and get embeddings
                batch_size, n_negatives = negatives.shape
                negative_embeddings = self.model.item_tower(negatives.view(-1))
                negative_embeddings = negative_embeddings.view(batch_size, n_negatives, -1)
                
                # Compute loss
                loss = self.loss_function(
                    user_embeddings, positive_embeddings, negative_embeddings
                )
        else:
            # Get embeddings
            user_embeddings = self.model.user_tower(users)
            positive_embeddings = self.model.item_tower(positives)
            
            # Reshape negatives and get embeddings
            batch_size, n_negatives = negatives.shape
            negative_embeddings = self.model.item_tower(negatives.view(-1))
            negative_embeddings = negative_embeddings.view(batch_size, n_negatives, -1)
            
            # Compute loss
            loss = self.loss_function(
                user_embeddings, positive_embeddings, negative_embeddings
            )
        
        return loss

    def train_epoch(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch and compute validation Recall@10 if validation data is provided.
        """
        # Run base training (includes val_loss computation)
        metrics = super().train_epoch(train_loader, validation_loader)

        # Compute validation Recall@10 for monitoring and early stopping
        if validation_loader is not None:
            self.model.eval()
            hits = 0.0
            total = 0
            K = 10
            device = self.device
            with torch.no_grad():
                # Precompute all item embeddings once
                all_item_vecs = self.model.get_all_item_embeddings(device)
                all_item_vecs = all_item_vecs.to(device)
                for batch in validation_loader:
                    batch = self._move_batch_to_device(batch)
                    users = batch['users']
                    positives = batch['positives']
                    u_vecs = self.model.user_tower(users)
                    scores = u_vecs @ all_item_vecs.T
                    topk = scores.topk(k=K, dim=1).indices
                    # Check hits
                    hit = (topk == positives.unsqueeze(1)).any(dim=1).float().sum().item()
                    hits += hit
                    total += positives.size(0)
            val_recall = float(hits / max(total, 1))
            metrics['val_recall@10'] = val_recall
            # Explicitly print Recall@10 so it is visible in logs
            logger.info(f"Epoch {self.current_epoch + 1}: val_recall@10={val_recall:.4f}")
        return metrics


class RankingTrainer(BaseTrainer):
    """
    Trainer specifically designed for ranking models.
    
    Handles MLP rankers with binary classification or
    ranking losses.
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_function: Optional[nn.Module] = None,
        **kwargs
    ):
        """
        Initialize ranking trainer.
        
        Args:
            model: MLP ranker or similar ranking model.
            optimizer: Optimizer for training.
            loss_function: Loss function (defaults to BCEWithLogitsLoss).
            **kwargs: Additional arguments for BaseTrainer.
        """
        if loss_function is None:
            loss_function = nn.BCEWithLogitsLoss()
        
        super().__init__(model, optimizer, loss_function, **kwargs)
    
    def _forward_pass(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for ranking model.
        
        Args:
            batch: Batch with features and labels (dict or tuple).
            
        Returns:
            Computed loss.
        """
        if isinstance(batch, dict):
            features = batch['features']
            labels = batch['labels']
        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
            features, labels = batch[0], batch[1]
        else:
            raise TypeError("Unsupported batch type for RankingTrainer")
        
        # Forward pass through model
        use_amp = bool(self.mixed_precision and self.device.type == 'cuda')
        if use_amp:
            with torch.amp.autocast('cuda', enabled=True):
                logits = self.model(features)
                loss = self.loss_function(logits, labels)
        else:
            logits = self.model(features)
            loss = self.loss_function(logits, labels)
        
        return loss


def create_trainer(
    trainer_type: str,
    model: nn.Module,
    optimizer: Optimizer,
    loss_function: Optional[nn.Module] = None,
    **kwargs
) -> BaseTrainer:
    """
    Factory function for creating trainers.
    
    Args:
        trainer_type: Type of trainer to create ('retrieval' or 'ranking').
        model: Model to train.
        optimizer: Optimizer for training.
        loss_function: Optional loss function.
        **kwargs: Additional arguments for the trainer.
        
    Returns:
        Initialized trainer.
        
    Raises:
        ValueError: If trainer_type is not recognized.
    """
    trainer_registry = {
        'retrieval': RetrievalTrainer,
        'ranking': RankingTrainer,
        'base': BaseTrainer,
    }
    
    if trainer_type.lower() not in trainer_registry:
        available_trainers = list(trainer_registry.keys())
        raise ValueError(
            f"Unknown trainer type: {trainer_type}. "
            f"Available trainers: {available_trainers}"
        )
    
    trainer_class = trainer_registry[trainer_type.lower()]
    return trainer_class(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        **kwargs
    )


def setup_training(
    model: nn.Module,
    config: Dict[str, Any],
    device: Union[str, torch.device] = 'cpu',
) -> Tuple[Optimizer, _LRScheduler, nn.Module]:
    """
    Set up optimizer, scheduler, and loss function from configuration.
    
    Args:
        model: Model to set up training for.
        config: Training configuration dictionary.
        device: Training device.
        
    Returns:
        Tuple of (optimizer, scheduler, loss_function).
    """
    # Setup optimizer
    optimizer_config = config.get('optimizer', {})
    optimizer_type = optimizer_config.get('type', 'adamw')
    optimizer_params = optimizer_config.get('params', {})
    
    if optimizer_type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
    elif optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif optimizer_type.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    # Setup scheduler
    scheduler = None
    scheduler_config = config.get('scheduler', {})
    
    if scheduler_config:
        scheduler_type = scheduler_config.get('type', 'plateau')
        scheduler_params = scheduler_config.get('params', {})
        
        if scheduler_type.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, **scheduler_params
            )
        elif scheduler_type.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
        elif scheduler_type.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, **scheduler_params
            )
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}")
    
    # Setup loss function
    loss_config = config.get('loss', {})
    loss_type = loss_config.get('type', 'infonce')
    loss_params = loss_config.get('params', {})
    
    if loss_type.lower() in ['infonce', 'bpr', 'contrastive', 'triplet']:
        loss_function = create_loss_function(loss_type, **loss_params)
    elif loss_type.lower() == 'bce':
        loss_function = nn.BCEWithLogitsLoss(**loss_params)
    elif loss_type.lower() == 'mse':
        loss_function = nn.MSELoss(**loss_params)
    else:
        logger.warning(f"Unknown loss type: {loss_type}, using InfoNCE")
        loss_function = create_loss_function('infonce', **loss_params)
    
    return optimizer, scheduler, loss_function
