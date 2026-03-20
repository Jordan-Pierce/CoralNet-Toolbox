"""
Simple callbacks for ultralytics training and evaluation to emit Qt signals.
"""

from PyQt5.QtCore import pyqtSignal, QObject


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TrainingSignalEmitter(QObject):
    """Emits Qt signals during training for status updates."""
    
    # Signals: (epoch, total_epochs, loss, lr)
    epoch_completed = pyqtSignal(int, int, float, float)
    
    # Signal: (message)
    training_status = pyqtSignal(str)


class EvaluationSignalEmitter(QObject):
    """Emits Qt signals during evaluation for status updates."""
    
    # Signal: (message)
    eval_status = pyqtSignal(str)
    
    # Signal: (metrics_dict)
    eval_completed = pyqtSignal(dict)


def create_training_callbacks(signal_emitter):
    """
    Create training callbacks that emit Qt signals.
    
    Args:
        signal_emitter: TrainingSignalEmitter instance
        
    Returns:
        dict: Callbacks to add to model
    """
    
    def on_train_start(trainer):
        """Called when training starts."""
        total_epochs = trainer.epochs
        msg = f"Training started: {total_epochs} epochs"
        signal_emitter.training_status.emit(msg)
    
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        epoch = trainer.epoch + 1
        total_epochs = trainer.epochs
        
        # Get loss - trainer.tloss is the averaged loss
        loss = float(trainer.tloss) if hasattr(trainer, 'tloss') else 0.0
        
        # Get learning rate from optimizer
        lr = trainer.optimizer.param_groups[0]['lr']
        
        signal_emitter.epoch_completed.emit(epoch, total_epochs, loss, lr)
        signal_emitter.training_status.emit(
            f"Epoch {epoch}/{total_epochs} - Loss: {loss:.4f}"
        )
    
    def on_train_end(trainer):
        """Called when training ends."""
        msg = "Training completed successfully"
        signal_emitter.training_status.emit(msg)
    
    return {
        'on_train_start': on_train_start,
        'on_train_epoch_end': on_train_epoch_end,
        'on_train_end': on_train_end,
    }


def create_evaluation_callbacks(signal_emitter):
    """
    Create evaluation callbacks that emit Qt signals.
    
    Args:
        signal_emitter: EvaluationSignalEmitter instance
        
    Returns:
        dict: Callbacks to add to validator
    """
    
    def on_val_start(validator):
        """Called when validation starts."""
        signal_emitter.eval_status.emit("Evaluation started...")
    
    def on_val_end(validator):
        """Called when validation ends."""
        # Gather key metrics
        metrics = {}
        if hasattr(validator, 'metrics') and validator.metrics:
            metrics = dict(validator.metrics)
        
        signal_emitter.eval_status.emit("Evaluation completed")
        signal_emitter.eval_completed.emit(metrics)
    
    return {
        'on_val_start': on_val_start,
        'on_val_end': on_val_end,
    }
