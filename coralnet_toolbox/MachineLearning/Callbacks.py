"""
Simple callbacks for ultralytics training and evaluation to emit Qt signals.
"""

import logging
from PyQt5.QtCore import pyqtSignal, QObject

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------------------------------------------


def _tensor_to_scalar(value, default=0.0):
    """
    Safely convert a tensor, array, or numeric value to a Python scalar.
    
    Args:
        value: Value to convert (tensor, numpy array, list, or scalar)
        default: Default value if conversion fails
        
    Returns:
        float: Scalar value, or None if the value is multi-element (cannot be converted)
    """
    try:
        # Handle None
        if value is None:
            return default
        
        # Check if it's a tensor (PyTorch)
        if hasattr(value, 'item'):
            # Works for single-element tensors
            return float(value.item())
        
        # Check if it has a shape attribute (numpy array or tensor-like)
        if hasattr(value, 'shape'):
            # Check for multi-element tensors first
            if hasattr(value, 'numel'):
                if value.numel() > 1:
                    # Multi-element tensor - can't convert to single scalar
                    logger.debug(f"Multi-element tensor detected with {value.numel()} elements: shape={value.shape}. Skipping.")
                    return None
            elif len(value.shape) > 0 and value.shape[0] > 1:
                # Multi-element array
                logger.debug(f"Multi-element array detected with shape={value.shape}. Skipping.")
                return None
            
            # Single element or squeeze-able
            if hasattr(value, 'squeeze'):
                squeezed = value.squeeze()
                if hasattr(squeezed, 'item'):
                    return float(squeezed.item())
                return float(squeezed)
            return float(value.flat[0])
        
        # Try direct conversion
        return float(value)
        
    except Exception as e:
        logger.debug(f"Failed to convert value to scalar: {e} (type: {type(value).__name__}). Using default: {default}")
        return default


def _clean_metric_name(key):
    """
    Clean up metric key names by removing common prefixes.
    
    Args:
        key: The metric key name
        
    Returns:
        str: Cleaned key name
    """
    # Remove common prefixes and suffixes
    cleaned = str(key)
    
    # Remove prefixes like "metric_metrics/", "metrics/", etc.
    prefixes_to_remove = ['metric_metrics/', 'metrics/', 'metric_', 'loss_', 'metric_val/', 'val/']
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    
    return cleaned


def _extract_loss_from_trainer(trainer):
    """
    Extract all loss values and available metrics from trainer object.
    
    Args:
        trainer: The ultralytics trainer object
        
    Returns:
        dict: Dictionary of all loss components and metrics
    """
    losses = {}
    
    # Try to get the main loss attributes
    loss_candidates = ['tloss', 'loss', 'losses']
    
    for attr_name in loss_candidates:
        if hasattr(trainer, attr_name):
            value = getattr(trainer, attr_name, None)
            if value is not None:
                if isinstance(value, dict):
                    # If it's a dict of losses, convert all values to scalars
                    for key, val in value.items():
                        scalar_val = _tensor_to_scalar(val)
                        # Only add if we got a valid scalar (not None from multi-element tensors)
                        if scalar_val is not None and (scalar_val > 0 or 'loss' in str(key).lower()):
                            clean_key = _clean_metric_name(key)
                            losses[clean_key] = round(scalar_val, 4)
                            logger.debug(f"Extracted {clean_key}: {scalar_val}")
                else:
                    # Single loss value
                    scalar_val = _tensor_to_scalar(value)
                    if scalar_val is not None and scalar_val > 0:
                        losses[attr_name] = round(scalar_val, 4)
                        logger.debug(f"Extracted {attr_name}: {scalar_val}")
    
    # Try to get metrics if available (some versions include metrics in trainer)
    if hasattr(trainer, 'metrics') and isinstance(trainer.metrics, dict):
        try:
            for key, val in trainer.metrics.items():
                scalar_val = _tensor_to_scalar(val)
                # Only add if we got a valid scalar
                if scalar_val is not None:
                    clean_key = _clean_metric_name(key)
                    if clean_key not in losses:  # Don't override loss values
                        losses[clean_key] = round(scalar_val, 4)
                        logger.debug(f"Extracted metric {clean_key}: {scalar_val}")
        except Exception as e:
            logger.debug(f"Could not extract trainer metrics: {e}")
    
    # If no losses were found, return a dict with status
    if not losses:
        logger.debug("No losses extracted from trainer")
        losses['status'] = 'no_loss_data'
    
    return losses


def _extract_metrics_from_validator(validator):
    """
    Extract metrics from validator object, checking multiple possible locations.
    
    Args:
        validator: The ultralytics validator object
        
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {}
    
    # Try multiple possible attributes where metrics might be stored
    candidates = [
        'metrics',  # Direct metrics attribute
        'results',  # Results dictionary
        'all_results',  # All accumulated results
    ]
    
    for attr_name in candidates:
        if hasattr(validator, attr_name):
            value = getattr(validator, attr_name)
            if value is not None:
                try:
                    if isinstance(value, dict):
                        for key, val in value.items():
                            scalar_val = _tensor_to_scalar(val)
                            # Only add if we got a valid scalar
                            if scalar_val is not None:
                                clean_key = _clean_metric_name(key)
                                metrics[clean_key] = round(scalar_val, 4)
                                logger.debug(f"Extracted metrics.{clean_key}")
                except (TypeError, ValueError) as e:
                    logger.debug(f"Could not process validator.{attr_name}: {e}")
    
    if metrics:
        logger.debug(f"Extracted metrics: {list(metrics.keys())}")
    else:
        logger.debug("No metrics found in validator object")
    
    return metrics


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class TrainingSignalEmitter(QObject):
    """Emits Qt signals during training for status updates."""
    
    # Signals: (epoch, total_epochs, losses_dict, lr)
    # losses_dict contains all individual loss components and metrics
    epoch_completed = pyqtSignal(int, int, dict, float)
    
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
        try:
            total_epochs = trainer.epochs
            msg = f"Training started: {total_epochs} epochs"
            signal_emitter.training_status.emit(msg)
            
            # Debug: Log available attributes in trainer
            logger.info(f"Trainer type: {type(trainer).__name__}")
            loss_attrs = [attr for attr in dir(trainer) if 'loss' in attr.lower()]
            logger.debug(f"Loss-related attributes available: {loss_attrs}")
        except Exception as e:
            logger.error(f"Error in on_train_start callback: {e}", exc_info=True)
            signal_emitter.training_status.emit(f"Error starting training: {e}")
    
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        try:
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            
            # Extract all losses and metrics as a dictionary
            losses_dict = _extract_loss_from_trainer(trainer)
            
            # Get learning rate from optimizer with safeguards
            lr = 0.0
            try:
                if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                    lr = trainer.optimizer.param_groups[0]['lr']
            except (IndexError, KeyError, TypeError, AttributeError) as e:
                logger.warning(f"Failed to get learning rate: {e}")
                lr = 0.0
            
            # Emit all losses and metrics
            signal_emitter.epoch_completed.emit(epoch, total_epochs, losses_dict, lr)
            
            # Create a readable summary for the status message
            loss_str = ", ".join([f"{k}: {v}" for k, v in list(losses_dict.items())[:5]])
            if len(losses_dict) > 5:
                loss_str += f", +{len(losses_dict) - 5} more"
            signal_emitter.training_status.emit(
                f"Epoch {epoch}/{total_epochs} - {loss_str}"
            )
        except Exception as e:
            logger.error(f"Error in on_train_epoch_end callback: {e}", exc_info=True)
            signal_emitter.training_status.emit(f"Error during epoch: {e}")
    
    def on_train_end(trainer):
        """Called when training ends."""
        try:
            msg = "Training completed successfully"
            signal_emitter.training_status.emit(msg)
        except Exception as e:
            logger.error(f"Error in on_train_end callback: {e}", exc_info=True)
            signal_emitter.training_status.emit(f"Error ending training: {e}")
    
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
        try:
            signal_emitter.eval_status.emit("Evaluation started...")
            # Debug: Log available attributes in validator
            logger.info(f"Validator type: {type(validator).__name__}")
            metric_attrs = [attr for attr in dir(validator) if 'metric' in attr.lower() or 'result' in attr.lower()]
            logger.debug(f"Metric-related attributes available: {metric_attrs}")
        except Exception as e:
            logger.error(f"Error in on_val_start callback: {e}", exc_info=True)
    
    def on_val_end(validator):
        """Called when validation ends."""
        try:
            # Extract metrics using intelligent detection
            metrics = _extract_metrics_from_validator(validator)
            
            signal_emitter.eval_status.emit("Evaluation completed")
            signal_emitter.eval_completed.emit(metrics)
        except Exception as e:
            logger.error(f"Error in on_val_end callback: {e}", exc_info=True)
            signal_emitter.eval_status.emit(f"Error during evaluation: {e}")
    
    return {
        'on_val_start': on_val_start,
        'on_val_end': on_val_end,
    }
