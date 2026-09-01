"""Small, dependency-free controls shared by the training loop."""


def data_parallel_device_ids(gpu_ids):
    """Return device IDs only when there is more than one visible GPU."""
    return list(gpu_ids) if len(gpu_ids) > 1 else []


def load_trusted_checkpoint(load_checkpoint, checkpoint_path, device):
    """Load a checkpoint saved by this training script, including optimizer metadata."""
    return load_checkpoint(checkpoint_path, map_location=device, weights_only=False)


def maybe_save_best_checkpoint(
    validation_auc,
    best_auc,
    stale_epochs,
    checkpoint,
    checkpoint_path,
    save_checkpoint,
):
    """Persist a checkpoint for a strict validation-AUC improvement."""
    if validation_auc > best_auc:
        save_checkpoint(checkpoint, checkpoint_path)
        return validation_auc, 0, True

    return best_auc, stale_epochs, False


def should_stop_early(epoch, stale_epochs, start_epoch, patience):
    """Return whether early stopping should end training at this 1-based epoch."""
    return epoch > start_epoch and stale_epochs >= patience


def should_evaluate_test(epoch, interval):
    """Return whether this 1-based epoch is a scheduled test evaluation."""
    if interval <= 0:
        raise ValueError('test evaluation interval must be positive')
    return epoch % interval == 0


def meets_test_auc_target(current_auc, target_auc, max_relative_loss):
    """Return whether test AUC is within the allowed relative loss of target AUC."""
    if target_auc <= 0:
        raise ValueError('target test AUC must be positive')
    return current_auc >= target_auc * (1 - max_relative_loss)
