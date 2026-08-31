"""Small, dependency-free controls shared by the training loop."""


def data_parallel_device_ids(gpu_ids):
    """Return device IDs only when there is more than one visible GPU."""
    return list(gpu_ids) if len(gpu_ids) > 1 else []


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
