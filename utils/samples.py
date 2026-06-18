import os


def find_latest_sample_epoch(samples_dir, max_epoch=None):
    """Return the most recent epoch number that has saved sample images.

    Sample images are written periodically (not every epoch), so the current
    epoch usually has no file yet. Instead of guessing the epoch number, scan
    the samples directory and return the largest epoch <= max_epoch that
    actually has files. Returns None if no samples are available.
    """
    if not os.path.isdir(samples_dir):
        return None
    epochs = set()
    for f in os.listdir(samples_dir):
        if not (f.startswith("epoch_") and f.lower().endswith(".png")):
            continue
        parts = f.split("_")
        try:
            ep = int(parts[1])
        except (IndexError, ValueError):
            continue
        if max_epoch is None or ep <= max_epoch:
            epochs.add(ep)
    return max(epochs) if epochs else None
