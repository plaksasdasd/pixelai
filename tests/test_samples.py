import os

from utils.samples import find_latest_sample_epoch

PROMPTS = [
    "creeper_boy_in_blue_hoodie",
    "red_knight_in_heavy_iron_armor",
    "cyberpunk_neon_green_ninja",
    "golden_king_with_a_royal_crown",
]


def _make_samples(directory, epochs):
    for ep in epochs:
        for prompt in PROMPTS:
            open(os.path.join(directory, f"epoch_{ep}_{prompt}.png"), "w").close()


def test_returns_none_for_missing_dir(tmp_path):
    assert find_latest_sample_epoch(str(tmp_path / "does_not_exist")) is None


def test_returns_none_for_empty_dir(tmp_path):
    assert find_latest_sample_epoch(str(tmp_path)) is None


def test_picks_largest_epoch_without_cap(tmp_path):
    _make_samples(str(tmp_path), [1, 5, 10, 100])
    assert find_latest_sample_epoch(str(tmp_path)) == 100


def test_respects_max_epoch_cap(tmp_path):
    # Samples saved at epoch 1, then every 5 epochs.
    _make_samples(str(tmp_path), [1, 5, 10, 15, 20])
    # The UI should always show the most recent available preview <= current epoch.
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=1) == 1
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=4) == 1
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=5) == 5
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=9) == 5
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=10) == 10
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=26) == 20


def test_prefix_does_not_confuse_10_and_100(tmp_path):
    _make_samples(str(tmp_path), [10, 100])
    # max_epoch=10 must not match epoch_100_* files.
    assert find_latest_sample_epoch(str(tmp_path), max_epoch=10) == 10


def test_ignores_non_sample_files(tmp_path):
    _make_samples(str(tmp_path), [10])
    open(os.path.join(str(tmp_path), "notes.txt"), "w").close()
    open(os.path.join(str(tmp_path), "epoch_x_bad.png"), "w").close()
    open(os.path.join(str(tmp_path), "epoch_50_skin.jpg"), "w").close()
    assert find_latest_sample_epoch(str(tmp_path)) == 10
