"""Unit tests for Minimal Reproducible Run (MRR) operations."""

import json
from pathlib import Path

import h5py
import pytest
import torch

from actgpr import mrr


class TestCreateRunDir:
    def test_creates_directory_with_timestamp(self, tmp_path: Path):
        run_dir = mrr.create_run_dir(
            base_path=tmp_path,
            fit_mode="training",
            training_iter=50,
            ei_threshold=0.001,
            max_iterations=20,
            noise=1e-4,
            lengthscale=None,
            outputscale=None,
        )
        assert run_dir.exists()
        assert run_dir.is_dir()
        assert run_dir.parent == tmp_path

    def test_training_mode_folder_name(self, tmp_path: Path):
        run_dir = mrr.create_run_dir(
            base_path=tmp_path,
            fit_mode="training",
            training_iter=50,
            ei_threshold=0.001,
            max_iterations=20,
            noise=1e-4,
            lengthscale=None,
            outputscale=None,
        )
        assert "training50iter" in run_dir.name
        assert "ei0.001" in run_dir.name
        assert "maxiter20" in run_dir.name
        assert "n0.0001" in run_dir.name

    def test_notraining_mode_folder_name(self, tmp_path: Path):
        run_dir = mrr.create_run_dir(
            base_path=tmp_path,
            fit_mode="notraining",
            training_iter=None,
            ei_threshold=0.01,
            max_iterations=15,
            noise=2e-4,
            lengthscale=1.5,
            outputscale=0.5,
        )
        assert "notraining" in run_dir.name
        assert "ei0.01" in run_dir.name
        assert "maxiter15" in run_dir.name
        assert "ls1.5" in run_dir.name
        assert "os0.5" in run_dir.name
        assert "n0.0002" in run_dir.name


class TestWriteConfig:
    def test_writes_valid_json(self, tmp_path: Path):
        config = {"fit_mode": "training", "ei_threshold": 0.001}
        mrr.write_config(tmp_path, config)

        config_path = tmp_path / "config.json"
        assert config_path.exists()

        loaded = json.loads(config_path.read_text())
        assert loaded == config


class TestWriteManifest:
    def test_checksums_config_json(self, tmp_path: Path):
        config = {"fit_mode": "training"}
        mrr.write_config(tmp_path, config)

        mrr.write_manifest(tmp_path)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        loaded = json.loads(manifest_path.read_text())
        assert "config.json" in loaded
        assert loaded["config.json"].startswith("sha256:")

    def test_skips_if_no_config(self, tmp_path: Path):
        mrr.write_manifest(tmp_path)
        assert not (tmp_path / "manifest.json").exists()


class TestWriteMeta:
    def test_contains_environment_info(self, tmp_path: Path):
        import datetime

        start = datetime.datetime.now()
        end = start + datetime.timedelta(seconds=12)

        mrr.write_meta(
            tmp_path,
            run_start=start,
            run_end=end,
            best_x=1.0,
            best_y=-0.5,
            n_iterations=5,
            stop_reason="ei_threshold",
        )

        meta_path = tmp_path / "meta.json"
        assert meta_path.exists()

        loaded = json.loads(meta_path.read_text())
        assert "timestamp_utc" in loaded
        assert loaded["duration_seconds"] == 12.0
        assert "python_version" in loaded
        assert "platform" in loaded
        assert "libraries" in loaded
        assert "git_commit" in loaded

    def test_contains_package_identity(self, tmp_path: Path):
        """Test that meta.json identifies the package on its own."""
        import datetime

        start = datetime.datetime.now()

        mrr.write_meta(
            tmp_path,
            run_start=start,
            run_end=start,
            best_x=1.0,
            best_y=-0.5,
            n_iterations=5,
            stop_reason="ei_threshold",
        )

        loaded = json.loads((tmp_path / "meta.json").read_text())
        assert loaded["package_name"] == "actgpr"
        assert loaded["actgpr_version"] != "unknown"
        assert loaded["repository"] == "https://github.com/AlxndrSchroeder/actgpr"

    def test_contains_output_summary(self, tmp_path: Path):
        import datetime

        start = datetime.datetime.now()

        mrr.write_meta(
            tmp_path,
            run_start=start,
            run_end=start,
            best_x=1.23,
            best_y=-0.45,
            n_iterations=10,
            stop_reason="max_iterations",
        )

        loaded = json.loads((tmp_path / "meta.json").read_text())
        assert loaded["output_summary"]["best_x"] == 1.23
        assert loaded["output_summary"]["best_y"] == -0.45
        assert loaded["output_summary"]["n_iterations"] == 10
        assert loaded["output_summary"]["stop_reason"] == "max_iterations"


class TestSaveHdf5:
    @pytest.fixture
    def dummy_data(self):
        config = {"noise": 1e-4, "fit_mode": "training"}
        results = [
            {
                "iteration": 1,
                "next_point": 0.5,
                "new_y": 0.2,
                "current_best": 0.2,
                "max_ei": 0.1,
                "prediction_error": 0.05,
                "improvement": 0.0,
            }
        ]
        return config, results

    def test_writes_hdf5_file(self, tmp_path: Path, dummy_data):
        config, results = dummy_data
        mrr.save_hdf5(
            tmp_path,
            results=results,
            config=config,
            store_snapshots=False,
            final_train_x=torch.tensor([0.0, 0.5]),
            final_train_y=torch.tensor([1.0, 0.2]),
            best_x=0.5,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=1,
        )

        h5_path = tmp_path / "results.h5"
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert f.attrs["noise"] == 1e-4
            assert f.attrs["fit_mode"] == "training"

            assert "final" in f
            assert f["final"].attrs["best_x"] == 0.5
            assert "train_x" in f["final"]
            assert len(f["final/train_x"]) == 2

    def test_history_group_has_scalar_series(self, tmp_path: Path, dummy_data):
        config, results = dummy_data
        mrr.save_hdf5(
            tmp_path,
            results=results,
            config=config,
            store_snapshots=False,
            final_train_x=torch.tensor([0.0, 0.5]),
            final_train_y=torch.tensor([1.0, 0.2]),
            best_x=0.5,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=1,
        )

        with h5py.File(tmp_path / "results.h5", "r") as f:
            # Per-iteration scalars live once, as aligned datasets under history.
            history = f["history"]
            for field in (
                "iteration",
                "next_point",
                "new_y",
                "current_best",
                "max_ei",
                "prediction_error",
                "improvement",
            ):
                assert field in history, f"missing history dataset: {field}"
                assert len(history[field]) == len(results)

            assert history["iteration"][0] == 1
            assert history["next_point"][0] == 0.5
            assert history["prediction_error"][0] == 0.05
            assert history["improvement"][0] == 0.0

            # Scalars are no longer duplicated as iter_NNN attributes.
            assert "iterations" not in f

    def test_tensor_datasets_only_with_snapshots(self, tmp_path: Path, dummy_data):
        config, results = dummy_data

        # Add tensors to results
        results[0].update(
            {
                "candidates": torch.tensor([0.1, 0.2]),
                "f_mean": torch.tensor([0.1, 0.2]),
                "f_var": torch.tensor([0.1, 0.2]),
                "ei_scores": torch.tensor([0.1, 0.2]),
                "train_x": torch.tensor([0.0]),
                "train_y": torch.tensor([1.0]),
            }
        )

        mrr.save_hdf5(
            tmp_path,
            results=results,
            config=config,
            store_snapshots=True,
            final_train_x=torch.tensor([0.0, 0.5]),
            final_train_y=torch.tensor([1.0, 0.2]),
            best_x=0.5,
            best_y=0.2,
            stop_reason="max_iterations",
            n_iterations=1,
        )

        with h5py.File(tmp_path / "results.h5", "r") as f:
            grp = f["iterations/iter_001"]
            assert "candidates" in grp
            assert "f_mean" in grp


class TestSetupFileLogger:
    def test_creates_run_log_file(self, tmp_path: Path):
        import logging

        logger = logging.getLogger("actgpr")
        handler = mrr.setup_file_logger(tmp_path)
        try:
            log_path = tmp_path / "run.log"
            assert log_path.exists()

            logger.info("Test message")

            content = log_path.read_text()
            assert "Test message" in content
        finally:
            # Detach and close so the handler does not leak into other tests
            logger.removeHandler(handler)
            handler.close()

        logger.removeHandler(handler)
