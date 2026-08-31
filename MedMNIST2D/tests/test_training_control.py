import json
import tempfile
import unittest
from pathlib import Path

from training_control import (
    data_parallel_device_ids,
    maybe_save_best_checkpoint,
    meets_test_auc_target,
    should_evaluate_test,
    should_stop_early,
)


class TrainingControlTests(unittest.TestCase):
    def test_new_best_score_persists_checkpoint_and_resets_counter(self):
        def write_json(payload, destination):
            Path(destination).write_text(json.dumps(payload), encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best_model.json"
            best_auc, stale_epochs, improved = maybe_save_best_checkpoint(
                validation_auc=0.81,
                best_auc=0.80,
                stale_epochs=4,
                checkpoint={"epoch": 12, "val_auc": 0.81, "net": {"weight": 7}},
                checkpoint_path=checkpoint_path,
                save_checkpoint=write_json,
            )

            self.assertTrue(improved)
            self.assertEqual(best_auc, 0.81)
            self.assertEqual(stale_epochs, 0)
            self.assertEqual(
                json.loads(checkpoint_path.read_text(encoding="utf-8")),
                {"epoch": 12, "val_auc": 0.81, "net": {"weight": 7}},
            )

    def test_stops_on_fifth_non_improvement_from_epoch_75(self):
        self.assertFalse(should_stop_early(epoch=74, stale_epochs=5, start_epoch=75, patience=5))
        self.assertFalse(should_stop_early(epoch=75, stale_epochs=5, start_epoch=75, patience=5))
        self.assertFalse(should_stop_early(epoch=79, stale_epochs=4, start_epoch=75, patience=5))
        self.assertTrue(should_stop_early(epoch=80, stale_epochs=5, start_epoch=75, patience=5))

    def test_uses_data_parallel_only_when_multiple_gpu_ids_are_given(self):
        self.assertEqual(data_parallel_device_ids([]), [])
        self.assertEqual(data_parallel_device_ids([0]), [])
        self.assertEqual(data_parallel_device_ids([0, 1]), [0, 1])

    def test_runs_test_evaluation_on_each_tenth_epoch(self):
        self.assertFalse(should_evaluate_test(epoch=9, interval=10))
        self.assertTrue(should_evaluate_test(epoch=10, interval=10))
        self.assertTrue(should_evaluate_test(epoch=20, interval=10))

    def test_accepts_test_auc_within_one_percent_relative_loss(self):
        self.assertTrue(meets_test_auc_target(current_auc=0.773 * 0.99, target_auc=0.773, max_relative_loss=0.01))
        self.assertFalse(meets_test_auc_target(current_auc=0.773 * 0.9899, target_auc=0.773, max_relative_loss=0.01))


if __name__ == "__main__":
    unittest.main()
