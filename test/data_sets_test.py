import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from allshowers import data_sets


class TestDataSets(unittest.TestCase):
    def test_batched_histogram(self):
        layer = torch.tensor(
            [
                [0, 0, 1, 1, 2, 2, 3, 0, 0, 0],
                [5, 9, 2, 3, 0, 4, 6, 7, 8, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )

        num_points = torch.tensor([7, 10, 0, 8])
        idx = torch.arange(layer.shape[1])
        mask = idx.reshape(1, -1) < num_points.reshape(-1, 1)

        num_points_per_layer = data_sets.batched_histogram(layer, mask)
        expected = torch.tensor(
            [
                [2, 2, 2, 1, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [8, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.all(num_points_per_layer == expected))

    def test_load_and_prepare(self):
        data = data_sets.load_and_prepare(path="data/showers.h5", stop=100)
        self.assertEqual(type(data), dict)
        self.assertEqual(
            set(data.keys()),
            {
                "label",
                "noise",
                "cond",
                "x",
                "layer",
                "mask",
                "num_points",
            },
        )

    def test_load_and_prepare_with_noise(self):
        data = data_sets.load_and_prepare(
            path="data/showers.h5",
            stop=100,
            return_noise=True,
        )
        self.assertEqual(type(data), dict)
        self.assertEqual(
            set(data.keys()),
            {
                "label",
                "noise",
                "cond",
                "x",
                "layer",
                "mask",
                "num_points",
            },
        )
        self.assertIsNotNone(data["noise"])

    def test_get_data_loaders(self):
        config_dataset = {
            "path": "data/showers.h5",
            "samples_energy_trafo": [
                ["Log", {}],
                ["Affine", {"scale": 2.0, "shift": 1.0}],
            ],
            "stop": 20,
        }
        train_loader, val_loader, trafos = data_sets.get_data_loaders(config_dataset, 2)
        self.assertEqual(len(train_loader), 9)
        self.assertEqual(len(val_loader), 1)
        for data in train_loader:
            self.assertEqual(
                set(data.keys()),
                {
                    "label",
                    "noise",
                    "cond",
                    "x",
                    "layer",
                    "mask",
                    "num_points",
                },
            )
            self.assertLess(torch.min(data["x"][:, :, 2]).item(), 0)
            self.assertEqual(tuple(data["x"].shape)[0:3:2], (2, 3))
        for data in val_loader:
            self.assertEqual(
                set(data.keys()),
                {
                    "label",
                    "noise",
                    "cond",
                    "x",
                    "layer",
                    "mask",
                    "num_points",
                },
            )
        self.assertEqual(
            set(trafos.keys()),
            {"samples_energy_trafo", "samples_coordinate_trafo", "cond_trafo"},
        )
        for element in trafos.values():
            self.assertTrue(isinstance(element, nn.Module))

    def test_load_data(self):
        data = data_sets.load_data("data/showers.h5", start=0, stop=50)
        self.assertEqual(type(data), dict)
        self.assertEqual(
            set(data.keys()), {"shower", "energy", "direction", "pdg", "noise"}
        )
        self.assertEqual(data["shower"].shape[0], 50)
        self.assertIsNone(data["noise"])

    def test_load_data_with_noise(self):
        data = data_sets.load_data(
            "data/showers.h5", start=0, stop=50, return_noise=True
        )
        self.assertIsNotNone(data["noise"])
        self.assertEqual(data["noise"].shape[0], 50)

    def test_create_label_list(self):
        pdg = torch.tensor([11, -11, 22, 11, 22, -11])
        label_list = data_sets.create_label_list(pdg)
        self.assertEqual(label_list, [11, -11, 22])

    def test_create_label_list_sorting(self):
        pdg = torch.tensor([22, -211, 211, 11])
        label_list = data_sets.create_label_list(pdg)
        self.assertEqual(label_list, [11, 22, 211, -211])

    def test_to_label_tensor(self):
        pdg = torch.tensor([11, -11, 22, 11, 22, -11])
        label_list = [11, -11, 22]
        result = data_sets.to_label_tensor(pdg, label_list)
        expected = torch.tensor([0, 1, 2, 0, 2, 1])
        self.assertTrue(torch.all(result == expected))

    def test_to_label_tensor_with_none(self):
        result = data_sets.to_label_tensor(None)
        self.assertIsNone(result)

    def test_to_label_tensor_auto_label_list(self):
        pdg = torch.tensor([22, 11, 22])
        result = data_sets.to_label_tensor(pdg)
        self.assertEqual(result.shape[0], 3)
        self.assertEqual(result.dtype, torch.int64)

    def test_load_and_prepare_with_direction(self):
        data = data_sets.load_and_prepare(
            path="data/showers.h5",
            stop=100,
            return_direction=True,
        )
        self.assertEqual(data["cond"].shape[1], 4)

    def test_load_and_prepare_mask_application(self):
        data = data_sets.load_and_prepare(path="data/showers.h5", stop=50)
        x_masked = data["x"][~data["mask"].repeat(1, 1, 3)]
        self.assertEqual(torch.count_nonzero(x_masked), 0)

    def test_get_data_loaders_split(self):
        config_dataset = {
            "path": "data/showers.h5",
            "stop": 100,
            "val_len": 20,
        }
        train_loader, val_loader, _ = data_sets.get_data_loaders(config_dataset, 10)
        train_samples = sum(batch["x"].shape[0] for batch in train_loader)
        val_samples = sum(batch["x"].shape[0] for batch in val_loader)
        self.assertEqual(train_samples, 80)
        self.assertEqual(val_samples, 20)

    def test_get_data_loaders_split_warning(self):
        config_dataset = {
            "path": "data/showers.h5",
            "stop": 40,
            "val_len": 30,
        }
        with self.assertWarns(UserWarning):
            train_loader, val_loader, _ = data_sets.get_data_loaders(config_dataset, 10)
        train_samples = sum(batch["x"].shape[0] for batch in train_loader)
        val_samples = sum(batch["x"].shape[0] for batch in val_loader)
        self.assertEqual(train_samples, 20)
        self.assertEqual(val_samples, 20)

    def test_get_data_loaders_transformations(self):
        config_dataset = {
            "path": "data/showers.h5",
            "samples_energy_trafo": [["Identity", {}]],
            "samples_coordinate_trafo": [["Identity", {}]],
            "cond_trafo": [["Identity", {}]],
            "stop": 20,
        }
        _, _, trafos = data_sets.get_data_loaders(config_dataset, 5)
        self.assertTrue(all(isinstance(t, nn.Module) for t in trafos.values()))

    def test_get_data_loaders_batch_properties(self):
        config_dataset = {"path": "data/showers.h5", "stop": 25}
        train_loader, _, _ = data_sets.get_data_loaders(config_dataset, 10)
        for batch in train_loader:
            self.assertLessEqual(batch["x"].shape[0], 10)
            self.assertEqual(batch["x"].shape[2], 3)
            break

    def test_get_data_loaders_distributed(self):
        config_dataset = {"path": "data/showers.h5", "stop": 50, "val_len": 10}
        with (
            patch("torch.distributed.barrier"),
            patch("builtins.print"),
            patch("time.sleep"),
            tempfile.TemporaryDirectory() as temporary_directory,
        ):
            trafos_file_path = os.path.join(temporary_directory, "trafos.pt")
            train_loader_0, val_loader_0, _ = data_sets.get_data_loaders(
                config_dataset,
                10,
                world_size=2,
                rank=0,
                trafos_file=trafos_file_path,
            )
            train_loader_1, val_loader_1, _ = data_sets.get_data_loaders(
                config_dataset,
                10,
                world_size=2,
                rank=1,
                trafos_file=trafos_file_path,
            )
        train_samples_0 = sum(batch["x"].shape[0] for batch in train_loader_0)
        val_samples_0 = sum(batch["x"].shape[0] for batch in val_loader_0)

        train_samples_1 = sum(batch["x"].shape[0] for batch in train_loader_1)
        val_samples_1 = sum(batch["x"].shape[0] for batch in val_loader_1)

        self.assertEqual(train_samples_0, 20)
        self.assertEqual(val_samples_0, 10)

        self.assertEqual(train_samples_1, 20)
        self.assertEqual(val_samples_1, 0)

    def test_load_and_prepare_layer_encoding(self):
        data = data_sets.load_and_prepare(path="data/showers.h5", stop=50)
        self.assertEqual(data["layer"].dtype, torch.int64)
        self.assertTrue(torch.all(data["layer"] >= 0))

    def test_load_and_prepare_num_points_default(self):
        data = data_sets.load_and_prepare(path="data/showers.h5", stop=50)
        self.assertEqual(data["num_points"].shape[0], 50)
        self.assertEqual(data["num_points"].dtype, torch.int32)


if __name__ == "__main__":
    unittest.main()
