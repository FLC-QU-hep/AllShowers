import unittest

import torch

from allshowers.data_loader import DataLoader, DictDataSet, ModelInputDict


class TestDataLoader(unittest.TestCase):
    def test_data_loader(self):
        data = ModelInputDict(
            x=torch.randn(10, 10, 3),
            cond=torch.randn(10, 5),
            num_points=torch.randint(1, 11, (10, 5)),
            layer=torch.randint(0, 4, (10, 10)),
            mask=torch.ones(10, 10, dtype=torch.bool),
            label=torch.randint(0, 2, (10, 1)),
            noise=None,
        )
        dataset = DictDataSet(data)
        dataloader = DataLoader(dataset, batch_size=3, drop_last=False, shuffle=False)
        batches = list(dataloader)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0]["x"].shape, (3, 10, 3))
        self.assertEqual(batches[1]["x"].shape, (3, 10, 3))
        self.assertEqual(batches[2]["x"].shape, (3, 10, 3))
        self.assertEqual(batches[3]["x"].shape, (1, 10, 3))
        for batch in batches:
            self.assertTrue(torch.all(batch["mask"]))
            self.assertEqual(batch["x"].shape[0], batch["cond"].shape[0])
            self.assertEqual(batch["x"].shape[0], batch["num_points"].shape[0])
            self.assertEqual(batch["x"].shape[0], batch["layer"].shape[0])
            self.assertEqual(batch["x"].shape[0], batch["label"].shape[0])
            self.assertIsNone(batch["noise"])


if __name__ == "__main__":
    unittest.main()
