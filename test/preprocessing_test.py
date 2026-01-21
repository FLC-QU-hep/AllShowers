import unittest

import torch

from allshowers import preprocessing


class TestDataUtil(unittest.TestCase):
    def test_identity(self):
        identity = preprocessing.Identity()
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(x, identity.forward(x)))
        self.assertTrue(torch.allclose(x, identity.inverse(x)))

    def test_log(self):
        alpha = 1e-6
        log = preprocessing.Log(alpha)
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(x, log.inverse(log.forward(x))))
        self.assertTrue(torch.allclose(x, log.forward(log.inverse(x)), atol=1e-6))
        self.assertTrue(torch.allclose(torch.log(x + alpha), log.forward(x)))

    def test_logit(self):
        alpha = 1e-6
        logit = preprocessing.LogIt(alpha)
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(x, logit.inverse(logit.forward(x))))
        self.assertTrue(torch.allclose(x, logit.forward(logit.inverse(x)), atol=1e-6))
        x_ = (1 - 2 * alpha) * x + alpha
        self.assertTrue(torch.allclose(torch.log(x_ / (1 - x_)), logit.forward(x)))

    def test_affine(self):
        affine = preprocessing.Affine(scale=2.0, shift=1.0)
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(x, affine.inverse(affine.forward(x))))
        self.assertTrue(torch.allclose(x, affine.forward(affine.inverse(x))))
        self.assertTrue(torch.allclose(2 * x + 1, affine.forward(x)))

    def test_sequence(self):
        alpha = 1e-6
        logit = preprocessing.LogIt(alpha)
        affine = preprocessing.Affine(scale=2.0, shift=1.0)
        identity = preprocessing.Identity()
        sequence = preprocessing.Sequence([logit, affine, identity])
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(x, sequence.inverse(sequence.forward(x))))
        self.assertTrue(
            torch.allclose(x, sequence.forward(sequence.inverse(x)), atol=1e-6)
        )

    def test_standard_scaler(self):
        scaler = preprocessing.StandardScaler((1, 10))
        x = torch.rand(100, 10)
        scaler.fit(x)
        x_ = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True)
        self.assertTrue(torch.allclose(x_, scaler.forward(x)))
        self.assertTrue(torch.allclose(x, scaler.inverse(scaler.forward(x))))

    def test_compose(self):
        conf = [
            ["LogIt", {"alpha": 1e-6}],
            ["Affine", {"scale": 2.0, "shift": 1.0}],
            ["Identity", {}],
        ]
        sequence1 = preprocessing.compose(conf)
        sequence2 = preprocessing.Sequence(
            [
                preprocessing.LogIt(alpha=1e-6),
                preprocessing.Affine(scale=2.0, shift=1.0),
                preprocessing.Identity(),
            ]
        )
        x = torch.rand(10, 10)
        self.assertTrue(torch.allclose(sequence1.forward(x), sequence2.forward(x)))
        self.assertTrue(torch.allclose(sequence1.inverse(x), sequence2.inverse(x)))
        self.assertTrue(torch.allclose(sequence1.inverse(sequence1.forward(x)), x))
        self.assertTrue(
            torch.allclose(sequence1.forward(sequence1.inverse(x)), x, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
