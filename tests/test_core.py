import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from vlm_survival.survival import survival_nll, survival_curves
from vlm_survival.data import fit_preprocessor, transform_row, clinical_text, read_manifest
from vlm_survival.models import TinySurvival
from vlm_survival.train import save_trainable, load_trainable


class CoreTests(unittest.TestCase):
    def test_exact_likelihood_inside_interval(self):
        # softplus(log(expm1(1)))=1 integrated hazard; daily rate 1/10.
        z = torch.full((2, 2), float(np.log(np.expm1(1))))
        loss = survival_nll(z, torch.tensor([5., 15.]), torch.tensor([0., 1.]), [0, 10, 20], 'none')
        np.testing.assert_allclose(loss.numpy(), [.5, 1.5+np.log(10)], rtol=1e-6)

    def test_after_censoring_has_no_gradient(self):
        z = torch.zeros((1, 3), requires_grad=True)
        survival_nll(z, torch.tensor([5.]), torch.tensor([0.]), [0, 10, 20, 30]).backward()
        self.assertGreater(z.grad[0, 0], 0)
        self.assertEqual(z.grad[0, 1:].abs().sum(), 0)

    def test_beyond_horizon_is_administratively_censored(self):
        z = torch.zeros((1, 2))
        a = survival_nll(z, torch.tensor([40.]), torch.tensor([1.]), [0, 10, 20])
        b = survival_nll(z, torch.tensor([20.]), torch.tensor([0.]), [0, 10, 20])
        self.assertAlmostEqual(a.item(), b.item())

    def test_curve_validity(self):
        curves = survival_curves(torch.randn(8, 6)*10)
        self.assertTrue(torch.all((curves >= 0) & (curves <= 1)))
        self.assertTrue(torch.all(curves.diff(dim=-1) <= 0))

    def test_preprocessing_and_no_outcome_in_text(self):
        train = [{'x': '1'}, {'x': '3'}]
        prep = fit_preprocessor(train, ['x'])
        test = {'x': '', 'time_days': '999', 'event': '1'}
        np.testing.assert_array_equal(transform_row(test, prep), [0, 1])
        self.assertEqual(clinical_text(test, ['x']), 'x: not measured')
        self.assertEqual(prep['median'], [2])

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'a.csv'
            path.write_text('patient_id,x\na,1\na,2\n')
            with self.assertRaises(ValueError):
                read_manifest(path, ['x'], False, False)

    def test_tiny_gradient_and_roundtrip(self):
        cfg = {'feature_dim': 16, 'visual_tokens': 2, 'image_mode': 'volume',
               'features': ['x'], 'time_edges': [0, 10, 20]}
        model = TinySurvival(cfg)
        batch = {'image': torch.randn(2, 1, 16, 16, 16), 'tabular': torch.randn(2, 2)}
        prediction = model(batch)
        survival_nll(prediction, torch.tensor([5., 15.]), torch.tensor([1., 0.]), cfg['time_edges']).backward()
        self.assertGreater(model.image.encoder.network[0].weight.grad.abs().sum(), 0)
        with tempfile.TemporaryDirectory() as d:
            save_trainable(model, Path(d)/'weights.pt')
            copy = TinySurvival(cfg)
            load_trainable(copy, Path(d)/'weights.pt')
            torch.testing.assert_close(model(batch), copy(batch))


if __name__ == '__main__':
    unittest.main()
