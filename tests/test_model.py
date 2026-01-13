import torch
import unittest
from diffusion_weather.models import DiffusionModel


class TestDiffusionModel(unittest.TestCase):
    def test_initialization(self):
        model = DiffusionModel(channels_in=3, channels_out=3, aux_channels=0)
        self.assertIsInstance(model, DiffusionModel)
        self.assertEqual(model.total_input_channels, 3)

    def test_forward_no_aux(self):
        model = DiffusionModel(channels_in=3, channels_out=3, aux_channels=0)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_forward_with_aux_passed(self):
        model = DiffusionModel(channels_in=3, channels_out=3, aux_channels=2)
        x = torch.randn(1, 3, 32, 32)
        cond = torch.randn(1, 2, 32, 32)
        out = model(x, conditioning=cond)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_forward_with_aux_none(self):
        # Test backward compatibility/default behavior
        model = DiffusionModel(channels_in=3, channels_out=3, aux_channels=2)
        x = torch.randn(1, 3, 32, 32)
        # Should internally pad with zeros
        out = model(x, conditioning=None)
        self.assertEqual(out.shape, (1, 3, 32, 32))

    def test_shape_mismatch(self):
        model = DiffusionModel(channels_in=3, channels_out=3, aux_channels=2)
        x = torch.randn(1, 3, 32, 32)
        cond = torch.randn(1, 1, 32, 32)  # Wrong channels
        with self.assertRaises(ValueError):
            model(x, conditioning=cond)


if __name__ == "__main__":
    unittest.main()
