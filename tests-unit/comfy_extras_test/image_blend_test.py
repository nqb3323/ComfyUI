import sys
from unittest.mock import patch, MagicMock

# `comfy.model_management` initializes the GPU at module import time, which
# fails in CPU-only environments. Stub it out before any `comfy.*` imports
# load it transitively. We don't use it in these tests.
sys.modules.setdefault("comfy.model_management", MagicMock())

import torch  # noqa: E402

# Mock nodes module to prevent CUDA initialization during import
mock_nodes = MagicMock()
mock_nodes.MAX_RESOLUTION = 16384

# Mock server module for PromptServer
mock_server = MagicMock()

with patch.dict("sys.modules", {"nodes": mock_nodes, "server": mock_server}):
    from comfy_extras.nodes_post_processing import Blend  # noqa: E402


class TestImageBlend:
    """Regression tests for the ImageBlend node, especially channel-count handling."""

    def create_test_image(self, batch_size=1, height=64, width=64, channels=3):
        return torch.rand(batch_size, height, width, channels)

    def test_same_shape_rgb(self):
        """Baseline: identical RGB inputs produce an RGB output."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=3)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 3)

    def test_rgb_plus_rgba(self):
        """RGB image1 + RGBA image2 should pad image1 to 4 channels."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=4)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 4)

    def test_rgba_plus_rgb(self):
        """RGBA image1 + RGB image2 should pad image2 to 4 channels."""
        image1 = self.create_test_image(channels=4)
        image2 = self.create_test_image(channels=3)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 4)

    def test_channel_gap_larger_than_one(self):
        """Channel-count gap > 1 (e.g. 3 vs 5) should not raise.

        This is the exact runtime error reported in CORE-103:
        'The size of tensor a (5) must match the size of tensor b (3) at
        non-singleton dimension 3'.

        The output is capped at 4 channels (RGBA) because downstream
        SaveImage/PreviewImage rely on PIL.Image.fromarray, which only
        supports 1/3/4-channel arrays. Without this cap, the failure would
        just shift from blend-time to save-time.
        """
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=5)
        result = Blend.execute(image1, image2, 0.5, "multiply")
        assert result[0].shape == (1, 64, 64, 4)

    def test_output_capped_at_four_channels(self):
        """Both inputs having > 4 channels should still produce a 4-channel
        output, since SaveImage/PreviewImage cannot serialize anything
        wider than RGBA via PIL.Image.fromarray."""
        image1 = self.create_test_image(channels=6)
        image2 = self.create_test_image(channels=5)
        result = Blend.execute(image1, image2, 0.5, "normal")
        assert result[0].shape == (1, 64, 64, 4)

    def test_save_compatible_output_passes_through_pil(self):
        """The blended result must be convertible by PIL.Image.fromarray,
        which is what SaveImage/PreviewImage do downstream. Catches the
        case where a >4-channel output would silently break save/preview."""
        from PIL import Image
        import numpy as np

        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=5)
        result = Blend.execute(image1, image2, 0.5, "normal")
        # Mirror SaveImage's exact conversion (nodes.py:1662)
        arr = np.clip(255.0 * result[0][0].cpu().numpy(), 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        assert img.mode in ("L", "RGB", "RGBA"), (
            f"Output mode {img.mode!r} cannot be saved by SaveImage"
        )

    def test_different_size_and_channels(self):
        """Different spatial size AND different channel counts should both be reconciled."""
        image1 = self.create_test_image(height=64, width=64, channels=3)
        image2 = self.create_test_image(height=32, width=32, channels=4)
        result = Blend.execute(image1, image2, 0.5, "screen")
        assert result[0].shape == (1, 64, 64, 4)

    def test_all_blend_modes_with_channel_mismatch(self):
        """Every blend mode should work with mismatched channel counts."""
        image1 = self.create_test_image(channels=3)
        image2 = self.create_test_image(channels=4)
        for mode in [
            "normal",
            "multiply",
            "screen",
            "overlay",
            "soft_light",
            "difference",
        ]:
            result = Blend.execute(image1, image2, 0.5, mode)
            assert result[0].shape == (1, 64, 64, 4), (
                f"blend mode {mode} produced wrong shape"
            )

    def test_output_clamped(self):
        """Output values should be clamped to [0, 1] even when intermediate
        results would go negative.

        With `difference` mode, image1=0 and image2=1, the unclamped blend
        produces image1*(1-bf) + (image1-image2)*bf = -bf, which is negative.
        The output therefore exercises the clamp branch.
        """
        image1 = torch.zeros(1, 8, 8, 3)
        image2 = torch.ones(1, 8, 8, 3)
        result = Blend.execute(image1, image2, 0.5, "difference")
        assert result[0].min() >= 0.0
        assert result[0].max() <= 1.0
        # All pixels would be -0.5 without the clamp; verify they were clipped to 0.
        assert torch.all(result[0] == 0.0)

    def test_padding_value_is_one(self):
        """Verify the padded channel(s) are filled with 1.0, not 0.0 or some
        other value. This is the semantic guarantee of the channel-alignment
        logic (it acts like an opaque alpha channel).

        Setup: image1 has 3 channels of zeros, image2 has 4 channels of ones.
        After padding, image1 becomes [0, 0, 0, X] where X is the pad value.
        With `multiply` blend_mode and blend_factor=1.0:
            output = image1 * (1 - 1) + (image1 * image2) * 1
                   = image1 * image2
                   = [0, 0, 0, X * 1] = [0, 0, 0, X]
        So output channel 4 reveals the pad value used for image1.
        """
        image1 = torch.zeros(1, 4, 4, 3)
        image2 = torch.ones(1, 4, 4, 4)
        result = Blend.execute(image1, image2, 1.0, "multiply")
        assert result[0].shape == (1, 4, 4, 4)
        # First three channels: 0 * 1 = 0
        assert torch.all(result[0][..., :3] == 0.0)
        # Fourth channel: pad_value * 1 = pad_value -> must be 1.0
        assert torch.all(result[0][..., 3] == 1.0)
