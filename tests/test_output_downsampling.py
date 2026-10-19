import numpy as np
import pytest

from wholistic_registration.utils.IO import downsample_xy_mean


def test_downsample_xy_mean_image_crops_incomplete_blocks():
    image = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    result = downsample_xy_mean(image, factor=2)
    expected = image[:, :4, :6].astype(np.float32).reshape(2, 2, 2, 3, 2)
    expected = expected.mean(axis=(2, 4), dtype=np.float32)
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected)


def test_downsample_xy_mean_vector_field_preserves_components():
    field = np.arange(2 * 5 * 7 * 3, dtype=np.float32).reshape(2, 5, 7, 3)
    result = downsample_xy_mean(field, factor=2, vector_field=True)
    expected = field[:, :4, :6, :].reshape(2, 2, 2, 3, 2, 3)
    expected = expected.mean(axis=(2, 4), dtype=np.float32)
    assert result.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(result, expected)


def test_downsample_xy_mean_factor_one_is_unchanged():
    image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    result = downsample_xy_mean(image, factor=1)
    assert result.dtype == image.dtype
    np.testing.assert_array_equal(result, image)


@pytest.mark.parametrize("factor", [0, -1, 1.5, True])
def test_downsample_xy_mean_rejects_invalid_factor(factor):
    with pytest.raises(ValueError, match="positive integer"):
        downsample_xy_mean(np.zeros((4, 4)), factor=factor)
