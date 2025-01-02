import pytest

from src.imageNoiseInfo import(
    calculate_mse,
    calculate_psnr,
    calculate_snr,
    apply_average_filter,
    apply_median_filter,
    apply_median_filter_on_noise,
    detect_noise_dynamic
)

# Sample input data for tests
@pytest.fixture
def sample_image():
    """Provides a sample 3x3 grayscale image."""
    return [
        [50, 80, 90],
        [120, 150, 160],
        [200, 220, 230]
    ]

@pytest.fixture
def noisy_image():
    """Provides a sample noisy 3x3 image."""
    return [
        [52, 78, 92],
        [119, 155, 158],
        [198, 225, 234]
    ]

def test_calculate_mse(sample_image, noisy_image):
    """Tests the MSE calculation."""
    mse = calculate_mse(sample_image, noisy_image)
    expected_mse = 9.6667  # Manually calculated
    assert round(mse, 4) == expected_mse

    mse_same = calculate_mse(sample_image, sample_image)
    assert mse_same == 0

def test_calculate_psnr(sample_image, noisy_image):
    """Tests the PSNR calculation."""
    psnr = calculate_psnr(sample_image, noisy_image)
    assert round(psnr, 4) == 38.2780  # manually calculated

def test_calculate_snr(sample_image, noisy_image):
    """Tests the SNR calculation."""
    snr = calculate_snr(sample_image, noisy_image)
    assert snr > 20  # Signal should dominate noise in this example

def test_apply_average_filter(noisy_image):
    """Tests the average filter."""
    filtered = apply_average_filter(noisy_image)
    assert filtered == [
        [44, 72, 53],
        [91, 145, 104],
        [77, 121, 85]
    ]  # Manually calculated output

def test_apply_median_filter(sample_image):
    """Tests the median filter."""
    filtered = apply_median_filter(sample_image)
    assert filtered == [
        [80, 90, 90],
        [120, 150, 150],
        [200, 200, 200]
    ]  # Manually calculated output

def test_detect_noise_dynamic(sample_image, noisy_image):
    """Tests noise detection."""
    noise_mask = detect_noise_dynamic(noisy_image)
    assert noise_mask == [
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 1]
    ]  # Expected based on local thresholds

def test_apply_median_filter_on_noise(sample_image, noisy_image):
    """Tests targeted median filtering."""
    noise_mask = detect_noise_dynamic(noisy_image)
    filtered = apply_median_filter_on_noise(noisy_image, noise_mask)
    assert filtered == [
        [50, 78, 90],
        [119, 150, 158],
        [198, 220, 230]
    ]  # Only noisy pixels are filtered
