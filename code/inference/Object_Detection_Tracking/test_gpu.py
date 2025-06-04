import pytest

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None


def test_gpu_available():
    if tf is None or not tf.test.is_built_with_cuda():
        pytest.skip("TensorFlow GPU not available")
    assert tf.test.gpu_device_name()
