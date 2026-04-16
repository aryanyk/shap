import numpy as np

from shap.benchmark._explanation_error import ExplanationError


class DummyMasker:
    clustering = None


def test_explanation_error_preserves_caller_rng_stream(monkeypatch):
    """
    Verify ExplanationError does not perturb caller's NumPy RNG stream.

    The benchmark uses its own seed internally, but should restore the
    caller's RNG state afterward so downstream random draws are
    unaffected.

    This test validates the fix for the bug where np.random.seed()
    was used instead of np.random.get_state()/set_state().
    """

    class DummyMaskedModel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, masks):
            return np.zeros(len(masks), dtype=float)

    monkeypatch.setattr(
        "shap.benchmark._explanation_error.MaskedModel", DummyMaskedModel
    )

    x = np.array([[1.0, 2.0]])
    attributions = np.array([[0.0, 0.0]])

    metric = ExplanationError(
        DummyMasker(),
        lambda data: np.sum(data, axis=1),
        x,
        num_permutations=2,
        seed=12345,
    )

    # Control: random stream without metric call
    np.random.seed(2024)
    control_before = np.random.rand(5)
    control_after = np.random.rand(5)

    # Test: random stream with metric call in between
    np.random.seed(2024)
    test_before = np.random.rand(5)
    metric(attributions, name="test", silent=True)
    test_after = np.random.rand(5)

    # The metric should not have affected the RNG stream
    np.testing.assert_array_equal(test_before, control_before)
    np.testing.assert_array_equal(test_after, control_after)


def test_explanation_error_internal_reproducibility(monkeypatch):
    """
    Verify ExplanationError produces identical results with same seed.

    Results should be identical regardless of external RNG state.
    """

    class DummyMaskedModel:
        def __init__(self, *args, **kwargs):
            self.call_count = 0

        def __call__(self, masks):
            # Use RNG to make results vary if seed isn't working
            self.call_count += 1
            return np.random.rand(len(masks))

    monkeypatch.setattr(
        "shap.benchmark._explanation_error.MaskedModel", DummyMaskedModel
    )

    x = np.array([[1.0, 2.0, 3.0]])
    attributions = np.array([[0.1, 0.2, 0.3]])

    # Run 1: with external RNG state A
    np.random.seed(111)
    metric1 = ExplanationError(
        DummyMasker(),
        lambda data: np.sum(data, axis=1),
        x,
        num_permutations=3,
        seed=99999,
    )
    result1 = metric1(attributions, name="test1", silent=True)

    # Run 2: with external RNG state B
    np.random.seed(222)
    metric2 = ExplanationError(
        DummyMasker(),
        lambda data: np.sum(data, axis=1),
        x,
        num_permutations=3,
        seed=99999,
    )
    result2 = metric2(attributions, name="test2", silent=True)

    # Results should be identical despite different external RNG states
    assert result1.value == result2.value
