import pytest
import numpy as np
from opto.features.priority_search.vector_score import VectorScore


class TestVectorScoreInit:
    """Test VectorScore initialization."""

    def test_init_with_list(self):
        """Test initialization with a list."""
        score = VectorScore([1, 2, 3])
        assert isinstance(score.vector, np.ndarray)
        assert np.array_equal(score.vector, np.array([1, 2, 3]))
        assert score.comparison_mode == 'sequence'

    def test_init_with_numpy_array(self):
        """Test initialization with a numpy array."""
        arr = np.array([4.5, 5.5, 6.5])
        score = VectorScore(arr)
        assert isinstance(score.vector, np.ndarray)
        assert np.array_equal(score.vector, arr)

    def test_init_with_custom_comparison_mode(self):
        """Test initialization with different comparison modes."""
        modes = ['sequence', 'sum', 'any', 'all', 'random']
        for mode in modes:
            score = VectorScore([1, 2, 3], comparison_mode=mode)
            assert score.comparison_mode == mode

    def test_init_with_single_element(self):
        """Test initialization with a single element."""
        score = VectorScore([5])
        assert score.vector.shape == (1,)
        assert score.vector[0] == 5


class TestVectorScoreLessThan:
    """Test VectorScore __lt__ (less than) operator."""

    def test_lt_sequence_mode_first_smaller(self):
        """Test sequence mode: first element decides."""
        score1 = VectorScore([1, 5, 10], comparison_mode='sequence')
        score2 = VectorScore([2, 3, 8], comparison_mode='sequence')
        assert score1 < score2

    def test_lt_sequence_mode_first_larger(self):
        """Test sequence mode: first element larger."""
        score1 = VectorScore([3, 1, 1], comparison_mode='sequence')
        score2 = VectorScore([2, 5, 10], comparison_mode='sequence')
        assert not (score1 < score2)

    def test_lt_sequence_mode_equal_first_second_decides(self):
        """Test sequence mode: when first equal, second element decides."""
        score1 = VectorScore([2, 3, 10], comparison_mode='sequence')
        score2 = VectorScore([2, 5, 1], comparison_mode='sequence')
        assert score1 < score2

    def test_lt_sequence_mode_all_equal(self):
        """Test sequence mode: all elements equal."""
        score1 = VectorScore([2, 3, 4], comparison_mode='sequence')
        score2 = VectorScore([2, 3, 4], comparison_mode='sequence')
        assert not (score1 < score2)

    def test_lt_sum_mode_smaller(self):
        """Test sum mode: sum is smaller."""
        score1 = VectorScore([1, 2, 3], comparison_mode='sum')  # sum = 6
        score2 = VectorScore([2, 3, 4], comparison_mode='sum')  # sum = 9
        assert score1 < score2

    def test_lt_sum_mode_larger(self):
        """Test sum mode: sum is larger."""
        score1 = VectorScore([5, 5, 5], comparison_mode='sum')  # sum = 15
        score2 = VectorScore([2, 3, 4], comparison_mode='sum')  # sum = 9
        assert not (score1 < score2)

    def test_lt_sum_mode_equal(self):
        """Test sum mode: sums are equal."""
        score1 = VectorScore([1, 2, 6], comparison_mode='sum')  # sum = 9
        score2 = VectorScore([3, 3, 3], comparison_mode='sum')  # sum = 9
        assert not (score1 < score2)

    def test_lt_any_mode_at_least_one_smaller(self):
        """Test any mode: at least one element is smaller."""
        score1 = VectorScore([1, 10, 10], comparison_mode='any')
        score2 = VectorScore([5, 5, 5], comparison_mode='any')
        assert score1 < score2

    def test_lt_any_mode_all_larger(self):
        """Test any mode: all elements are larger."""
        score1 = VectorScore([10, 10, 10], comparison_mode='any')
        score2 = VectorScore([5, 5, 5], comparison_mode='any')
        assert not (score1 < score2)

    def test_lt_all_mode_all_smaller(self):
        """Test all mode: all elements are smaller."""
        score1 = VectorScore([1, 2, 3], comparison_mode='all')
        score2 = VectorScore([4, 5, 6], comparison_mode='all')
        assert score1 < score2

    def test_lt_all_mode_one_not_smaller(self):
        """Test all mode: at least one element is not smaller."""
        score1 = VectorScore([1, 5, 3], comparison_mode='all')
        score2 = VectorScore([4, 5, 6], comparison_mode='all')
        assert not (score1 < score2)

    def test_lt_random_mode_both_smaller_and_larger(self):
        """Test random mode: some elements smaller, some larger."""
        score1 = VectorScore([1, 10], comparison_mode='random')
        score2 = VectorScore([5, 5], comparison_mode='random')
        # Result is random, so we just test it doesn't crash
        result = score1 < score2
        assert isinstance(result, bool)

    def test_lt_random_mode_all_smaller(self):
        """Test random mode: all elements smaller."""
        score1 = VectorScore([1, 2, 3], comparison_mode='random')
        score2 = VectorScore([4, 5, 6], comparison_mode='random')
        assert score1 < score2

    def test_lt_different_comparison_modes_raises_assertion(self):
        """Test that comparing different comparison modes raises assertion."""
        score1 = VectorScore([1, 2, 3], comparison_mode='sequence')
        score2 = VectorScore([1, 2, 3], comparison_mode='sum')
        with pytest.raises(AssertionError):
            _ = score1 < score2

    def test_lt_different_shapes_raises_assertion(self):
        """Test that comparing different shapes raises assertion."""
        score1 = VectorScore([1, 2, 3], comparison_mode='sequence')
        score2 = VectorScore([1, 2], comparison_mode='sequence')
        with pytest.raises(AssertionError):
            _ = score1 < score2


class TestVectorScoreLessThanOrEqual:
    """Test VectorScore __le__ (less than or equal) operator."""

    def test_le_smaller(self):
        """Test less than or equal: smaller case."""
        score1 = VectorScore([1, 2, 3], comparison_mode='sum')
        score2 = VectorScore([4, 5, 6], comparison_mode='sum')
        assert score1 <= score2

    def test_le_equal(self):
        """Test less than or equal: equal case."""
        score1 = VectorScore([1, 2, 3], comparison_mode='sum')
        score2 = VectorScore([1, 2, 3], comparison_mode='sum')
        assert score1 <= score2

    def test_le_larger(self):
        """Test less than or equal: larger case."""
        score1 = VectorScore([4, 5, 6], comparison_mode='sum')
        score2 = VectorScore([1, 2, 3], comparison_mode='sum')
        assert not (score1 <= score2)


class TestVectorScoreEqual:
    """Test VectorScore __eq__ (equality) operator."""

    def test_eq_same_values(self):
        """Test equality with same values."""
        score1 = VectorScore([1, 2, 3])
        score2 = VectorScore([1, 2, 3])
        assert score1 == score2

    def test_eq_different_values(self):
        """Test equality with different values."""
        score1 = VectorScore([1, 2, 3])
        score2 = VectorScore([1, 2, 4])
        assert not (score1 == score2)

    def test_eq_different_lengths(self):
        """Test equality with different lengths."""
        score1 = VectorScore([1, 2, 3])
        score2 = VectorScore([1, 2])
        # np.array_equal returns False for different shapes, no assertion error
        assert not (score1 == score2)

    def test_eq_float_values(self):
        """Test equality with float values."""
        score1 = VectorScore([1.0, 2.5, 3.7])
        score2 = VectorScore([1.0, 2.5, 3.7])
        assert score1 == score2


class TestVectorScoreNegation:
    """Test VectorScore __neg__ (negation) operator."""

    def test_neg_positive_values(self):
        """Test negation of positive values."""
        score = VectorScore([1, 2, 3], comparison_mode='sum')
        neg_score = -score
        assert np.array_equal(neg_score.vector, np.array([-1, -2, -3]))
        assert neg_score.comparison_mode == 'sum'

    def test_neg_negative_values(self):
        """Test negation of negative values."""
        score = VectorScore([-1, -2, -3], comparison_mode='sequence')
        neg_score = -score
        assert np.array_equal(neg_score.vector, np.array([1, 2, 3]))
        assert neg_score.comparison_mode == 'sequence'

    def test_neg_mixed_values(self):
        """Test negation of mixed positive and negative values."""
        score = VectorScore([1, -2, 3], comparison_mode='all')
        neg_score = -score
        assert np.array_equal(neg_score.vector, np.array([-1, 2, -3]))
        assert neg_score.comparison_mode == 'all'


class TestVectorScoreEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_array(self):
        """Test with empty array."""
        score = VectorScore([])
        assert score.vector.shape == (0,)

    def test_large_array(self):
        """Test with large array."""
        large_arr = list(range(1000))
        score = VectorScore(large_arr)
        assert score.vector.shape == (1000,)
        assert score.vector[0] == 0
        assert score.vector[-1] == 999

    def test_comparison_with_floats(self):
        """Test comparisons with float values."""
        score1 = VectorScore([1.1, 2.2, 3.3], comparison_mode='sum')
        score2 = VectorScore([1.2, 2.1, 3.4], comparison_mode='sum')
        assert score1 < score2

    def test_comparison_with_negative_values(self):
        """Test comparisons with negative values."""
        score1 = VectorScore([-1, -2, -3], comparison_mode='sum')
        score2 = VectorScore([-4, -5, -6], comparison_mode='sum')
        assert not (score1 < score2)  # sum of score1 is -6, sum of score2 is -15

    def test_comparison_with_zero(self):
        """Test comparisons involving zero."""
        score1 = VectorScore([0, 0, 0], comparison_mode='sum')
        score2 = VectorScore([1, 1, 1], comparison_mode='sum')
        assert score1 < score2

    def test_multidimensional_array_flattened(self):
        """Test that multidimensional arrays are handled."""
        arr = np.array([[1, 2], [3, 4]])
        score = VectorScore(arr)
        # np.array() should reshape it
        assert score.vector.shape == (2, 2)


class TestVectorScoreComparisonModes:
    """Test all comparison modes systematically."""

    def test_all_modes_with_same_data(self):
        """Test that all modes can be initialized with the same data."""
        data = [1, 2, 3]
        modes = ['sequence', 'sum', 'any', 'all', 'random']

        for mode in modes:
            score = VectorScore(data, comparison_mode=mode)
            assert np.array_equal(score.vector, np.array(data))
            assert score.comparison_mode == mode

    def test_sequence_mode_comprehensive(self):
        """Comprehensive test for sequence mode."""
        # Test various scenarios
        test_cases = [
            ([1, 2, 3], [2, 1, 4], True),   # First element decides
            ([2, 1, 3], [2, 2, 4], True),   # Second element decides
            ([2, 2, 3], [2, 2, 4], True),   # Third element decides
            ([2, 2, 4], [2, 2, 3], False),  # Third element larger
            ([3, 1, 1], [2, 9, 9], False),  # First element larger
        ]

        for vec1, vec2, expected in test_cases:
            score1 = VectorScore(vec1, comparison_mode='sequence')
            score2 = VectorScore(vec2, comparison_mode='sequence')
            assert (score1 < score2) == expected

    def test_any_mode_edge_cases(self):
        """Test edge cases for 'any' mode."""
        # All equal
        score1 = VectorScore([5, 5, 5], comparison_mode='any')
        score2 = VectorScore([5, 5, 5], comparison_mode='any')
        assert not (score1 < score2)

        # One smaller in first position
        score1 = VectorScore([4, 5, 5], comparison_mode='any')
        score2 = VectorScore([5, 5, 5], comparison_mode='any')
        assert score1 < score2

    def test_all_mode_edge_cases(self):
        """Test edge cases for 'all' mode."""
        # All strictly smaller
        score1 = VectorScore([1, 2, 3], comparison_mode='all')
        score2 = VectorScore([2, 3, 4], comparison_mode='all')
        assert score1 < score2

        # One equal, rest smaller
        score1 = VectorScore([1, 2, 3], comparison_mode='all')
        score2 = VectorScore([2, 2, 4], comparison_mode='all')
        assert not (score1 < score2)
