import numpy as np

class VectorScore:
    """ A class to represent a (vector) score with customizable comparison modes. """

    def __init__(self, vector: np.array, comparison_mode: str = 'sequence'):
        self.vector = np.array(vector)
        self.comparison_mode = comparison_mode

    def __lt__(self, other):
        assert isinstance(other, VectorScore)
        assert self.vector.shape == other.vector.shape
        assert self.comparison_mode == other.comparison_mode

        if self.comparison_mode == 'sequence':
            for a, b in zip(self.vector, other.vector):
                if a < b:
                    return True
                elif a > b:
                    return False
        elif self.comparison_mode == 'sum':
            return np.sum(self.vector) < np.sum(other.vector)
        elif self.comparison_mode == 'any':
            return any(self.vector < other.vector)
        elif self.comparison_mode == 'all':
            return all(self.vector < other.vector)
        elif self.comparison_mode == 'random':
            smaller = any(self.vector < other.vector)
            larger = any(self.vector > other.vector)
            if smaller and larger:
                # randomly decide
                print('randomly decide')
                import random
                return random.choice([True, False])
            return smaller

    def __le__(self, other):
        return self < other or self == other

    def __eq__(self, other):
        assert isinstance(other, VectorScore)
        return np.array_equal(self.vector, other.vector)

    def __neg__(self):
        return VectorScore(-self.vector, self.comparison_mode)