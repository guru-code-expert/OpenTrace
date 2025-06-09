from opto.trace.nodes import ParameterNode


class Projection:
    """
    Abstract base class for projection methods.
    """

    def __init__(self, *args, **kwargs):
        pass

    def project(self, x: ParameterNode) -> ParameterNode:
        """
        Project the parameter node `x` onto the feasible set.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    