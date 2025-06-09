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
    

class BlackCodeFormatter(Projection):
    # This requires the `black` package to be installed.
    
    def project(self, x: str) -> str:
        # importing here to avoid necessary dependencies on black
        # use black formatter for code reformatting
        from black import format_str, FileMode
        if type(x) == str and 'def' in x:
            x = format_str(x, mode=FileMode())
        return x
