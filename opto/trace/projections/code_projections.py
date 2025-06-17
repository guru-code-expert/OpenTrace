
from opto.trace.projections import Projection

class BlackCodeFormatter(Projection):
    # This requires the `black` package to be installed.
    
    def project(self, x: str) -> str:
        # importing here to avoid necessary dependencies on black
        # use black formatter for code reformatting
        from black import format_str, FileMode
        if type(x) == str and 'def' in x:
            x = format_str(x, mode=FileMode())
        return x

class DocstringProjection(Projection):
    """
    Projection that formats docstrings.
    """
    def __init__(self, docstring: str):
        self.docstring = docstring    

    def project(self, x: str) -> str:
        """ Replace the docstring in the code wit the stored docstring. """
        if type(x) == str and '"""' in x:
            # replace the docstring in the code with the stored docstring
            x = x.split('"""', 2)
            if len(x) > 2:
                x = f'{x[0]}"""{self.docstring}"""{x[2]}'
            else:
                x = f'{x[0]}"""{self.docstring}"""'
        return x