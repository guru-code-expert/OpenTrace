import os
import pickle
import copy
import inspect
import textwrap
from opto.trace.containers import ParameterContainer
from opto.trace.nodes import ParameterNode


def model(cls):
    """
    Wrap a class with this decorator. This helps collect parameters for the optimizer. This decorated class cannot be pickled.
    """

    class ModelWrapper(cls, Module):
        def model_dump(self, filename):
            methods = [
                method for name, method in cls.__dict__.items()
                if inspect.isfunction(method)
            ]

            with open(filename, "w") as f:
                f.write(f"class {cls.__name__}:\n")

                for i, method in enumerate(methods):
                    source = inspect.getsource(method)
                    source = textwrap.dedent(source)
                    indented = textwrap.indent(source, "    ")
                    f.write(indented)
                    if i < len(methods) - 1:
                        f.write("\n")  # only one newline between methods

    return ModelWrapper


class Module(ParameterContainer):
    """Module is a ParameterContainer which has a forward method."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def save(self, file_name: str):
        """Save the parameters of the model to a pickle file."""
        # detect if the directory exists
        directory = os.path.dirname(file_name)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(copy.deepcopy(self.parameters_dict()), f)

    def load(self, file_name):
        """Load the parameters of the model from a pickle file."""
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)
        self._set(loaded_data)

    def _set(self, new_parameters):
        """Set the parameters of the model from a dictionary.
        new_parameters is a ParamterContainer or a parameter dict.
        """
        assert isinstance(new_parameters, (dict, ParameterContainer))
        if isinstance(new_parameters, ParameterContainer):
            new_parameters_dict = new_parameters.parameters_dict()
        else:
            new_parameters_dict = new_parameters  # dictionary

        parameters_dict = self.parameters_dict()

        assert all(
            k in new_parameters_dict for k in parameters_dict.keys()
        ), """ Not all model parameters are in the new parameters dictionary. """

        for k, v in new_parameters_dict.items():
            if k in parameters_dict:  # if the parameter exists
                assert isinstance(v, (ParameterNode, ParameterContainer))
                parameters_dict[k]._set(v)
            else:  # if the parameter does not exist
                assert k not in self.__dict__
                setattr(self, k, v)