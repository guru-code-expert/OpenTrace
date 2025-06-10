from opto.trace.projections import BlackCodeFormatter, DocstringProjection

def test_black_code_formatter():
    code = """
def example_function():
                print("Hello, World!")


                print("This is a test function.")


                
    """     
    projection = BlackCodeFormatter()
    formatted_code = projection.project(code)
    assert formatted_code == 'def example_function():\n    print("Hello, World!")\n\n    print("This is a test function.")\n'


def test_docstring_projection():
    code = """
def example_function():
    \"\"\"This is an example function.\"\"\"
    print("Hello, World!")
    """
    docstring = "This is a new docstring."
    projection = DocstringProjection(docstring)
    formatted_code = projection.project(code)
    
    new_code = """
def example_function():
    \"\"\"This is a new docstring.\"\"\"
    print("Hello, World!")
    """

    assert formatted_code == new_code

    # assert '"""This is a new docstring."""' in formatted_code    
    # assert 'print("Hello, World!")' in formatted_code