from opto.trace.projections import BlackCodeFormatter

def test_black_code_formatter():
    code = """
def example_function():
                print("Hello, World!")


                print("This is a test function.")


                
    """     
    projection = BlackCodeFormatter()
    formatted_code = projection.project(code)
    assert formatted_code == 'def example_function():\n    print("Hello, World!")\n\n    print("This is a test function.")\n'
