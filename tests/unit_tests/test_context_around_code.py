from opto import trace
from opto.optimizers import OptoPrime

def test_context_around_code():
    # This tests running code with extra import and function definitions
    @trace.bundle(trainable=True)
    def fun(x):
        hidden()
        time.sleep(x)

    fun.parameter._data  = \
    """
import time
""" + \
    fun.parameter._data + \
    """
def hidden():
    print('This is a hidden fun')
"""
    print(fun.parameter._data)
    fun(1)

    @trace.bundle(trainable=True)
    def fun2(x):
        hidden()
        time.sleep(x)


    fun2.parameter._data  = \
    """
import time

def hidden():
    print('This is a hidden fun')
""" + \
    fun2.parameter._data

    fun2(1)
