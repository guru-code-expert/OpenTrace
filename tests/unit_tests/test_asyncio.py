
import asyncio
import time
from opto import trace

@trace.bundle()
async def basic(a=0):
    await asyncio.sleep(1)
    return 'basic'

@trace.bundle()
async def error(a=0):
    raise ValueError('error')


@trace.bundle()
async def basic2(a=0, t=1):
    await asyncio.sleep(t)
    return 'basic'





def test_async():
    async def main():
        # single task
        a = trace.node('a')
        st = time.time()
        x = await basic(a)
        ed = time.time()
        print("Time taken: ", ed - st)
        print(type(x), x)
        assert type(x) == trace.nodes.MessageNode
        assert x == 'basic'
        assert a in x.parents
        assert len(x.parents) == 1

    async def main2():
        # multiple tasks
        a = trace.node('a')
        st = time.time()
        x, y, z = await asyncio.gather(basic(a), basic(a), basic(a))  # run in parallel
        ed = time.time()
        print("Time taken: ", ed - st)

        assert type(x) == trace.nodes.MessageNode
        assert x == 'basic'
        assert a in x.parents
        assert len(x.parents) == 1
        assert type(y) == trace.nodes.MessageNode
        assert y == 'basic'
        assert a in y.parents
        assert len(y.parents) == 1
        assert type(z) == trace.nodes.MessageNode
        assert z == 'basic'
        assert a in z.parents
        assert len(z.parents) == 1

    async def main3():
        # error handling
        a = trace.node('a')
        st = time.time()
        try:
            x = await error(a)
        except trace.ExecutionError as e:
            print(e)
            x = e
        ed = time.time()
        print("Time taken: ", ed - st)
        print(type(x), 'developer message:', x)
        assert isinstance(x, trace.ExecutionError)
        x = x.exception_node
        print(type(x), 'optimizer message:', x.data)
        assert isinstance(x, trace.nodes.MessageNode)
        assert a in x.parents
        assert len(x.parents) == 1

    async def main4():
        # multiple error handling
        a = trace.node('a')
        b = trace.node('b')
        c = trace.node('c')
        st = time.time()
        try:
            x, y, z = await asyncio.gather(error(a), error(b), error(c))  # run in parallel
        except trace.ExecutionError as e:
            # print(e)
            x = e  # This will catch the first error
            print(e.exception_node.parents)
        ed = time.time()
        print("Time taken: ", ed - st)
        print(type(x), 'developer message:', x)
        assert isinstance(x, trace.ExecutionError)
        x = x.exception_node
        print(type(x), 'optimizer message:', x.data)
        assert isinstance(x, trace.nodes.MessageNode)
        assert a in x.parents
        assert len(x.parents) == 1

    
    other_node = trace.node('other_node')

    async def main5():
        # competitive tasks
        st = time.time()
        parent_1 = trace.node('parent_1')
        parent_2 = trace.node('parent_2')
        parent_3 = trace.node('parent_3')

        x, y, z = await asyncio.gather(basic2(parent_1, 1), basic2(parent_2, 2), basic2(parent_3, 3))  # run in parallel
        ed = time.time()
        print("Time taken: ", ed - st)

        assert type(x) == trace.nodes.MessageNode
        assert x == 'basic'
        assert parent_1 in x.parents
        assert len(x.parents) == 2
        assert type(y) == trace.nodes.MessageNode
        assert y == 'basic'
        assert parent_2 in y.parents
        assert len(y.parents) == 2
        assert type(z) == trace.nodes.MessageNode
        assert z == 'basic'
        assert parent_3 in z.parents
        assert len(z.parents) == 2


    asyncio.run(main())
    asyncio.run(main2())
    asyncio.run(main3())
    asyncio.run(main4())
    asyncio.run(main5())