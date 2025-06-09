import asyncio

def by_run():
    async def main():
        await asyncio.sleep(1)
        cur_loop = asyncio.get_event_loop()
        print("cur loop id:", id(cur_loop))
        print("done")

    asyncio.run(main())  # 自动创建和关闭 event loop
    # loop2 = asyncio.get_event_loop()
    print("exit")


def by_loop():
    async def foo():
        print("start")
        await asyncio.sleep(1)
        print("end")
    
    async def foo2():
        print("start2")
        await asyncio.sleep(1)
        print("end2")
    loop0 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop0)
    loop1 = asyncio.get_event_loop()  # 获取事件循环对象
    loop1.run_until_complete(foo())   # 驱动协程执行
    loop2 = asyncio.get_event_loop()  # 获取事件循环对象
    loop2.run_until_complete(foo2())   # 驱动协程执行
    print(f"loop0 id: {id(loop0)}, loop1 id: {id(loop1)}, loop2 id: {id(loop2)}")

def add_task():
    async def foo():
        print("start")
        await asyncio.sleep(1)
        print("end")
    loop = asyncio.get_event_loop()
    loop.create_task(foo())
    loop.run_forever()

if __name__ == "__main__":
    by_loop()
    by_run()