import asyncio
import websockets
import datetime

async def consumer(message):
    print("consumer: ", message)
    # TODO: Pass message on

async def producer(message):
    # TODO: Put relevant messages in here, if at all (we may ask our questions via http instead of ws)
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    return (now)

async def consumer_handler(websocket):
    while True:
        async for message in websocket:
            #print("this went in glob_message: {}".format(message))
            await consumer(message)    #global glob_message

async def producer_handler(websocket):
    #global glob_message
    while True:
        message = await producer()
        print("producer: ", message)
        #await websocket.send(message)
        #await asyncio.sleep(5.0)

async def handler(websocket):
    consumer_task = asyncio.create_task(consumer_handler(websocket))
    producer_task = asyncio.create_task(producer_handler(websocket))
    done, pending = await asyncio.wait(
        [consumer_task, producer_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()

async def main():
    async with websockets.connect("ws://localhost:20598/register/c") as websocket:
        await handler(websocket)
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())