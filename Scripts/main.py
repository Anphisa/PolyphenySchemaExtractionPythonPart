#!/usr/bin/env python
import sys
import json
import logging
import requests
import asyncio
import websockets
import datetime
import ast

async def test(data):
    datamodel = data["datamodel"]
    print("datamodel:", datamodel)
    if datamodel != "RELATIONAL":
        logging.error('Datamodel is not relational. Not implemented!')
        raise ValueError('Datamodel is not relational. Not implemented!')
        sys.exit('Fatal schema extractor error')

    for i in data["tables"]:
        print("table name:", i["tableName"] + ".", "number of columns:", len(i["columnNames"]))

        # Execute a Polypheny query
        URL = "http://127.0.0.1:20598/query"
        # defining a params dict for the parameters to be sent to the API
        PARAMS = {'querylanguage': 'SQL', 'query': "SELECT * FROM " + i["tableName"]}
        # sending get request and saving the response as response object
        r = requests.get(url=URL, params=PARAMS)
        print("Result Set: ", r)

async def consumer(message):
    print("consumer: ", message)
    d_message = ast.literal_eval(message)
    if d_message["topic"] == "namespaceInfo":
        await test(ast.literal_eval(d_message["message"]))
    #except Exception as e:
     #   print("Evaluation of message failed: ", e)

async def producer():
    # TODO: Put relevant messages in here, if at all (we may ask our questions via http instead of ws)
    # Or use as ping?
    now = datetime.datetime.utcnow().isoformat() + 'Z'
    return (now)

async def consumer_handler(websocket):
    while True:
        async for message in websocket:
            await consumer(message)    #global glob_message

async def producer_handler(websocket):
    #global glob_message
    while True:
        message = await producer()
        print("producer: ", message)
        await websocket.send(message)
        await asyncio.sleep(5.0)

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
        # Connect to Polypheny: websocket connection
    # Here we get user input (i.e. call to run Python script)
    async with websockets.connect("ws://localhost:20598/register/c") as websocket:
        await handler(websocket)
        await asyncio.Future()  # run forever

    # Define a pipeline of steps to be performed (later to be optimized/determined dynamically due to user inputs, etc.)
    #pipeline = ['pkfkfinder']

if __name__ == "__main__":
    #logging.basicConfig(filename='PolyphenySchemaIntegrationPython.log',
    #                    level=logging.INFO,
    #                    format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.info('Started schema integration Python part (main.py)')

    asyncio.run(main())

    #logging.info('Closing Polypheny connection. Exiting schema integration Python part.')