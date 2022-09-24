#!/usr/bin/env python
import sys
import json
import logging
import requests
import asyncio
import websockets
import datetime
import ast

from PkFkFinder import PkFkFinder

async def pk_fk_relationship_finder(data):
    datamodel = data["datamodel"]
    print("datamodel:", datamodel)
    if datamodel != "RELATIONAL":
        logging.error('Datamodel is not relational. Not implemented!')
        raise ValueError('Datamodel is not relational. Not implemented!')
    else:
        pk_fk_finder = PkFkFinder("http://127.0.0.1", "20598", data["tables"], 20)
        sampled_pk_fk_relationships = pk_fk_finder.above_similarity_threshold_pk_comparison()
        validated_pk_fk_relationships = pk_fk_finder.validate_pk_fk_relationships(sampled_pk_fk_relationships)
        print("Validated pk_fk relationships", validated_pk_fk_relationships)

async def consumer(message):
    print("consumer: ", message)
    d_message = ast.literal_eval(message)
    if d_message["topic"] == "namespaceInfo":
        await pk_fk_relationship_finder(ast.literal_eval(d_message["message"]))
    #except Exception as e:
     #   print("Evaluation of message failed: ", e)

async def producer():
    #now = datetime.datetime.utcnow().isoformat() + 'Z'
    #return (now)
    pass

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
    try:
        async with websockets.connect("ws://localhost:20598/register") as websocket:
            await handler(websocket)
            await asyncio.Future()  # run forever
    except ConnectionRefusedError as e:
        logging.error("Polypheny websocket server not running or responding")
        print("Polypheny websocket server not running or responding", e)

    # Define a pipeline of steps to be performed (later to be optimized/determined dynamically due to user inputs, etc.)
    pipeline = ['pkfkfinder']

if __name__ == "__main__":
    logging.getLogger("asyncio").setLevel(logging.INFO)
    logging.Formatter('%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    # TODO: Set logging file name (https://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a)
    logging.info('Started schema integration Python part (main.py)')

    asyncio.run(main())

    logging.info('Closing Polypheny connection. Exiting schema integration Python part.')