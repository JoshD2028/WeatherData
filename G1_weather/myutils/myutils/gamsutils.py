import gams.transfer as gt
import time
from contextlib import contextmanager


@contextmanager
def timer(description: str):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print(f"{description} took {elapsed_time} seconds")


def read_gdx(ws, gdxfile: str):
    print(f'Starting to read the {gdxfile}')
    with timer(f"Reading {gdxfile}"):
        rdgdx = gt.Container(gdxfile, ws.system_directory)
    return rdgdx.data
