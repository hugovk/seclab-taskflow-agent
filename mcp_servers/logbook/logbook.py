import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='logs/mcp_logbook.log',
    filemode='a'
)
#from mcp.server.fastmcp import FastMCP
from fastmcp import FastMCP # move to FastMCP 2.0
import json
from pathlib import Path
import os

mcp = FastMCP("Logbook")

LOG = {}

LOGBOOK = Path(__file__).parent.resolve() / Path(os.getenv('LOGBOOK_STATE_DIR', default='./')) / Path("logbook.json")


def ensure_log():
    global LOG
    global LOGBOOK
    try:
        LOGBOOK.parent.mkdir(exist_ok=True, parents=True)
        with open(LOGBOOK, 'x') as logbook:
            logbook.write(json.dumps(LOG, indent=2))
            logbook.flush()
    except FileExistsError:
        pass


def deflate_log():
    ensure_log()
    global LOG
    global LOGBOOK
    with open(LOGBOOK, 'w') as logbook:
        logbook.write(json.dumps(LOG, indent=2))
        logbook.flush()


def inflate_log():
    ensure_log()
    global LOG
    global LOGBOOK
    with open(LOGBOOK, 'r') as logbook:
        LOG = json.loads(logbook.read())


def with_log(f):
    def wrapper(*args, **kwargs):
        inflate_log()
        ret = f(*args, **kwargs)
        deflate_log()
        return ret
    return wrapper


@mcp.tool()
def logbook_write(entry: str, key: str) -> str:
    """Appends a logbook entry to an identifying key. This lets you write to your logbook."""
    @with_log
    def _logbook_write(entry: str, key: str) -> str:
        global LOG
        LOG[key] = LOG.get(key, []) + [entry]
        return f"Stored logbook entry for `{key}`"
    return _logbook_write(entry, key)


@mcp.tool()
def logbook_read(key: str) -> str:
    """Reads the entries stored for an identifying key. This lets you read from your logbook."""
    @with_log
    def _logbook_read(key: str) -> str:
        global LOG
        return json.dumps(LOG.get(key, []), indent=2)
    return _logbook_read(key)


@mcp.tool()
def logbook_erase(key: str) -> str:
    """Erase the entries stored for an identifying key. This lets you erase in your logbook."""
    @with_log
    def _logbook_erase(key) -> str:
        global LOG
        LOG[key] = []
        return f"Erased logbook entries stored for `{key}`"
    return _logbook_erase(key)


if __name__ == "__main__":
    mcp.run(show_banner=False)
