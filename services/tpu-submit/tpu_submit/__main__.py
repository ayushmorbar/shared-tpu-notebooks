"""python -m tpu_submit"""

from __future__ import annotations

import asyncio
import logging
import signal
from concurrent.futures import ThreadPoolExecutor

from .app import make_app, start_background
from .config import Settings
from .core import Core
from .hubauth import HubAuth


def build_store(settings: Settings):
    from .store import GCSStore, LocalStore

    if settings.store_dir:
        return LocalStore(settings.store_dir)
    return GCSStore(settings.student_bucket)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    from .cluster import KubeCluster

    core = Core(settings, build_store(settings), KubeCluster(settings.namespace))
    auth = HubAuth(settings.hub_api_url, settings.service_name, settings.require_access_scope)
    executor = ThreadPoolExecutor(max_workers=32)
    app = make_app(core, auth, settings, executor)
    server = app.listen(settings.port, max_buffer_size=64 * 1024 * 1024, idle_connection_timeout=900)
    start_background(core, settings, executor)
    logging.getLogger("tpu_submit").info("listening on :%d", settings.port)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    server.stop()
    await server.close_all_connections()


if __name__ == "__main__":
    asyncio.run(main())
