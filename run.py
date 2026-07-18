import multiprocessing
import os

import uvicorn

if __name__ == "__main__":
    multiprocessing.freeze_support()  # required for spawn on Windows
    from server.main import app

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("TC_PORT", "8000")),
        log_level="warning",
        ws_per_message_deflate=False,
    )
