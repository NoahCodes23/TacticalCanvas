---
name: starlette-pin-and-missing-web
description: Two boot-blocking gotchas for the TacticalCanvas core server (starlette version pin, missing web/ dir)
metadata:
  type: project
---

Two non-obvious things can stop the TacticalCanvas core (`python run.py` / `tc.py start`) from booting:

1. **starlette must stay `<0.51`.** The core pins `fastapi 0.128.0`, which requires `starlette>=0.40,<0.51`. Installing `fastmcp` (for [[mcp-server]]) upgrades starlette to 1.3.x, which breaks the server with `Router.__init__() got an unexpected keyword argument 'on_startup'`. Fix: `pip install "starlette>=0.40,<0.51"` (0.50.0 works, and fastmcp still runs fine on it). Watch for any future `pip install`/upgrade re-bumping it.

**Why:** one shared Python env serves both the core and the MCP tooling, and their starlette needs conflict.
**How to apply:** if the server dies with an `on_startup`/Router TypeError, downgrade starlette back under 0.51 before debugging anything else.

2. **The `web/` frontend is absent from this repo.** Only the Python backend is committed (`server/`, `vision/`). `server/main.py` mounts `StaticFiles(directory="web")` at import, so with no `web/` dir the app raises `RuntimeError: Directory 'web' does not exist` and never starts. An empty `web/` placeholder lets it boot for backend/WS testing, but `/dashboard` and `/projector` will 404 until the real frontend exists.
