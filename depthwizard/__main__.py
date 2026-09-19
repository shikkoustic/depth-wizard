"""`python -m depthwizard` / `depthwizard`: start the local server and open the viewer."""
import argparse, threading, webbrowser


def main():
    ap = argparse.ArgumentParser(prog="depthwizard", description="Single-view satellite image → DSM → 3D flythrough")
    ap.add_argument("--data", default="depthwizard_data", help="folder for uploads and processed scenes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    import uvicorn
    from .server import create_app
    app = create_app(a.data)
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host}:{a.port}/")).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
