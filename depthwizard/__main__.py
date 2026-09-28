"""DepthWizard command line.

  depthwizard                         start the local app (viewer + upload API) and open the browser
  depthwizard process IMAGE -o OUT    run the elevation module only: IMAGE (GeoTIFF/PNG/JPG) → DSM GeoTIFFs + scene
"""
import argparse, json, os, sys, threading, webbrowser


def _weights(path):
    if path:
        os.environ["DEPTHWIZARD_WEIGHTS"] = os.path.abspath(path)


def serve(argv):
    ap = argparse.ArgumentParser(prog="depthwizard", description="Single-view satellite image → DSM → 3D flythrough")
    ap.add_argument("--data", default="depthwizard_data", help="folder for uploads and processed scenes")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--weights", help="height-model checkpoint (default: downloaded once to ~/.cache/depthwizard)")
    a = ap.parse_args(argv)
    _weights(a.weights)
    import uvicorn
    from .server import create_app
    app = create_app(a.data)
    if not a.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(f"http://{a.host}:{a.port}/")).start()
    print(f"DepthWizard running at http://{a.host}:{a.port}/  (data: {os.path.abspath(a.data)})")
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


def process_cmd(argv):
    ap = argparse.ArgumentParser(prog="depthwizard process", description="Image → DSM (GeoTIFF) without the UI")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", required=True, help="output folder")
    ap.add_argument("--reference", help="reference DSM GeoTIFF to score against")
    ap.add_argument("--dem", help="terrain DEM to use instead of FABDEM (offline use)")
    ap.add_argument("--gcps", help="CSV of ground control points x,y,z in the image CRS")
    ap.add_argument("--gsd", type=float, help="pixel size in metres for PNG/JPG")
    ap.add_argument("--tta", type=int, help="test-time augmentation variants (1-8)")
    ap.add_argument("--weights")
    a = ap.parse_args(argv)
    _weights(a.weights)
    from .pipeline import process
    rep = process(a.image, a.out, reference=a.reference, dem=a.dem, gcps=a.gcps, gsd=a.gsd, n_tta=a.tta,
                  progress=lambda m: print("  " + m, file=sys.stderr))
    print(json.dumps(rep, indent=1))


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "process":
        process_cmd(argv[1:])
    else:
        serve(argv)


if __name__ == "__main__":
    main()
