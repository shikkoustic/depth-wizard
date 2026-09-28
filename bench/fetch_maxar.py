"""Crop a GeoTIFF from Maxar Open Data (30–60 cm satellite imagery, CC-BY-NC-4.0) for an India demo.

Maxar releases imagery for disaster events; "India-Floods-Oct-2023" covers Sikkim (Himalayan terrain with
towns) at 0.37–0.58 m, which is close to the 0.66 m the height model was trained at — unlike Sentinel-2 (10 m),
where individual buildings are invisible.

  python bench/fetch_maxar.py --list
  python bench/fetch_maxar.py --name sikkim_town --bbox 88.36 27.16 88.38 27.18
Attribution required: "© 2023 Maxar Open Data Program (CC BY-NC 4.0)".
"""
import argparse, json, os, urllib.request
import numpy as np, rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

EVENT = "https://maxar-opendata.s3.amazonaws.com/events/India-Floods-Oct-2023"
OUT = os.path.join(os.path.dirname(__file__), "..", "runs", "demos", "india")


def get(u):
    return json.load(urllib.request.urlopen(u, timeout=60))


def items():
    col = get(f"{EVENT}/collection.json")
    out = []
    for a in [l["href"].replace("./", f"{EVENT}/") for l in col["links"] if l["rel"] == "child"]:
        c = get(a)
        for l in [x for x in c["links"] if x["rel"] == "item"]:
            u = (a.rsplit("/", 1)[0] + "/" + l["href"]).replace("/acquisition_collections/../", "/")
            try:
                it = get(u)
            except Exception:
                continue
            out.append(dict(bbox=it["bbox"], gsd=it["properties"].get("gsd"), date=it["properties"].get("datetime", "")[:10],
                            visual=it["assets"]["visual"]["href"].replace("./", u.rsplit("/", 1)[0] + "/"), id=it["id"]))
    return out


def crop(item, bbox, path):
    with rasterio.open(f"/vsicurl/{item['visual']}") as src:
        b = transform_bounds("EPSG:4326", src.crs, *bbox, densify_pts=21)
        win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read([1, 2, 3], window=win)
        prof = src.profile.copy()
        prof.update(driver="GTiff", width=data.shape[2], height=data.shape[1], count=3,
                    transform=src.window_transform(win), compress="deflate", tiled=True, photometric="rgb")
        for k in ("interleave",):
            prof.pop(k, None)
        with rasterio.open(path, "w", **prof) as dst:
            dst.write(data)
        return dict(item=item["id"], gsd=item["gsd"], date=item["date"], crs=str(src.crs), shape=list(data.shape),
                    filled=float((data.max(0) > 0).mean()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--name"); ap.add_argument("--bbox", nargs=4, type=float)
    a = ap.parse_args()
    its = items()
    if a.list:
        for it in sorted(its, key=lambda i: (i["bbox"][1], i["bbox"][0])):
            print([round(x, 3) for x in it["bbox"]], it["gsd"], it["date"], it["id"])
        raise SystemExit
    os.makedirs(OUT, exist_ok=True)
    x0, y0, x1, y1 = a.bbox
    cand = [i for i in its if i["bbox"][0] <= x0 and i["bbox"][1] <= y0 and i["bbox"][2] >= x1 and i["bbox"][3] >= y1]
    if not cand:
        raise SystemExit("no Maxar item covers this bbox; run --list")
    info = crop(sorted(cand, key=lambda i: i["gsd"])[0], a.bbox, f"{OUT}/{a.name}_maxar_rgb.tif")
    json.dump(dict(name=a.name, bbox=a.bbox, source="Maxar Open Data Program (CC BY-NC 4.0), event India-Floods-Oct-2023", **info),
              open(f"{OUT}/{a.name}_maxar.json", "w"), indent=1)
    print(json.dumps(info))
