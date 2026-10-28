# GeoTIFF Elevation Matcher with Interactive 3D Terrain

## Features
- Interactive 2D elevation map
- Interactive 3D terrain surface (rotate, zoom, hover)
- Optional electrode/location CSV overlay on the 3D terrain
- Sample GeoTIFF elevations at CSV longitude/latitude coordinates using bilinear or nearest-pixel sampling
- Export matched values to CSV or Excel
- GeoTIFF metadata and band scale/offset display
- Stale-result detection when input data or sampling settings change

## Run on Windows
Open PowerShell in the folder containing `app.py`, then run:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload a single-band elevation GeoTIFF. In the **3D Terrain** tab, optionally upload the electrode CSV to overlay its points. The longitude/latitude columns should be geographic coordinates in EPSG:4326. The app transforms them into the GeoTIFF CRS.

The 3D surface is downsampled for display performance; sampling and exports continue to use the original raster. Confirm vertical units and datum from the data provider before treating values as metres above sea level.
