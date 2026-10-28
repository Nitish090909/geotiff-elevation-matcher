# GeoTIFF Elevation Matcher

A Python-based geospatial application for working with GeoTIFF elevation data, sampling elevation values at geographic coordinates, and visualizing elevation results in 2D and 3D.

## 🚀 Live Demo

**[Open GeoTIFF Elevation Matcher](https://geotiff-elevation-matcher-m2m3iphltodwatqdknraeh.streamlit.app/)**

## ✨ Features

- Load and process GeoTIFF elevation data.
- Work with geographic coordinates (latitude and longitude).
- Extract elevation values at specified locations.
- Compare elevation values and calculate differences.
- Visualize elevation data using interactive Plotly charts.
- Explore elevation surfaces in 2D and 3D.

## 🛠️ Technologies Used

- **Python** — Core programming language
- **Streamlit** — Interactive web application
- **Rasterio** — GeoTIFF and raster data processing
- **NumPy** — Numerical computation
- **Pandas** — Tabular data processing
- **Plotly** — Interactive data visualization
- **SciPy** — Scientific computing and interpolation
- **PyProj** — Coordinate reference system transformations

## 💻 Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Nitish090909/geotiff-elevation-matcher.git
cd geotiff-elevation-matcher
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

Navigate to the directory containing `app.py`, if necessary, and run:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## 📂 Project Purpose

This project helps explore workflows involving elevation rasters, coordinate-based elevation sampling, interpolation, and elevation-difference analysis.

## 📄 License

A license has not yet been specified for this project.
