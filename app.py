import io
import math
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import plotly.graph_objects as go
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from rasterio.warp import transform as transform_coords, reproject, Resampling, transform_bounds
from rasterio.transform import from_bounds
from affine import Affine


def dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with mixed object columns safe for Arrow.

    The source dataframe is not modified, so numeric calculations and exports
    continue using the original values.
    """
    display_df = df.copy()
    for column in display_df.columns:
        if display_df[column].dtype == "object":
            non_null_types = {type(value) for value in display_df[column].dropna()}
            if len(non_null_types) > 1:
                display_df[column] = display_df[column].map(
                    lambda value: "" if pd.isna(value) else str(value)
                ).astype("string")
    return display_df


st.set_page_config(page_title="GeoTIFF Elevation Viewer", page_icon="🗺️", layout="wide")
st.title("🗺️ GeoTIFF Elevation Viewer")
st.caption("View a GeoTIFF and sample its elevation values at latitude/longitude points from a CSV.")

@st.cache_data(show_spinner=False)
def read_raster(file_bytes):
    with rasterio.MemoryFile(file_bytes) as mem:
        with mem.open() as src:
            band = src.read(1, masked=True)
            data = np.asarray(band.astype(np.float64).filled(np.nan), dtype=np.float64)
            scale = float(src.scales[0]) if src.scales else 1.0
            offset = float(src.offsets[0]) if src.offsets else 0.0
            # Convert stored samples to physical values if the GeoTIFF defines scale/offset.
            data = data * scale + offset
            return {
                "data": data,
                "scale": scale,
                "offset": offset,
                "transform": src.transform,
                "crs": src.crs.to_string() if src.crs else None,
                "bounds": tuple(src.bounds),
                "width": src.width,
                "height": src.height,
                "count": src.count,
                "dtype": str(src.dtypes[0]),
                "nodata": src.nodata,
                "resolution": tuple(src.res),
                "driver": src.driver,
                "units": getattr(src, "units", None),
                "tags": src.tags(),
            }

def auto_column(columns, candidates):
    lower = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    for c in columns:
        lc = str(c).strip().lower()
        if any(candidate in lc for candidate in candidates):
            return c
    return columns[0] if len(columns) else None

def transform_lonlat_to_raster(crs, longitude, latitude):
    if crs is None:
        raise ValueError("The GeoTIFF has no CRS. Assign the correct CRS to the raster before sampling.")
    xs, ys = transform_coords("EPSG:4326", crs, longitude, latitude)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

def sample_dem(data, affine, xs, ys):
    """Return nearest-pixel and bilinear values at map coordinates."""
    inverse = ~affine
    n = len(xs)
    nearest = np.full(n, np.nan, dtype=float)
    bilinear = np.full(n, np.nan, dtype=float)
    rows_center = np.full(n, np.nan, dtype=float)
    cols_center = np.full(n, np.nan, dtype=float)
    inside = np.zeros(n, dtype=bool)

    h, w = data.shape
    for i, (x, y) in enumerate(zip(xs, ys)):
        col_corner, row_corner = inverse * (float(x), float(y))
        # Affine inverse returns pixel-corner coordinates; subtract 0.5 for pixel-center index coordinates.
        col = col_corner - 0.5
        row = row_corner - 0.5
        rows_center[i], cols_center[i] = row, col
        # Extent check uses pixel-corner coordinates.
        inside[i] = (0 <= col_corner < w) and (0 <= row_corner < h)

        rn, cn = int(math.floor(row + 0.5)), int(math.floor(col + 0.5))
        if 0 <= rn < h and 0 <= cn < w and np.isfinite(data[rn, cn]):
            nearest[i] = data[rn, cn]

        r0, c0 = int(math.floor(row)), int(math.floor(col))
        dr, dc = row - r0, col - c0
        if 0 <= r0 < h - 1 and 0 <= c0 < w - 1:
            four = np.array([
                data[r0, c0], data[r0, c0 + 1],
                data[r0 + 1, c0], data[r0 + 1, c0 + 1]
            ], dtype=float)
            if np.isfinite(four).all():
                v00, v01, v10, v11 = four
                bilinear[i] = (
                    v00 * (1 - dr) * (1 - dc)
                    + v01 * (1 - dr) * dc
                    + v10 * dr * (1 - dc)
                    + v11 * dr * dc
                )
    return nearest, bilinear, inside, rows_center, cols_center

def parse_kml_surveys(kml_bytes):
    """Read KML Point placemarks and pair Start/End points within each Folder."""
    root = ET.fromstring(kml_bytes)
    def local(tag):
        return tag.split("}")[-1] if "}" in tag else tag
    def child_text(element, child_name, default=""):
        for child in element:
            if local(child.tag) == child_name and child.text:
                return child.text.strip()
        return default
    records = []
    def walk(element, folders=None):
        folders = folders or []
        for child in list(element):
            tag = local(child.tag)
            if tag == "Folder":
                folder_name = child_text(child, "name", f"Folder {len(folders)+1}")
                walk(child, folders + [folder_name])
            elif tag == "Placemark":
                name = child_text(child, "name", "Point")
                coord_text = None
                for node in child.iter():
                    if local(node.tag) == "Point":
                        for sub in node.iter():
                            if local(sub.tag) == "coordinates" and sub.text:
                                coord_text = sub.text.strip()
                                break
                    if coord_text:
                        break
                if coord_text:
                    parts = coord_text.split(",")
                    if len(parts) >= 2:
                        try:
                            lon, lat = float(parts[0]), float(parts[1])
                            alt = float(parts[2]) if len(parts) >= 3 and parts[2].strip() else np.nan
                            if -180 <= lon <= 180 and -90 <= lat <= 90:
                                records.append({"survey": folders[-1] if folders else "KML Survey", "point_name": name,
                                                "longitude": lon, "latitude": lat, "kml_altitude": alt})
                        except ValueError:
                            pass
            else:
                # Traverse Document and other containers, but avoid duplicate folder recursion.
                if tag not in ("Point", "LineString", "Polygon", "MultiGeometry"):
                    walk(child, folders)
    walk(root, [])
    if not records:
        raise ValueError("No KML Point placemarks with valid longitude/latitude coordinates were found.")
    points = pd.DataFrame(records)
    lines = []
    for survey, group in points.groupby("survey", sort=False):
        # Match Start/End labels, case-insensitively; tolerate names such as Start.
        start_rows = group[group["point_name"].str.strip().str.lower().str.rstrip(".").eq("start")]
        end_rows = group[group["point_name"].str.strip().str.lower().str.rstrip(".").eq("end")]
        if len(start_rows) and len(end_rows):
            a, b = start_rows.iloc[0], end_rows.iloc[0]
            lines.append({"survey": survey, "start_lon": a.longitude, "start_lat": a.latitude,
                          "start_kml_altitude": a.kml_altitude, "end_lon": b.longitude,
                          "end_lat": b.latitude, "end_kml_altitude": b.kml_altitude})
    return points, pd.DataFrame(lines)


def make_line_samples(lines_df, raster, data, samples_per_line):
    """Densify each start/end segment in lon/lat and sample the DEM."""
    profile_rows, summary_rows = [], []
    for _, line in lines_df.iterrows():
        n = int(samples_per_line)
        t = np.linspace(0.0, 1.0, n)
        # For short local survey lines, linear interpolation in lon/lat is adequate for sampling.
        lons = line.start_lon + (line.end_lon - line.start_lon) * t
        lats = line.start_lat + (line.end_lat - line.start_lat) * t
        xs, ys = transform_lonlat_to_raster(raster["crs"], lons.tolist(), lats.tolist())
        nearest, bilinear, inside, _, _ = sample_dem(data, raster["transform"], xs, ys)
        sampled = np.where(np.isfinite(bilinear), bilinear, nearest)
        # Approximate ground distance using a local haversine calculation.
        r_earth = 6371008.8
        lat1, lat2 = np.radians(line.start_lat), np.radians(line.end_lat)
        dlat = lat2 - lat1
        dlon = np.radians(line.end_lon - line.start_lon)
        aa = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        length_m = 2*r_earth*np.arcsin(np.sqrt(np.clip(aa, 0, 1)))
        distances = t * length_m
        for i in range(n):
            profile_rows.append({"survey": line.survey, "distance_m": distances[i], "longitude": lons[i],
                                 "latitude": lats[i], "raster_x": xs[i], "raster_y": ys[i],
                                 "DEM_elevation": sampled[i], "inside_raster_extent": bool(inside[i]),
                                 "kml_altitude_interpolated": line.start_kml_altitude + (line.end_kml_altitude-line.start_kml_altitude)*t[i]
                                 if np.isfinite(line.start_kml_altitude) and np.isfinite(line.end_kml_altitude) else np.nan})
        start_sample = sampled[0] if len(sampled) else np.nan
        end_sample = sampled[-1] if len(sampled) else np.nan
        summary_rows.append({"survey": line.survey, "start_longitude": line.start_lon, "start_latitude": line.start_lat,
                             "end_longitude": line.end_lon, "end_latitude": line.end_lat,
                             "approx_length_m": length_m, "start_kml_altitude": line.start_kml_altitude,
                             "start_DEM_elevation": start_sample, "end_kml_altitude": line.end_kml_altitude,
                             "end_DEM_elevation": end_sample,
                             "start_DEM_minus_KML_altitude": start_sample-line.start_kml_altitude if np.isfinite(start_sample) and np.isfinite(line.start_kml_altitude) else np.nan,
                             "end_DEM_minus_KML_altitude": end_sample-line.end_kml_altitude if np.isfinite(end_sample) and np.isfinite(line.end_kml_altitude) else np.nan})
    return pd.DataFrame(profile_rows), pd.DataFrame(summary_rows)

def make_excel_bytes(summary_df, results_df):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        results_df.to_excel(writer, sheet_name="Elevation Matches", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = __import__("openpyxl").styles.Font(bold=True, color="FFFFFF")
                cell.fill = __import__("openpyxl").styles.PatternFill("solid", fgColor="1F4E78")
                cell.alignment = __import__("openpyxl").styles.Alignment(wrap_text=True)
            for col_cells in ws.columns:
                max_len = max((min(len(str(c.value or "")), 45) for c in col_cells[:min(ws.max_row, 250)]), default=12)
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 40)
    return buffer.getvalue()

tif_file = st.file_uploader("1. Upload elevation GeoTIFF (.tif / .tiff)", type=["tif", "tiff"], key="dem_upload")
if tif_file is None:
    st.info("Upload a DEM/DTM GeoTIFF to start. Band 1 should contain elevation values, not ordinary RGB satellite imagery.")
    st.stop()

try:
    raster = read_raster(tif_file.getvalue())
except Exception as exc:
    st.error(f"Could not read this raster: {exc}")
    st.stop()

data = raster["data"]
valid = data[np.isfinite(data)]
if valid.size == 0:
    st.error("Band 1 contains no valid pixels.")
    st.stop()
if raster["crs"] is None:
    st.error("This TIFF has no CRS. Sampling by latitude/longitude requires the correct CRS to be defined first.")
    st.stop()

unit = (raster["units"][0] if raster["units"] else None) or "source units"
m1, m2, m3, m4 = st.columns(4)
m1.metric("Raster dimensions", f'{raster["width"]:,} × {raster["height"]:,}')
m2.metric("Valid pixels", f"{valid.size:,}")
m3.metric("Minimum value", f"{np.nanmin(data):,.3f}")
m4.metric("Maximum value", f"{np.nanmax(data):,.3f}")
st.caption(f'CRS: {raster["crs"]} · Pixel resolution: {raster["resolution"]} · Band 1 units: {unit}')

tab_map, tab_3d, tab_match, tab_kml, tab_osm, tab_meta = st.tabs(["2D Elevation Map", "3D Terrain", "Match CSV Coordinates", "KML Survey Lines", "OpenStreetMap & Area", "Raster Metadata"])

with tab_map:
    max_dim = st.slider("Maximum display dimension", 250, 2000, 1000, 50)
    stride = max(1, math.ceil(max(data.shape) / max_dim))
    view = data[::stride, ::stride]
    fig = go.Figure(go.Heatmap(
        z=view, colorscale="Earth", connectgaps=False,
        colorbar=dict(title=f"Elevation<br>({unit})"),
        hovertemplate="Display row: %{y}<br>Display column: %{x}<br>Value: %{z:.3f}<extra></extra>"
    ))
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_title=f"Column (stride {stride})", yaxis_title="Row")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.caption("The map display is downsampled for performance; coordinate sampling uses the original full-resolution raster.")

with tab_3d:
    st.subheader("Interactive 3D terrain surface")
    st.write("Rotate with the mouse, zoom with the scroll wheel, and hover over the surface to inspect elevation values. Coordinates are shown in the GeoTIFF CRS.")
    max_3d_dim = st.slider("3D surface maximum dimension", 80, 500, 250, 10,
                          help="Lower this value if the 3D view is slow. Sampling/export still uses the full-resolution raster.")
    stride_3d = max(1, math.ceil(max(data.shape) / max_3d_dim))
    z_surface = data[::stride_3d, ::stride_3d]
    row_idx = np.arange(0, data.shape[0], stride_3d, dtype=float) + 0.5
    col_idx = np.arange(0, data.shape[1], stride_3d, dtype=float) + 0.5
    cols_grid, rows_grid = np.meshgrid(col_idx, row_idx)
    affine = raster["transform"]
    x_surface = affine.a * cols_grid + affine.b * rows_grid + affine.c
    y_surface = affine.d * cols_grid + affine.e * rows_grid + affine.f

    fig3d = go.Figure(data=[go.Surface(
        x=x_surface, y=y_surface, z=z_surface,
        colorscale="Earth", connectgaps=False,
        colorbar=dict(title=f"Elevation<br>({unit})"),
        customdata=np.stack([x_surface, y_surface], axis=-1),
        hovertemplate="X: %{customdata[0]:.3f}<br>Y: %{customdata[1]:.3f}<br>Elevation: %{z:.3f}<extra></extra>",
        name="Elevation surface"
    )])

    overlay_csv = st.file_uploader("Optional: upload electrode/location CSV to overlay points", type=["csv"], key="points_3d_upload")
    if overlay_csv is not None:
        try:
            overlay_df = pd.read_csv(overlay_csv)
            if overlay_df.empty:
                st.warning("The overlay CSV has no rows.")
            else:
                overlay_cols = list(overlay_df.columns)
                olon_default = auto_column(overlay_cols, ["longitude_deg", "longitude", "lon", "long", "x_lon"])
                olat_default = auto_column(overlay_cols, ["latitude_deg", "latitude", "lat", "y_lat"])
                oc1, oc2 = st.columns(2)
                olon = oc1.selectbox("3D overlay longitude", overlay_cols, index=overlay_cols.index(olon_default), key="3d_lon_col")
                olat = oc2.selectbox("3D overlay latitude", overlay_cols, index=overlay_cols.index(olat_default), key="3d_lat_col")
                if olon == olat:
                    st.warning("Choose different longitude and latitude columns for the overlay.")
                else:
                    lon_values = pd.to_numeric(overlay_df[olon], errors="coerce")
                    lat_values = pd.to_numeric(overlay_df[olat], errors="coerce")
                    good = lon_values.notna() & lat_values.notna() & lon_values.between(-180, 180) & lat_values.between(-90, 90)
                    if good.any():
                        ox, oy = transform_lonlat_to_raster(raster["crs"], lon_values[good].astype(float).tolist(), lat_values[good].astype(float).tolist())
                        nearest, bilinear, inside, _, _ = sample_dem(data, affine, ox, oy)
                        oz = np.where(np.isfinite(bilinear), bilinear, nearest)
                        keep = inside & np.isfinite(oz)
                        if keep.any():
                            hover_text = [str(v) for v in overlay_df.loc[good].index.to_list()]
                            hover_text = np.asarray(hover_text)[keep]
                            fig3d.add_trace(go.Scatter3d(
                                x=ox[keep], y=oy[keep], z=oz[keep], mode="markers",
                                marker=dict(size=4, color="red", line=dict(width=1, color="white")),
                                text=hover_text,
                                hovertemplate="CSV row index: %{text}<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Elevation: %{z:.3f}<extra></extra>",
                                name="Electrode points"
                            ))
                            st.caption(f"Overlay: {int(keep.sum())} points placed on the raster surface; {int((~keep).sum())} valid-coordinate points were outside the raster or over NoData.")
                        else:
                            st.warning("No overlay points could be placed inside valid raster pixels.")
                    else:
                        st.warning("No valid longitude/latitude pairs found in the CSV.")
        except Exception as exc:
            st.error(f"Could not prepare the 3D point overlay: {exc}")

    fig3d.update_layout(
        height=720, margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis_title=f"X ({raster['crs']})", yaxis_title=f"Y ({raster['crs']})",
            zaxis_title=f"Elevation ({unit})", aspectmode="auto",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.1))
        ), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig3d, width="stretch")
    st.caption(f"3D surface uses every {stride_3d} pixel(s) for display performance. The original raster remains unchanged; coordinate matching uses full-resolution data.")

with tab_match:
    st.subheader("Sample elevation at your electrode coordinates")
    st.write("Upload the electrode CSV. The app transforms longitude/latitude from EPSG:4326 into the GeoTIFF CRS, then samples the raster at each location.")
    csv_file = st.file_uploader("2. Upload electrode/location CSV", type=["csv"], key="points_upload")
    if csv_file is None:
        st.info("Choose your electrode CSV (for example, electrodes_180.csv).")
    else:
        try:
            points_df = pd.read_csv(csv_file)
        except Exception as exc:
            st.error(f"Could not read the CSV: {exc}")
            st.stop()

        if points_df.empty:
            st.error("The CSV has no rows.")
            st.stop()
        columns = list(points_df.columns)
        default_lon = auto_column(columns, ["longitude_deg", "longitude", "lon", "long", "x_lon"])
        default_lat = auto_column(columns, ["latitude_deg", "latitude", "lat", "y_lat"])
        default_existing = next((c for c in columns if str(c).strip().lower() in ["elevation_m", "elevation", "elev", "altitude", "z"]), "(none)")

        c1, c2, c3 = st.columns(3)
        lon_col = c1.selectbox("Longitude column", columns, index=columns.index(default_lon))
        lat_col = c2.selectbox("Latitude column", columns, index=columns.index(default_lat))
        existing_options = ["(none)"] + columns
        existing_index = existing_options.index(default_existing) if default_existing in existing_options else 0
        existing_col = c3.selectbox("Existing elevation column (optional)", existing_options, index=existing_index)

        method = st.radio("Sampling method", ["Bilinear interpolation", "Nearest pixel"], horizontal=True,
                          help="Bilinear estimates a continuous surface from the four surrounding pixel centers. Nearest pixel returns the closest raster cell.")
        signature_payload = "|".join([
            hashlib.sha256(tif_file.getvalue()).hexdigest(),
            hashlib.sha256(csv_file.getvalue()).hexdigest(),
            str(lon_col), str(lat_col), str(existing_col), str(method)
        ])
        current_signature = hashlib.sha256(signature_payload.encode("utf-8")).hexdigest()
        if lon_col == lat_col:
            st.error("Choose different columns for longitude and latitude.")
        elif st.button("Sample elevation for every row", type="primary"):
            work = points_df.copy()
            longitude = pd.to_numeric(work[lon_col], errors="coerce")
            latitude = pd.to_numeric(work[lat_col], errors="coerce")
            valid_coords = longitude.notna() & latitude.notna() & longitude.between(-180, 180) & latitude.between(-90, 90)
            result = work.copy()
            result["input_longitude"] = longitude
            result["input_latitude"] = latitude
            result["raster_crs"] = raster["crs"]
            result["raster_x"] = np.nan
            result["raster_y"] = np.nan
            result["DEM_nearest_pixel_elevation"] = np.nan
            result["DEM_bilinear_elevation"] = np.nan
            result["inside_raster_extent"] = False
            result["sampled_elevation"] = np.nan
            result["elevation_difference_sampled_minus_existing"] = np.nan
            result["match_status"] = "Invalid/missing coordinates"

            indices = np.flatnonzero(valid_coords.to_numpy())
            if len(indices):
                try:
                    xs, ys = transform_lonlat_to_raster(
                        raster["crs"],
                        longitude.iloc[indices].astype(float).tolist(),
                        latitude.iloc[indices].astype(float).tolist()
                    )
                    nearest, bilinear, inside, rows, cols = sample_dem(data, raster["transform"], xs, ys)
                    result.loc[result.index[indices], "raster_x"] = xs
                    result.loc[result.index[indices], "raster_y"] = ys
                    result.loc[result.index[indices], "DEM_nearest_pixel_elevation"] = nearest
                    result.loc[result.index[indices], "DEM_bilinear_elevation"] = bilinear
                    result.loc[result.index[indices], "inside_raster_extent"] = inside
                    chosen = bilinear if method == "Bilinear interpolation" else nearest
                    result.loc[result.index[indices], "sampled_elevation"] = chosen
                    status = []
                    for j in range(len(indices)):
                        if not inside[j]:
                            status.append("Outside raster extent")
                        elif method == "Bilinear interpolation" and not np.isfinite(bilinear[j]):
                            status.append("Inside extent, but bilinear neighbors include NoData/edge")
                        elif not np.isfinite(chosen[j]):
                            status.append("NoData pixel")
                        else:
                            status.append("Sampled successfully")
                    result.loc[result.index[indices], "match_status"] = status
                    if existing_col != "(none)":
                        existing = pd.to_numeric(result[existing_col], errors="coerce")
                        result["elevation_difference_sampled_minus_existing"] = result["sampled_elevation"] - existing
                except Exception as exc:
                    st.error(f"Sampling failed: {exc}")
                    st.stop()

            st.session_state["elevation_match_result"] = result
            st.session_state["elevation_match_source_name"] = csv_file.name
            st.session_state["elevation_match_method"] = method
            st.session_state["elevation_match_signature"] = current_signature

        if ("elevation_match_result" in st.session_state and
                st.session_state.get("elevation_match_signature") == current_signature):
            result = st.session_state["elevation_match_result"]
            st.markdown("### Results")
            success = result["match_status"].eq("Sampled successfully")
            outside = result["match_status"].eq("Outside raster extent")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("CSV rows", f"{len(result):,}")
            k2.metric("Successfully sampled", f"{int(success.sum()):,}")
            k3.metric("Outside raster", f"{int(outside.sum()):,}")
            k4.metric("Could not sample", f"{int((~success).sum()):,}")

            if existing_col != "(none)" and "elevation_difference_sampled_minus_existing" in result:
                diff = pd.to_numeric(result["elevation_difference_sampled_minus_existing"], errors="coerce").dropna()
                if len(diff):
                    a, b, c = st.columns(3)
                    a.metric("Mean absolute elevation difference", f"{diff.abs().mean():.4f} {unit}")
                    b.metric("Maximum absolute difference", f"{diff.abs().max():.4f} {unit}")
                    c.metric(f"Within 0.5 {unit}", f"{int((diff.abs() <= 0.5).sum())} points")
                    st.caption("Difference is sampled DEM elevation minus the selected existing elevation column. Confirm both use the same vertical units and datum.")
            st.dataframe(dataframe_for_display(result), width="stretch", height=420)
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download matched results as CSV", csv_bytes,
                               file_name="electrode_elevation_matches.csv", mime="text/csv")
            summary = pd.DataFrame([
                ["Input CSV", st.session_state.get("elevation_match_source_name", csv_file.name)],
                ["GeoTIFF CRS", raster["crs"]],
                ["Sampling method", st.session_state.get("elevation_match_method", method)],
                ["Total input rows", len(result)],
                ["Successfully sampled", int(success.sum())],
                ["Outside raster extent", int(outside.sum())],
                ["NoData/edge/invalid", int((~success).sum())],
                ["Elevation units", unit],
            ], columns=["Item", "Value"])
            try:
                xlsx_bytes = make_excel_bytes(summary, result)
                st.download_button("Download matched results as Excel", xlsx_bytes,
                                   file_name="electrode_elevation_matches.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                st.warning(f"Excel export could not be prepared: {exc}. CSV export is available.")
        elif "elevation_match_result" in st.session_state:
            st.info("The uploaded CSV or selected columns/method changed. Click **Sample elevation for every row** to refresh the results.")

with tab_kml:
    st.subheader("KML survey lines, DEM elevations, and profiles")
    st.write("Upload a KML containing Start/End point placemarks. The app pairs points within each KML folder, draws the survey lines, samples the GeoTIFF, and creates elevation profiles.")
    kml_file = st.file_uploader("Upload KML survey file", type=["kml"], key="kml_survey_upload")
    if kml_file is not None:
        try:
            kml_points, kml_lines = parse_kml_surveys(kml_file.getvalue())
            st.metric("KML point placemarks", f"{len(kml_points):,}")
            st.dataframe(dataframe_for_display(kml_points), width="stretch", hide_index=True)
            if kml_lines.empty:
                st.warning("KML points were found, but no folder contained both a Start and an End placemark. The map/profile needs paired Start/End points in each folder.")
            else:
                st.success(f"Detected {len(kml_lines)} survey line(s) from {len(kml_points)} point placemark(s).")
                samples_per_line = st.slider("Samples along each survey line", min_value=20, max_value=500, value=150, step=10,
                                              help="More samples make smoother profiles but take longer to calculate.")
                profile_df, line_summary = make_line_samples(kml_lines, raster, data, samples_per_line)
                st.markdown("#### Survey line summary")
                st.dataframe(dataframe_for_display(line_summary), width="stretch", hide_index=True)
                good_count = int(profile_df["DEM_elevation"].notna().sum())
                st.caption(f"DEM samples available: {good_count:,} of {len(profile_df):,}. Elevation values use the raster's source units ({unit}). KML altitude may use a different vertical reference.")

                st.markdown("#### Survey lines on the 2D raster map")
                stride_kml = max(1, math.ceil(max(data.shape) / 1000))
                view_kml = data[::stride_kml, ::stride_kml]
                fig_kml_2d = go.Figure(go.Heatmap(z=view_kml, colorscale="Earth", connectgaps=False,
                    colorbar=dict(title=f"Elevation<br>({unit})"), hovertemplate="Display row: %{y}<br>Display column: %{x}<br>Value: %{z:.3f}<extra></extra>"))
                inv = ~raster["transform"]
                for _, line in kml_lines.iterrows():
                    tt = np.linspace(0, 1, 100)
                    lon_line = line.start_lon + (line.end_lon-line.start_lon)*tt
                    lat_line = line.start_lat + (line.end_lat-line.start_lat)*tt
                    xx, yy = transform_lonlat_to_raster(raster["crs"], lon_line.tolist(), lat_line.tolist())
                    cols, rows = [], []
                    for x, y in zip(xx, yy):
                        cc, rr = inv * (float(x), float(y))
                        cols.append(cc / stride_kml)
                        rows.append(rr / stride_kml)
                    fig_kml_2d.add_trace(go.Scatter(x=cols, y=rows, mode="lines+markers", name=str(line.survey),
                        marker=dict(size=4), line=dict(width=3), hovertemplate=f"{line.survey}<br>Column: %{{x:.1f}}<br>Row: %{{y:.1f}}<extra></extra>"))
                fig_kml_2d.update_layout(height=600, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="Raster column (display scale)", yaxis_title="Raster row")
                fig_kml_2d.update_yaxes(autorange="reversed")
                st.plotly_chart(fig_kml_2d, width="stretch")

                st.markdown("#### Survey lines on the 3D terrain")
                max_dim_kml = st.slider("3D surface maximum dimension for KML view", 80, 400, 180, 10, key="kml_3d_dim")
                stride3 = max(1, math.ceil(max(data.shape) / max_dim_kml))
                z3 = data[::stride3, ::stride3]
                rr3 = np.arange(0, data.shape[0], stride3, dtype=float) + 0.5
                cc3 = np.arange(0, data.shape[1], stride3, dtype=float) + 0.5
                cgrid, rgrid = np.meshgrid(cc3, rr3)
                aff = raster["transform"]
                x3 = aff.a*cgrid + aff.b*rgrid + aff.c
                y3 = aff.d*cgrid + aff.e*rgrid + aff.f
                fig_kml_3d = go.Figure(data=[go.Surface(x=x3, y=y3, z=z3, colorscale="Earth", connectgaps=False,
                    colorbar=dict(title=f"Elevation<br>({unit})"), name="DEM surface", showscale=True)])
                for survey, group in profile_df.groupby("survey", sort=False):
                    valid_line = group["DEM_elevation"].notna() & group["inside_raster_extent"]
                    if valid_line.any():
                        fig_kml_3d.add_trace(go.Scatter3d(x=group.loc[valid_line,"raster_x"], y=group.loc[valid_line,"raster_y"],
                            z=group.loc[valid_line,"DEM_elevation"], mode="lines", line=dict(width=6), name=str(survey),
                            hovertemplate=f"{survey}<br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>DEM elevation: %{{z:.3f}}<extra></extra>"))
                fig_kml_3d.update_layout(height=680, margin=dict(l=0,r=0,t=10,b=0),
                    scene=dict(xaxis_title=f"X ({raster['crs']})", yaxis_title=f"Y ({raster['crs']})", zaxis_title=f"Elevation ({unit})", aspectmode="auto"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                st.plotly_chart(fig_kml_3d, width="stretch")

                st.markdown("#### Elevation profile by survey line")
                chosen_survey = st.selectbox("Choose survey line", list(profile_df["survey"].drop_duplicates()), key="kml_profile_survey")
                one_profile = profile_df[profile_df["survey"] == chosen_survey]
                fig_profile = go.Figure()
                fig_profile.add_trace(go.Scatter(x=one_profile["distance_m"], y=one_profile["DEM_elevation"], mode="lines", name="DEM elevation",
                    connectgaps=False, hovertemplate="Distance: %{x:.2f} m<br>DEM elevation: %{y:.3f}<extra></extra>"))
                fig_profile.update_layout(height=420, xaxis_title="Distance from start (approx. metres)", yaxis_title=f"Elevation ({unit})", margin=dict(l=10,r=10,t=20,b=10))
                st.plotly_chart(fig_profile, width="stretch")
                st.download_button("Download KML survey endpoint summary (CSV)", line_summary.to_csv(index=False).encode("utf-8"),
                    file_name="kml_survey_dem_summary.csv", mime="text/csv")
                st.download_button("Download DEM samples along survey lines (CSV)", profile_df.to_csv(index=False).encode("utf-8"),
                    file_name="kml_survey_dem_profiles.csv", mime="text/csv")
                try:
                    kml_xlsx = make_excel_bytes(line_summary, profile_df)
                    st.download_button("Download KML survey results (Excel)", kml_xlsx,
                        file_name="kml_survey_dem_results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                except Exception as exc:
                    st.warning(f"Excel export failed: {exc}. CSV downloads are still available.")
        except Exception as exc:
            st.error(f"Could not process this KML file: {exc}")
            st.info("Expected input: KML Point placemarks with longitude,latitude[,altitude], with a Start and End point inside each survey folder.")



with tab_osm:
    st.subheader("OpenStreetMap and survey-area explorer")
    st.write("Explore the DEM coverage, zoom to KML survey lines, or draw a custom area. The basemap is OpenStreetMap; elevation values come from your uploaded GeoTIFF.")

    area_mode = st.radio(
        "Area to display",
        ["KML survey bounds", "Full GeoTIFF coverage", "Custom drawn area"],
        horizontal=True,
        key="osm_area_mode",
        help="Custom area: draw a rectangle or polygon on the map, then the map will zoom to that selection."
    )
    display_mode = st.radio(
        "Map layers",
        ["OpenStreetMap only", "Satellite imagery only", "OpenStreetMap + elevation overlay", "Satellite imagery + elevation overlay", "Elevation overlay only", "3D satellite globe"],
        horizontal=True,
        key="osm_display_mode"
    )
    opacity = st.slider("Elevation overlay opacity", 0.10, 1.00, 0.65, 0.05, key="osm_elevation_opacity")
    kml_for_map = st.file_uploader("Optional: upload KML survey lines for this map", type=["kml"], key="osm_kml_upload")

    map_points = pd.DataFrame()
    map_lines = pd.DataFrame()
    if kml_for_map is not None:
        try:
            map_points, map_lines = parse_kml_surveys(kml_for_map.getvalue())
            st.caption(f"Loaded {len(map_points)} KML point(s) and {len(map_lines)} paired survey line(s).")
        except Exception as exc:
            st.warning(f"Could not read KML for map overlay: {exc}")

    # Transform the raster extent to geographic coordinates for a correctly located web map.
    try:
        west, south, east, north = transform_bounds(
            raster["crs"], "EPSG:4326", *raster["bounds"], densify_pts=21
        )
        dem_bounds = [[south, west], [north, east]]
        dem_center = [(south + north) / 2, (west + east) / 2]
    except Exception as exc:
        st.error(f"Could not transform the GeoTIFF bounds to latitude/longitude: {exc}")
        dem_bounds = None
        dem_center = [20.0, 0.0]

    # Compute the KML bounding box, when valid line/point coordinates are available.
    kml_bounds = None
    if not map_points.empty:
        min_lon = float(map_points["longitude"].min())
        max_lon = float(map_points["longitude"].max())
        min_lat = float(map_points["latitude"].min())
        max_lat = float(map_points["latitude"].max())
        # Add a small margin so endpoints do not sit against the map edge.
        lon_pad = max((max_lon - min_lon) * 0.12, 0.0005)
        lat_pad = max((max_lat - min_lat) * 0.12, 0.0005)
        kml_bounds = [[min_lat - lat_pad, min_lon - lon_pad], [max_lat + lat_pad, max_lon + lon_pad]]

    # Persist the custom geometry across Streamlit reruns.
    drawn = st.session_state.get("osm_custom_drawings", [])
    custom_bounds = None
    if drawn:
        try:
            coords = []
            for feature in drawn:
                geometry = feature.get("geometry", {})
                def collect_xy(obj):
                    if isinstance(obj, (list, tuple)):
                        if len(obj) >= 2 and all(isinstance(v, (int, float)) for v in obj[:2]):
                            coords.append((float(obj[0]), float(obj[1])))
                        else:
                            for part in obj:
                                collect_xy(part)
                collect_xy(geometry.get("coordinates", []))
            if coords:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                custom_bounds = [[min(ys), min(xs)], [max(ys), max(xs)]]
                if custom_bounds[0] == custom_bounds[1]:
                    custom_bounds = None
        except Exception:
            custom_bounds = None

    if area_mode == "KML survey bounds":
        selected_bounds = kml_bounds or dem_bounds
        if kml_bounds is None:
            st.info("Upload a KML to zoom to its survey bounds. Showing the DEM extent until a KML is supplied.")
    elif area_mode == "Custom drawn area":
        selected_bounds = custom_bounds or dem_bounds
        st.caption("Draw a rectangle or polygon using the drawing tools on the map. After drawing, select Custom drawn area to zoom to it.")
    else:
        selected_bounds = dem_bounds

    center = dem_center
    if selected_bounds:
        center = [(selected_bounds[0][0] + selected_bounds[1][0]) / 2,
                  (selected_bounds[0][1] + selected_bounds[1][1]) / 2]

    if display_mode == "3D satellite globe":
        st.caption("3D satellite globe view. It zooms to the selected area and can show KML survey points and lines. For the DEM's measured elevation surface, use the 3D Terrain tab.")
        if selected_bounds:
            south_sel, west_sel = selected_bounds[0]
            north_sel, east_sel = selected_bounds[1]
        else:
            south_sel, west_sel, north_sel, east_sel = -10, -10, 10, 10
        points_js = []
        for _, point in map_points.iterrows():
            points_js.append({
                "lon": float(point["longitude"]), "lat": float(point["latitude"]),
                "label": f"{point['survey']} — {point['point_name']}"
            })
        lines_js = []
        for _, line in map_lines.iterrows():
            lines_js.append({
                "name": str(line["survey"]),
                "start": [float(line["start_lon"]), float(line["start_lat"])],
                "end": [float(line["end_lon"]), float(line["end_lat"])]
            })
        points_json = json.dumps(points_js).replace("</", "<\\/")
        lines_json = json.dumps(lines_js).replace("</", "<\\/")
        cesium_html = f'''<!doctype html>
<html><head><meta charset="utf-8">
<link href="https://cesium.com/downloads/cesiumjs/releases/1.125/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<style>html,body,#cesiumContainer{{width:100%;height:100%;margin:0;padding:0;overflow:hidden;background:#10151d}} .credit-note{{position:absolute;bottom:4px;left:8px;z-index:5;color:#fff;background:rgba(0,0,0,.55);padding:4px 7px;font:12px Arial}}</style>
<script src="https://cesium.com/downloads/cesiumjs/releases/1.125/Build/Cesium/Cesium.js"></script></head>
<body><div id="cesiumContainer"></div><div class="credit-note">Satellite imagery © Esri and contributors · 3D globe is imagery on the ellipsoid, not DEM-derived terrain</div>
<script>
const viewer = new Cesium.Viewer('cesiumContainer', {{
  animation:false, timeline:false, baseLayerPicker:false, geocoder:false,
  homeButton:true, sceneModePicker:false, navigationHelpButton:false,
  fullscreenButton:true, infoBox:true, selectionIndicator:true,
  terrainProvider:new Cesium.EllipsoidTerrainProvider(),
  imageryProvider:new Cesium.UrlTemplateImageryProvider({{
    url:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    credit:'Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community'
  }})
}});
viewer.scene.globe.depthTestAgainstTerrain = false;
viewer.camera.flyTo({{destination:Cesium.Rectangle.fromDegrees({west_sel}, {south_sel}, {east_sel}, {north_sel}), duration:1.2}});
const points = {points_json};
for (const p of points) {{
 viewer.entities.add({{position:Cesium.Cartesian3.fromDegrees(p.lon,p.lat), point:{{pixelSize:9,color:Cesium.Color.RED,outlineColor:Cesium.Color.WHITE,outlineWidth:2}}, label:{{text:p.label,font:'13px sans-serif',fillColor:Cesium.Color.WHITE,outlineColor:Cesium.Color.BLACK,outlineWidth:2,style:Cesium.LabelStyle.FILL_AND_OUTLINE,verticalOrigin:Cesium.VerticalOrigin.BOTTOM,pixelOffset:new Cesium.Cartesian2(0,-10)}}}});
}}
const lines = {lines_json};
for (const l of lines) {{
 viewer.entities.add({{name:l.name,polyline:{{positions:Cesium.Cartesian3.fromDegreesArray([...l.start,...l.end]),width:4,material:Cesium.Color.RED}}}});
}}
</script></body></html>'''
        st.components.v1.html(cesium_html, height=700, scrolling=False)
    else:
        fmap = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True)
        if display_mode in ("OpenStreetMap only", "OpenStreetMap + elevation overlay"):
            folium.TileLayer(
                tiles="OpenStreetMap", name="OpenStreetMap", control=False, overlay=False
            ).add_to(fmap)
        elif display_mode in ("Satellite imagery only", "Satellite imagery + elevation overlay"):
            folium.TileLayer(
                tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
                name="Satellite imagery", control=False, overlay=False, max_zoom=19
            ).add_to(fmap)
        else:
            # Elevation-only mode uses a neutral background under the DEM overlay.
            folium.TileLayer(
                tiles="CartoDB positron", name="Neutral background", control=False, overlay=False
            ).add_to(fmap)

        # Downsample and reproject the DEM to WGS84 for a light-weight map overlay.
        if display_mode in ("OpenStreetMap + elevation overlay", "Satellite imagery + elevation overlay", "Elevation overlay only") and dem_bounds:
            try:
                max_overlay_dim = 900
                overlay_stride = max(1, math.ceil(max(data.shape) / max_overlay_dim))
                source_overlay = data[::overlay_stride, ::overlay_stride]
                geographic_ratio = max((east - west) / max(north - south, 1e-9), 1e-6)
                if geographic_ratio >= 1:
                    dst_width = max_overlay_dim
                    dst_height = max(1, min(max_overlay_dim, int(round(max_overlay_dim / geographic_ratio))))
                else:
                    dst_height = max_overlay_dim
                    dst_width = max(1, min(max_overlay_dim, int(round(max_overlay_dim * geographic_ratio))))
                destination = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
                source_transform = raster["transform"] * Affine.scale(overlay_stride, overlay_stride)
                destination_transform = from_bounds(west, south, east, north, dst_width, dst_height)
                reproject(
                    source=source_overlay.astype(np.float32), destination=destination,
                    src_transform=source_transform, src_crs=raster["crs"], src_nodata=np.nan,
                    dst_transform=destination_transform, dst_crs="EPSG:4326", dst_nodata=np.nan,
                    resampling=Resampling.bilinear
                )
                overlay_values = destination
                finite_overlay = overlay_values[np.isfinite(overlay_values)]
                if finite_overlay.size:
                    lo, hi = np.nanpercentile(finite_overlay, [2, 98])
                    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                        lo, hi = float(np.nanmin(finite_overlay)), float(np.nanmax(finite_overlay) + 1e-9)
                    norm = np.clip((overlay_values - lo) / (hi - lo), 0, 1)
                    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
                    palette = np.array([[35, 100, 65], [120, 145, 75], [190, 170, 115], [205, 205, 185], [250, 250, 245]], dtype=float)
                    rgb = np.stack([np.interp(norm, stops, palette[:, ch]) for ch in range(3)], axis=-1).astype(np.uint8)
                    alpha = (np.isfinite(overlay_values).astype(np.uint8) * int(255 * opacity))[:, :, None]
                    rgba = np.concatenate([rgb, alpha], axis=-1)
                    # Pass a NumPy RGBA array, not a PIL Image object. Folium converts
                    # the array to a PNG data URL that is JSON/HTML serializable.
                    folium.raster_layers.ImageOverlay(
                        image=rgba,
                        bounds=dem_bounds,
                        opacity=1.0,
                        interactive=True,
                        cross_origin=False,
                        zindex=2,
                        name="GeoTIFF elevation"
                    ).add_to(fmap)
                    st.caption(f"Elevation overlay is downsampled to approximately {dst_width} × {dst_height} pixels for map performance. Display stretch: 2nd–98th percentile ({lo:.3f}–{hi:.3f} {unit}); original DEM values are unchanged.")
            except Exception as exc:
                st.warning(f"Could not render the elevation overlay: {exc}")

        if not map_points.empty:
            for _, point in map_points.iterrows():
                label = f"{point['survey']} — {point['point_name']}"
                folium.CircleMarker(
                    location=[float(point["latitude"]), float(point["longitude"])],
                    radius=5, color="#d62728", weight=2, fill=True, fill_opacity=0.95,
                    tooltip=label,
                    popup=f"{label}<br>Lat: {point['latitude']:.7f}<br>Lon: {point['longitude']:.7f}"
                ).add_to(fmap)
        if not map_lines.empty:
            for _, line in map_lines.iterrows():
                folium.PolyLine(
                    locations=[[float(line.start_lat), float(line.start_lon)], [float(line.end_lat), float(line.end_lon)]],
                    color="#ff2d2d", weight=4, opacity=0.95, tooltip=str(line.survey)
                ).add_to(fmap)

        Draw(
            export=False,
            draw_options={"polyline": True, "rectangle": True, "polygon": True, "circle": False, "marker": False, "circlemarker": False},
            edit_options={"edit": True, "remove": True}
        ).add_to(fmap)
        if selected_bounds:
            fmap.fit_bounds(selected_bounds)
        map_state = st_folium(
            fmap,
            width=None,
            height=680,
            returned_objects=["all_drawings", "last_active_drawing"],
            key="osm_survey_map"
        )
        if map_state and map_state.get("all_drawings") is not None:
            st.session_state["osm_custom_drawings"] = map_state["all_drawings"]
        if area_mode == "Custom drawn area" and custom_bounds is None:
            st.info("No custom shape has been saved yet. Draw a rectangle or polygon above; the map will use its bounds after the next rerun.")
        if selected_bounds:
            st.caption(f"Current map bounds (approx.): lat {selected_bounds[0][0]:.6f} to {selected_bounds[1][0]:.6f}, lon {selected_bounds[0][1]:.6f} to {selected_bounds[1][1]:.6f}.")
        st.caption("OpenStreetMap © OpenStreetMap contributors. Satellite imagery © Esri and contributors. The elevation overlay is derived from the uploaded GeoTIFF; it is not satellite imagery.")

with tab_meta:
    metadata = {
        "Driver": raster["driver"],
        "CRS": raster["crs"],
        "Width": raster["width"],
        "Height": raster["height"],
        "Band count": raster["count"],
        "Band 1 data type": raster["dtype"],
        "Resolution": str(raster["resolution"]),
        "Bounds": str(raster["bounds"]),
        "NoData value": str(raster["nodata"]),
        "Band 1 units": str(unit),
        "Band 1 scale": raster["scale"],
        "Band 1 offset": raster["offset"],
        "Valid pixels": int(valid.size),
        "Minimum valid value": float(np.nanmin(data)),
        "Maximum valid value": float(np.nanmax(data)),
        "Mean valid value": float(np.nanmean(data)),
        "Median valid value": float(np.nanmedian(data)),
    }
    metadata_df = pd.DataFrame([{"Property": k, "Value": str(v)} for k, v in metadata.items()])
    st.dataframe(dataframe_for_display(metadata_df), width="stretch", hide_index=True)
    if raster["tags"]:
        st.subheader("GeoTIFF tags")
        st.json(raster["tags"])
    st.warning("The app samples band 1. Verify that this band represents elevation and confirm its vertical units/datum in the source metadata.")
