import io
import math
import hashlib
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import plotly.graph_objects as go
from rasterio.warp import transform as transform_coords

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

tab_map, tab_3d, tab_match, tab_meta = st.tabs(["2D Elevation Map", "3D Terrain", "Match CSV Coordinates", "Raster Metadata"])

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
    st.plotly_chart(fig, use_container_width=True)
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
    st.plotly_chart(fig3d, use_container_width=True)
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
            st.dataframe(result, use_container_width=True, height=420)
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
    st.dataframe(pd.DataFrame([{"Property": k, "Value": v} for k, v in metadata.items()]),
                 use_container_width=True, hide_index=True)
    if raster["tags"]:
        st.subheader("GeoTIFF tags")
        st.json(raster["tags"])
    st.warning("The app samples band 1. Verify that this band represents elevation and confirm its vertical units/datum in the source metadata.")
