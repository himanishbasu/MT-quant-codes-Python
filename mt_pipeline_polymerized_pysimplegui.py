# =====================================================================================
# Polymerized Microtubule (TUJ1) Detection Pipeline (PySimpleGUI, batch-capable)
# =====================================================================================
#
# GUI:
#   - PySimpleGUI replaces the entire PyQt6 GUI.
#   - Multiple TIF/TIFF files can be selected.
#   - File selection AND all parameters are in the SAME window.
#   - Previously used parameters are remembered in ~/.mt_pipeline_settings.json.
#
# Processing:
#   - Single image: shows a Tuj1/skeleton sanity-check plot.
#   - Multiple images: saves *_polSkeletonMask.tif for each image and
#     polymerized_mt_summary.csv next to the first selected image.
#
# DEPENDENCIES:
#   conda install -c conda-forge "scikit-image>=0.19" tifffile numpy scipy matplotlib pandas
#   pip install PySimpleGUI
# =====================================================================================
# line 688 # <-- new line: enforce a floor on the auto threshold

# %% Imports

import json
import os

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import gaussian_filter, binary_dilation, label as ndi_label

from skimage import measure
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt

import PySimpleGUI as sg


# %% Step 1: PySimpleGUI file selection + parameter input

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".mt_pipeline_settings.json")


def load_settings():
    defaults = {
        "tuj1Channel": 1,
        "microtubuleThreshold": -1,
        "tubuleThickness": 2,
        "minTubuleLength": 20,
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
        except Exception:
            pass

    return defaults


def save_settings(settings):
    """Merge current settings into the shared settings file."""
    existing = {}

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                existing = json.load(f)
        except Exception:
            pass

    existing.update(settings)

    with open(SETTINGS_FILE, "w") as f:
        json.dump(existing, f, indent=2)


def get_gui_parameters(settings):
    """
    Opens one PySimpleGUI window containing both:
      1. multifile TIFF selection
      2. polymerized-MT parameters

    Returns:
        (file_paths, params)

    or:

        (None, None) if cancelled.
    """

    sg.theme("SystemDefault")

    layout = [
        [
            sg.Text("Polymerized Microtubule (TUJ1) Detection", font=("Arial", 14, "bold"))
        ],

        [
            sg.Text("Select one or multiple TIF/TIFF images and set the detection parameters below.", size=(75, 2))
        ],

        [sg.HorizontalSeparator()],

        [
            sg.Text("Input image(s):", size=(38, 1)),
            sg.Input(key="-FILES-", expand_x=True, readonly=True),
            sg.FilesBrowse(
                "Browse...",
                target="-FILES-",
                file_types=(("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")),
            ),
        ],

        [
            sg.Text(
                "Selected files are displayed above. Ctrl/Shift-click can be used in the file browser.",
                size=(75, 1),
            )
        ],

        [sg.HorizontalSeparator()],

        [
            sg.Text("Detection parameters", font=("Arial", 11, "bold"))
        ],

        [
            sg.Text("Tuj1 channel (1-based):", size=(38, 1)),
            sg.Spin(
                values=list(range(1, 21)),
                initial_value=int(settings["tuj1Channel"]),
                key="-TUJ1-",
                size=(12, 1),
            ),
        ],

        [
            sg.Text("Baseline / background threshold (-1 = auto):", size=(38, 1)),
            sg.Input(str(settings["microtubuleThreshold"]), key="-THRESHOLD-", size=(14, 1)),
        ],

        [
            sg.Text("MT thickness (px; affects DoG scales):", size=(38, 1)),
            sg.Input(str(settings["tubuleThickness"]), key="-THICKNESS-", size=(14, 1)),
        ],

        [
            sg.Text("Minimum polymerized MT Feret length (px):", size=(38, 1)),
            sg.Input(str(settings["minTubuleLength"]), key="-MINLENGTH-", size=(14, 1)),
        ],

        [
            sg.Text(
                "Use -1 for automatic baseline estimation. The baseline is estimated independently for each image.",
                size=(75, 2),
            )
        ],

        [sg.HorizontalSeparator()],

        [
            sg.Push(),
            sg.Button("Run Detection", key="-RUN-", bind_return_key=True, size=(18, 1)),
            sg.Button("Cancel", key="-CANCEL-", size=(12, 1)),
            sg.Push(),
        ],
    ]

    window = sg.Window("Polymerized Microtubule Detection", layout, resizable=True, finalize=True)

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "-CANCEL-"):
            window.close()
            return None, None

        if event == "-RUN-":

            # ---------------------------------------------------------
            # Files
            # ---------------------------------------------------------

            selected = values["-FILES-"].strip()

            if not selected:
                sg.popup_error("Please select at least one TIF/TIFF image.", title="No images selected")
                continue

            # PySimpleGUI returns multiple selected files separated by ;
            file_paths = [p for p in selected.split(";") if p.strip()]

            if not file_paths:
                sg.popup_error("No valid image files were selected.", title="No images selected")
                continue

            # Verify that every selected file still exists.
            missing = [p for p in file_paths if not os.path.isfile(p)]

            if missing:
                sg.popup_error(
                    "The following selected file(s) could not be found:\n\n" + "\n".join(missing),
                    title="Missing file(s)",
                )
                continue

            # ---------------------------------------------------------
            # Parameters
            # ---------------------------------------------------------

            try:
                tuj1Channel = int(values["-TUJ1-"])
                microtubuleThreshold = float(values["-THRESHOLD-"])
                tubuleThickness = float(values["-THICKNESS-"])
                minTubuleLength = float(values["-MINLENGTH-"])

                if not 1 <= tuj1Channel <= 20:
                    raise ValueError("Tuj1 channel must be between 1 and 20.")

                if microtubuleThreshold < -1:
                    raise ValueError("Baseline / background threshold must be -1 or greater.")

                if tubuleThickness <= 0:
                    raise ValueError("MT thickness must be greater than 0.")

                if minTubuleLength <= 0:
                    raise ValueError("Minimum polymerized MT Feret length must be greater than 0.")

            except (ValueError, TypeError) as e:
                sg.popup_error(f"Invalid parameter:\n\n{e}", title="Invalid parameters")
                continue

            params = {
                "tuj1Channel": tuj1Channel,
                "microtubuleThreshold": microtubuleThreshold,
                "tubuleThickness": tubuleThickness,
                "minTubuleLength": minTubuleLength,
            }

            save_settings(params)

            window.close()
            return file_paths, params


# %% Step 2: channel extraction

def get_channel(image, channel_index_1based):
    """
    Extract a single 2D channel from a possibly multi-dimensional hyperstack,
    mirroring Fiji's Stack.setChannel().

    Uses a size heuristic to find the channel axis:
    channel counts are small while Y/X are large, so the channel axis is
    assumed to be the smallest non-spatial dimension.

    NOTE:
        Verify this against your actual TIFF shape if the hyperstack has an
        unusual axis order.
    """

    arr = np.asarray(image)
    ch = channel_index_1based - 1

    if arr.ndim == 2:
        if ch != 0:
            raise ValueError(f"Image is 2D and therefore only has one channel; requested channel {channel_index_1based}.")
        return arr

    if arr.ndim == 3:
        c_axis = int(np.argmin(arr.shape))

        if not 0 <= ch < arr.shape[c_axis]:
            raise ValueError(
                f"Requested channel {channel_index_1based}, but the detected channel axis has size {arr.shape[c_axis]} for image shape {arr.shape}."
            )

        return np.take(arr, ch, axis=c_axis)

    if arr.ndim >= 4:
        spatial_axes = (arr.ndim - 2, arr.ndim - 1)
        candidate_axes = list(range(arr.ndim - 2))

        if not candidate_axes:
            raise ValueError(f"Cannot determine channel axis for shape {arr.shape}")

        c_axis = min(candidate_axes, key=lambda a: arr.shape[a])

        if not 0 <= ch < arr.shape[c_axis]:
            raise ValueError(
                f"Requested channel {channel_index_1based}, but the detected channel axis has size {arr.shape[c_axis]} for image shape {arr.shape}."
            )

        idx = [0] * arr.ndim
        idx[c_axis] = ch
        idx[spatial_axes[0]] = slice(None)
        idx[spatial_axes[1]] = slice(None)

        return arr[tuple(idx)]

    raise ValueError(f"Unsupported image shape: {arr.shape}")


# %% Step 3: polymerized MT detection

# -------------------- baseline estimation --------------------

def peak_baseline_finder(profile, poly_order=4, max_iter=200, tol=1e-4):
    """
    Estimates a baseline for a 1D intensity profile using iterative
    asymmetric polynomial fitting.
    """

    n = len(profile)

    if n <= poly_order:
        return float(np.mean(profile))

    x = np.arange(n, dtype=np.float64)
    y = profile.astype(np.float64)

    # Downsample very long profiles for speed/stability.
    if n > 1000:
        win = n // 1000 + 1
        n_new = n // win

        x = np.array([x[i * win:(i + 1) * win].mean() for i in range(n_new)])
        y = np.array([y[i * win:(i + 1) * win].mean() for i in range(n_new)])

    # Ensure enough points for polynomial fitting.
    if len(x) <= poly_order:
        return float(np.mean(y))

    baseline = y.copy()

    for _ in range(max_iter):
        coeffs = np.polyfit(x, baseline, poly_order)
        fit_vals = np.polyval(coeffs, x)
        new_baseline = np.minimum(y, fit_vals)

        mean_diff = np.mean(np.abs(new_baseline - baseline))

        baseline = new_baseline

        if mean_diff < tol:
            break

    return float(np.mean(baseline))


def grid_scan_baseline(tuj1_img, scan_interval_percent=10, poly_order=4, max_iter=200, tol=1e-4):
    """
    Estimates the polymerized-MT background baseline by scanning several
    horizontal lines across the image.
    """

    h, w = tuj1_img.shape
    baselines = []

    for p in range(scan_interval_percent, 100, scan_interval_percent):
        y_pos = min(int((p / 100) * h), h - 1)
        profile = tuj1_img[y_pos, :]
        baselines.append(peak_baseline_finder(profile, poly_order, max_iter, tol))

    return float(np.mean(baselines)) if baselines else 0.0


# -------------------- single-pixel stripe removal --------------------

def remove_1px_stripes(mask, iterations=3, horizontal=True, vertical=True, diagonal=False):
    """
    Removes single-pixel-wide isolated stripes from a boolean mask.
    """

    m = mask.copy()

    for _ in range(iterations):
        keep = np.ones_like(m, dtype=bool)

        if horizontal:
            left = np.roll(m, 1, axis=1)
            left[:, 0] = False

            right = np.roll(m, -1, axis=1)
            right[:, -1] = False

            keep &= left | right

        if vertical:
            up = np.roll(m, 1, axis=0)
            up[0, :] = False

            down = np.roll(m, -1, axis=0)
            down[-1, :] = False

            keep &= up | down

        if diagonal:
            upleft = np.roll(np.roll(m, 1, axis=0), 1, axis=1)
            upleft[0, :] = False
            upleft[:, 0] = False

            downright = np.roll(np.roll(m, -1, axis=0), -1, axis=1)
            downright[-1, :] = False
            downright[:, -1] = False

            upright = np.roll(np.roll(m, 1, axis=0), -1, axis=1)
            upright[0, :] = False
            upright[:, -1] = False

            downleft = np.roll(np.roll(m, -1, axis=0), 1, axis=1)
            downleft[-1, :] = False
            downleft[:, 0] = False

            keep &= (upleft | downright) & (upright | downleft)

        m = m & keep

    return m


# -------------------- short blob removal --------------------

def remove_short_blobs_by_feret(mask, minTubuleLength):
    """
    Removes connected components whose maximum Feret diameter is below
    minTubuleLength.
    """

    labeled = measure.label(mask, connectivity=2)
    out = mask.copy()

    for p in measure.regionprops(labeled):
        if p.feret_diameter_max < minTubuleLength:
            out[labeled == p.label] = False

    return out


# -------------------- stubby isolated MT removal --------------------

def remove_small_components(skeleton, min_size):
    """
    Removes connected components smaller than min_size pixels.

    Uses 8-connectivity so diagonal skeleton pixels remain connected.
    """

    structure = np.ones((3, 3), dtype=bool)
    labels, num = ndi_label(skeleton, structure=structure)
    sizes = np.bincount(labels.ravel())
    output = np.zeros_like(skeleton, dtype=bool)

    for i in range(1, num + 1):
        if sizes[i] >= min_size:
            output[labels == i] = True

    return output


# -------------------- noisy skeleton removal --------------------

def remove_high_complexity_tubules(skeleton, max_density, size):
    """
    Removes skeleton connected-components whose junction-pixel density is
    too high.

    Only objects with pixel count <= size are checked. Larger objects are
    left untouched.
    """

    labeled = measure.label(skeleton, connectivity=2)

    if labeled.max() == 0:
        return skeleton

    padded = np.pad(skeleton.astype(np.uint8), 1, mode="constant")

    neighbor_count = (
        padded[0:-2, 0:-2]
        + padded[0:-2, 1:-1]
        + padded[0:-2, 2:]
        + padded[1:-1, 0:-2]
        + padded[1:-1, 2:]
        + padded[2:, 0:-2]
        + padded[2:, 1:-1]
        + padded[2:, 2:]
    )

    junction_mask = skeleton & (neighbor_count >= 3)
    out = skeleton.copy()

    for p in measure.regionprops(labeled):
        object_size = p.area

        if object_size > size:
            continue

        junction_pixels = int(np.count_nonzero(junction_mask[labeled == p.label]))

        if object_size > 0 and (junction_pixels / object_size) > max_density:
            out[labeled == p.label] = False

    return out


# -------------------- break small enclosed loops --------------------

def break_small_loops(skeleton, max_diameter):
    """
    Finds small enclosed loops and opens each with a single-pixel cut.
    """

    background = ~skeleton

    structure_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)

    labeled_bg, num_labels = ndi_label(background, structure=structure_4)

    if num_labels == 0:
        return skeleton

    border_labels = set(labeled_bg[0, :]) | set(labeled_bg[-1, :]) | set(labeled_bg[:, 0]) | set(labeled_bg[:, -1])
    border_labels.discard(0)

    mask = skeleton.copy()
    dilation_structure = np.ones((3, 3), dtype=bool)

    for lbl in range(1, num_labels + 1):
        if lbl in border_labels:
            continue

        hole = labeled_bg == lbl
        area = int(hole.sum())

        if area <= 1:
            continue

        ys, xs = np.nonzero(hole)

        diameter = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1)

        if diameter >= max_diameter:
            continue

        dilated_hole = binary_dilation(hole, structure=dilation_structure)
        border_pixels = dilated_hole & mask

        if not border_pixels.any():
            continue

        cut_y, cut_x = np.argwhere(border_pixels)[0]
        mask[cut_y, cut_x] = False

    return mask


# -------------------- branch pruning --------------------

def count_neighbors(mask, y, x):
    h, w = mask.shape

    y0 = max(y - 1, 0)
    y1 = min(y + 2, h)

    x0 = max(x - 1, 0)
    x1 = min(x + 2, w)

    return int(mask[y0:y1, x0:x1].sum()) - (1 if mask[y, x] else 0)


def get_neighbor_coords(mask, y, x, exclude=None):
    h, w = mask.shape
    coords = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and (ny, nx) != exclude:
                coords.append((ny, nx))

    return coords


def prune_short_branches(skeleton, minTubuleLength, max_iterations=25):
    """
    Removes terminal spurs shorter than minTubuleLength from the skeleton.
    """

    mask = skeleton.copy()

    for _ in range(max_iterations):
        pruned_any = False
        ys, xs = np.nonzero(mask)

        endpoints = [
            (y, x)
            for y, x in zip(ys, xs)
            if count_neighbors(mask, y, x) == 1
        ]

        for sy, sx in endpoints:

            if not mask[sy, sx]:
                continue

            path = [(sy, sx)]

            cy, cx = sy, sx
            prev = None

            reached_junction = False
            reached_dead_end = False

            while not reached_junction and not reached_dead_end:

                candidates = get_neighbor_coords(mask, cy, cx, exclude=prev)

                if len(candidates) == 0:
                    reached_dead_end = True

                elif len(candidates) == 1:
                    ny, nx = candidates[0]

                    path.append((ny, nx))

                    prev = (cy, cx)
                    cy, cx = ny, nx

                    total_n = count_neighbors(mask, cy, cx)

                    if total_n >= 3:
                        reached_junction = True

                    elif total_n == 1:
                        reached_dead_end = True

                else:
                    reached_junction = True

            length = 0.0

            for k in range(1, len(path)):
                dy = path[k][0] - path[k - 1][0]
                dx = path[k][1] - path[k - 1][1]

                length += np.sqrt(2) if dy != 0 and dx != 0 else 1.0

            if length < minTubuleLength:

                end_idx = len(path) - 1

                if reached_junction:
                    end_idx = len(path) - 2

                for k in range(0, end_idx + 1):
                    py, px = path[k]
                    mask[py, px] = False

                pruned_any = True

        if not pruned_any:
            break

    return mask


# -------------------- main polymerized MT detection --------------------

def detect_polymerized_microtubules(tuj1_img, microtubuleThreshold, tubuleThickness, minTubuleLength):
    tuj1 = tuj1_img.astype(np.float64)

    # 1. Baseline
    if microtubuleThreshold < 0:

        microtubuleThreshold = grid_scan_baseline(tuj1)
        microtubuleThreshold = np.floor(microtubuleThreshold * 1.1)
        microtubuleThreshold = max(microtubuleThreshold, 25)   # <-- new line: enforce a floor on the auto threshold


        print(f"Auto-estimated microtubuleThreshold: {microtubuleThreshold}")

    baseline_subtracted = np.clip(tuj1 - microtubuleThreshold, 0, None)

    # 2. DoG + threshold
    sigma_fine = tubuleThickness * 1
    sigma_coarse = tubuleThickness * 10

    gauss_fine = gaussian_filter(baseline_subtracted, sigma=sigma_fine)
    gauss_coarse = gaussian_filter(baseline_subtracted, sigma=sigma_coarse)

    dog = gauss_fine - gauss_coarse
    mask = dog > 0

    # 3. Remove single-pixel stripes
    mask = remove_1px_stripes(mask)

    # 4. Remove short blobs by Feret length
    mask = remove_short_blobs_by_feret(mask, minTubuleLength)

    # 5. Skeletonize
    skeleton = skeletonize(mask)

    # 5.5 Remove very small and stubby MTs after skeletonization
    skeleton = remove_small_components(skeleton, minTubuleLength)

    # 6. Remove noisy skeleton regions
    print("Removing noisy skeleton regions...")

    skeleton = remove_high_complexity_tubules(skeleton, 0.3, tuj1.shape[0] * tuj1.shape[1])
    skeleton = remove_high_complexity_tubules(skeleton, 0.125, minTubuleLength * 10)

    # 7. Break small loops and prune branches
    print("Breaking small loops...")

    skeleton = break_small_loops(skeleton, minTubuleLength * 2)

    print("Pruning short branches...")
    skeleton = prune_short_branches(skeleton, minTubuleLength)
    
    print("Final Cleanup...")
    skeleton = remove_small_components(skeleton, minTubuleLength)

    return skeleton, microtubuleThreshold


# %% Step 4: batch processing

def process_images(file_paths, params):
    """
    Process one or multiple images.

    Single-image behavior:
        - no mask is automatically saved
        - sanity-check plot is shown

    Multi-image behavior:
        - each skeleton is saved as *_polSkeletonMask.tif
        - polymerized_mt_summary.csv is saved beside the first image
        - no individual sanity-check plots are shown
    """

    tuj1Channel = params["tuj1Channel"]
    microtubuleThreshold = params["microtubuleThreshold"]
    tubuleThickness = params["tubuleThickness"]
    minTubuleLength = params["minTubuleLength"]

    summary_rows = []
    last_result = None

    total_files = len(file_paths)

    for idx, image_path in enumerate(file_paths, start=1):

        image_name = os.path.basename(image_path)
        image_dir = os.path.dirname(image_path) or os.getcwd()

        print(f"\nProcessing ({idx}/{total_files}): {image_name}")

        try:

            image = tifffile.imread(image_path)

            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")

            tuj1_img = get_channel(image, tuj1Channel)

            print(f"  Tuj1 shape: {tuj1_img.shape}")

            pol_skeleton, resolved_threshold = detect_polymerized_microtubules(
                tuj1_img,
                microtubuleThreshold,
                tubuleThickness,
                minTubuleLength,
            )

            neurite_length_px = int(np.count_nonzero(pol_skeleton))

            print(f"  Polymerized MT skeleton length: {neurite_length_px} px")

            row = {
                "image": image_name,
                "neurite_length_px": neurite_length_px,
                "threshold_used": resolved_threshold,
                "tubuleThickness": tubuleThickness,
                "minTubuleLength": minTubuleLength,
            }

            # Save skeleton masks

            mask_out_name = os.path.splitext(image_name)[0] + "_polSkeletonMask.tif"
            mask_out_path = os.path.join(image_dir, mask_out_name)

            tifffile.imwrite(mask_out_path, pol_skeleton.astype(np.uint8) * 255)

            row["mask_saved_to"] = mask_out_path

            print(f"  Mask saved: {mask_out_path}")

            summary_rows.append(row)

            if total_files == 1:
                last_result = {
                    "tuj1_img": tuj1_img,
                    "pol_skeleton": pol_skeleton,
                }

        except Exception as e:

            print(f"  ERROR processing {image_name}: {e}")

            summary_rows.append({
                "image": image_name,
                "neurite_length_px": np.nan,
                "threshold_used": np.nan,
                "tubuleThickness": tubuleThickness,
                "minTubuleLength": minTubuleLength,
                "error": str(e),
            })

    return summary_rows, last_result


# %% Step 5: main

def main():

    settings = load_settings()

    file_paths, params = get_gui_parameters(settings)

    if file_paths is None:
        print("No files selected. Exiting.")
        return

    print(f"\nSelected {len(file_paths)} file(s).")

    print("\nParameters:")

    for key, value in params.items():
        print(f"  {key}: {value}")

    summary_rows, last_result = process_images(file_paths, params)

    # -------------------------------------------------------------
    # Multi-file summary
    # -------------------------------------------------------------

    if len(file_paths) > 1:

        summary_df = pd.DataFrame(summary_rows)

        print("\n=== Summary ===")
        print(summary_df.to_string(index=False))

        summary_csv_path = os.path.join(
            os.path.dirname(file_paths[0]) or os.getcwd(),
            "polymerized_mt_summary.csv",
        )

        summary_df.to_csv(summary_csv_path, index=False)

        print(f"\nSummary saved to:\n{summary_csv_path}")

        successful = int(summary_df["neurite_length_px"].notna().sum())
        failed = len(summary_df) - successful

        message = (
            "Processing complete.\n\n"
            f"Images selected: {len(file_paths)}\n"
            f"Successfully processed: {successful}\n"
            f"Failed: {failed}\n\n"
            "Skeleton masks were saved next to their source images.\n\n"
            "Summary CSV:\n"
            f"{summary_csv_path}"
        )

        sg.popup(message, title="Polymerized MT Detection Complete")

    # -------------------------------------------------------------
    # Single-file sanity check
    # -------------------------------------------------------------

    elif len(file_paths) == 1 and last_result is not None:

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        last_result["tuj1_img"] = np.clip(
            last_result["tuj1_img"],
            0,
            np.percentile(last_result["tuj1_img"], 90),
        )

        axes[0].imshow(last_result["tuj1_img"], cmap="gray")
        axes[0].set_title("Tuj1 channel")
        axes[0].axis("off")

        axes[1].imshow(last_result["pol_skeleton"], cmap="gray")
        axes[1].set_title("Polymerized MT skeleton")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\nNeurite length (px): {summary_rows[0]['neurite_length_px']}")


if __name__ == "__main__":
    main()

