# =====================================================================================
# Depolymerized Microtubule (TUJ1) Detection Pipeline (PySimpleGUI, batch-capable)
# =====================================================================================
#
# GUI:
#   - PySimpleGUI handles file selection and parameter entry.
#   - Multiple TIF/TIFF files can be selected.
#   - File selection AND all parameters are in the SAME window.
#   - Previously used parameters are remembered in ~/.mt_pipeline_settings.json.
#
# Processing:
#   - Single image: shows a Tuj1/depolymerized-mask sanity-check plot, no summary CSV.
#   - Multiple images: saves *_depolMask.tif for each image (as it finishes) and
#     depolymerized_mt_summary.csv next to the first selected image.
#
# DEPENDENCIES:
#   conda install -c conda-forge "scikit-image>=0.19" tifffile numpy scipy matplotlib pandas
#   pip install PySimpleGUI
# =====================================================================================


# %% Imports

import json
import os

import numpy as np
import pandas as pd
import tifffile

from scipy.signal import convolve2d
from scipy.ndimage import binary_dilation

from skimage import measure

import matplotlib.pyplot as plt

import PySimpleGUI as sg


# %% Step 1: PySimpleGUI file selection + parameter input

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".mt_pipeline_settings.json")


def load_settings():
    defaults = {
        "tuj1Channel": 1,
        "depolymerizedMtThreshold": 4000,
        "minAreaDepol": 50,
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
      2. depolymerized-MT parameters

    Returns:
        (file_paths, params)

    or:

        (None, None) if cancelled.
    """

    sg.theme("SystemDefault")

    layout = [
        [
            sg.Text("Depolymerized Microtubule (TUJ1) Detection", font=("Arial", 14, "bold"))
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
            sg.Text("Depolymerized MT threshold:", size=(38, 1)),
            sg.Input(str(settings["depolymerizedMtThreshold"]), key="-THRESHOLD-", size=(14, 1)),
        ],

        [
            sg.Text("Minimum area of depolymerized MT objects (px):", size=(38, 1)),
            sg.Input(str(settings["minAreaDepol"]), key="-MINAREA-", size=(14, 1)),
        ],

        [
            sg.Text(
                "The threshold is applied identically to every selected image (no auto-estimation).",
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

    window = sg.Window("Depolymerized Microtubule Detection", layout, resizable=True, finalize=True)

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
                depolymerizedMtThreshold = float(values["-THRESHOLD-"])
                minAreaDepol = float(values["-MINAREA-"])

                if not 1 <= tuj1Channel <= 20:
                    raise ValueError("Tuj1 channel must be between 1 and 20.")

                if minAreaDepol <= 0:
                    raise ValueError("Minimum area of depolymerized MT objects must be greater than 0.")

            except (ValueError, TypeError) as e:
                sg.popup_error(f"Invalid parameter:\n\n{e}", title="Invalid parameters")
                continue

            params = {
                "tuj1Channel": tuj1Channel,
                "depolymerizedMtThreshold": depolymerizedMtThreshold,
                "minAreaDepol": minAreaDepol,
            }

            save_settings(params)

            window.close()
            return file_paths, params


# %% Step 2: channel extraction

def get_channel(image, channel_index_1based):
    """
    Extract a single 2D channel from a possibly multi-dimensional hyperstack,
    mirroring Fiji's Stack.setChannel().

    Uses a size heuristic to find the channel axis: channel counts are small
    while Y/X are large, so the channel axis is assumed to be the smallest
    non-spatial dimension.

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


# %% Step 3: depolymerized MT detection (core logic unchanged from original)

# -------------------- remove 1px-wide isolated stripes --------------------

def remove_1px_stripes(mask, iterations=3, horizontal=True, vertical=True, diagonal=False):
    """
    Removes single-pixel-wide isolated stripes from a boolean mask. A
    foreground pixel is dropped if, along every *enabled* axis, it has no
    foreground neighbor on either side of that axis. Repeats for the given
    number of iterations.
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


# -------------------- small object removal --------------------

def remove_small_objects_below(mask, min_area):
    """
    Erases connected components with area <= min_area (8-connected).
    """

    labeled = measure.label(mask, connectivity=2)
    out = mask.copy()

    for p in measure.regionprops(labeled):
        if p.area <= min_area:
            out[labeled == p.label] = False

    return out


# -------------------- puncta shape filter (Circ./Round/Solidity) --------------------

def filter_puncta_shape(mask):
    """Keeps only round/solid (puncta-like) objects; discards elongated shapes."""

    labeled = measure.label(mask, connectivity=2)
    out = mask.copy()

    for p in measure.regionprops(labeled):
        area = p.area
        perimeter = p.perimeter if p.perimeter > 0 else 1e-9
        circ = min(4 * np.pi * area / (perimeter ** 2), 1.0)
        major_axis = p.major_axis_length if p.major_axis_length > 0 else 1e-9
        roundness = 4 * area / (np.pi * (major_axis ** 2))
        solidity = p.solidity

        keep = (circ > 0.15) and (roundness > 0.3) and (solidity > 0.3)

        if not keep:
            out[labeled == p.label] = False

    return out


# -------------------- 1px binary dilation (3x3, 8-connected) --------------------

def binary_dilate_1px(mask):
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure, iterations=1)


# -------------------- main depolymerized MT detection --------------------

def detect_depolymerized_neurites(tuj1_img, depolymerizedMtThreshold, minAreaDepol):
    tuj1 = tuj1_img.astype(np.float64)

    # isolate dot-like (puncta) structures with a 7x7 center-weighted kernel
    kernel = -np.ones((7, 7))
    kernel[3, 3] = 300
    kernel = kernel / kernel.sum()  # normalize weights to sum to 1
    filtered = convolve2d(tuj1, kernel, mode='same', boundary='symm')

    # subtract baseline (median + stddev) to remove background
    filtered = filtered - (np.median(filtered) + np.std(filtered))

    # rescale intensity back to roughly match the original image's range
    filtered = filtered * (np.percentile(tuj1, 99.9) / np.percentile(filtered, 99.9))

    # threshold to a boolean mask
    mask = filtered > depolymerizedMtThreshold

    mask = remove_1px_stripes(mask)
    mask = remove_small_objects_below(mask, minAreaDepol)
    mask = filter_puncta_shape(mask)

    if not mask.any():  # guard against a fully empty mask
        mask[0, 0] = True

    mask = binary_dilate_1px(mask)
    return mask


# %% Step 4: batch processing

def process_images(file_paths, params):
    """
    Process one or multiple images.

    Single-image behavior:
        - mask is still saved to disk
        - sanity-check plot is shown
        - no summary CSV

    Multi-image behavior:
        - each mask is saved as *_depolMask.tif immediately after that image finishes
        - depolymerized_mt_summary.csv is saved beside the first image
        - no sanity-check plots are shown
    """

    tuj1Channel = params["tuj1Channel"]
    depolymerizedMtThreshold = params["depolymerizedMtThreshold"]
    minAreaDepol = params["minAreaDepol"]

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

            depol_mask = detect_depolymerized_neurites(
                tuj1_img,
                depolymerizedMtThreshold,
                minAreaDepol,
            )

            depolymerized_pixel_count = int(np.count_nonzero(depol_mask))

            print(f"  Depolymerized MT pixel count: {depolymerized_pixel_count}")

            row = {
                "image": image_name,
                "depolymerized_pixel_count": depolymerized_pixel_count,
                "tuj1Channel": tuj1Channel,
                "depolymerizedMtThreshold": depolymerizedMtThreshold,
                "minAreaDepol": minAreaDepol,
            }

            # Save mask immediately after this image is done

            mask_out_name = os.path.splitext(image_name)[0] + "_depolMask.tif"
            mask_out_path = os.path.join(image_dir, mask_out_name)

            tifffile.imwrite(mask_out_path, depol_mask.astype(np.uint8) * 255)

            row["mask_saved_to"] = mask_out_path

            print(f"  Mask saved: {mask_out_path}")

            summary_rows.append(row)

            if total_files == 1:
                last_result = {
                    "tuj1_img": tuj1_img,
                    "depol_mask": depol_mask,
                }

        except Exception as e:

            print(f"  ERROR processing {image_name}: {e}")

            summary_rows.append({
                "image": image_name,
                "depolymerized_pixel_count": np.nan,
                "tuj1Channel": tuj1Channel,
                "depolymerizedMtThreshold": depolymerizedMtThreshold,
                "minAreaDepol": minAreaDepol,
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
            "depolymerized_mt_summary.csv",
        )

        summary_df.to_csv(summary_csv_path, index=False)

        print(f"\nSummary saved to:\n{summary_csv_path}")

        successful = int(summary_df["depolymerized_pixel_count"].notna().sum())
        failed = len(summary_df) - successful

        message = (
            "Processing complete.\n\n"
            f"Images selected: {len(file_paths)}\n"
            f"Successfully processed: {successful}\n"
            f"Failed: {failed}\n\n"
            "Depolymerized MT masks were saved next to their source images.\n\n"
            "Summary CSV:\n"
            f"{summary_csv_path}"
        )

        sg.popup(message, title="Depolymerized MT Detection Complete")

    # -------------------------------------------------------------
    # Single-file sanity check
    # -------------------------------------------------------------

    elif len(file_paths) == 1 and last_result is not None:

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        vmin_tuj1 = np.percentile(last_result["tuj1_img"], 0)
        vmax_tuj1 = np.percentile(last_result["tuj1_img"], 99.0)

        axes[0].imshow(last_result["tuj1_img"], cmap="gray", vmin=vmin_tuj1, vmax=vmax_tuj1)
        axes[0].set_title("Tuj1 channel")
        axes[0].axis("off")

        axes[1].imshow(last_result["depol_mask"], cmap="gray")
        axes[1].set_title("Depolymerized MT mask")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

        print(f"\nDepolymerized MT pixel count: {summary_rows[0]['depolymerized_pixel_count']}")


if __name__ == "__main__":
    main()
