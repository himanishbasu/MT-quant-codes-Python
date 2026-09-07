# =====================================================================================
# Polymerized / Depolymerized / Nuclei Mask Combination & Summary Pipeline
# (PySimpleGUI, batch-capable)
# =====================================================================================
#
# WORKFLOW:
#   1. Select one or multiple *_polSkeletonMask.tif files (the file browser is
#      restricted to this suffix).
#   2. For each selected polymerized skeleton mask, the corresponding files are
#      derived automatically:
#           <base>_polSkeletonMask.tif   (selected)
#           <base>_depolMask.tif         (depolymerized MT mask)
#           <base>_nucleiMask.tif        (nuclei mask)
#      If the depol or nuclei mask is missing, a blank (all-False) mask of the
#      same shape is used instead so processing can proceed uniformly.
#   3. neuriteLength_raw is recorded immediately upon loading the polymerized
#      skeleton mask (total positive pixels, before any subtraction).
#   4. Nuclei are counted (connected components) and their total area is
#      measured BEFORE any dilation.
#   5. Masks are optionally dilated and subtracted from one another in this
#      order:
#           - dilate nuclei mask
#           - (optional) subtract dilated nuclei mask from polymerized mask
#           - (optional) subtract dilated nuclei mask from depolymerized mask
#           - dilate depolymerized mask
#           - (optional) subtract dilated depolymerized mask from polymerized mask
#           - record polymerizedMT_raw (post-subtraction, pre-dilation)
#           - dilate polymerized mask
#           - record polymerizedMT_dilated and depolMT_dilated
#   6. A final RGB PNG is saved next to the source masks:
#           depolymerized -> red, polymerized -> green, nuclei -> cyan
#   7. A summary Excel workbook (one row per image) is saved alongside the
#      first selected image, and printed to the console. It includes the
#      parameters used, plus a second sheet explaining each column.
#
# DEPENDENCIES:
#   conda install -c conda-forge "scikit-image>=0.19" tifffile numpy scipy matplotlib pandas openpyxl
#   pip install PySimpleGUI
# =====================================================================================


# %% Imports

import json
import os

import numpy as np
import pandas as pd
import tifffile

from scipy.ndimage import binary_dilation, label as ndi_label

from skimage import measure

import matplotlib.pyplot as plt

import PySimpleGUI as sg


# %% Step 1: settings persistence

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".mt_pipeline_settings.json")


def load_settings():
    defaults = {
        "excludeDepolMask": True,
        "excludeNucleiMask": True,
        "dilatePolymerizedPx": 0,
        "dilateDepolymerizedPx": 0,
        "dilateNucleiPx": 0,
    }

    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
        except Exception:
            pass

    return defaults


def save_settings(settings):
    """Merge current settings into the shared settings file (read-modify-write)."""
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


# %% Step 2: PySimpleGUI file selection + parameter input

REQUIRED_SUFFIX = "_polSkeletonMask.tif"

PROCESSING_STEPS_TEXT = (
    "1. Load the polymerized skeleton mask; count all positive pixels as "
    "neuriteLength_raw (total neurite length before anything is subtracted).\n"
    "2. Load the matching _depolMask.tif and _nucleiMask.tif (blank masks are "
    "used if either file is missing).\n"
    "3. Count nuclei (connected components) and total nuclei area on the raw "
    "nuclei mask -> NucCounts, TotalNucArea.\n"
    "4. Dilate the nuclei mask by the amount set below.\n"
    "5. If 'Exclude nuclei mask' is checked, remove nuclei-overlapping pixels "
    "from both the polymerized and depolymerized masks.\n"
    "6. Dilate the depolymerized mask by the amount set below.\n"
    "7. If 'Exclude depolymerized mask' is checked, remove depolymerized-"
    "overlapping pixels from the polymerized mask.\n"
    "8. Record polymerizedMT_raw (pixel count after subtraction, before its "
    "own dilation).\n"
    "9. Dilate the polymerized mask by the amount set below.\n"
    "10. Record the final pixel counts, polymerizedMT_dilated and "
    "depolMT_dilated.\n"
    "11. Save an RGB overlay PNG (depolymerized=red, polymerized=green, "
    "nuclei=cyan) and write all results to an Excel summary."
)


def get_gui_parameters(settings):
    """
    Opens one PySimpleGUI window containing both:
      1. multifile selection restricted to *_polSkeletonMask.tif
      2. mask-combination parameters

    Returns:
        (file_paths, params) or (None, None) if cancelled.
    """

    sg.theme("SystemDefault")

    layout = [
        [sg.Text("Polymerized / Depolymerized / Nuclei Mask Combination", font=("Arial", 14, "bold"))],

        [sg.Text(
            "Select one or multiple *_polSkeletonMask.tif files. The matching "
            "_depolMask.tif and _nucleiMask.tif files (if present) will be found automatically.",
            size=(80, 2),
        )],

        [sg.HorizontalSeparator()],

        [
            sg.Text("Polymerized skeleton mask(s):", size=(38, 1)),
            sg.Input(key="-FILES-", expand_x=True, readonly=True),
            sg.FilesBrowse(
                "Browse...",
                target="-FILES-",
                file_types=(("Polymerized Skeleton Mask", "*" + REQUIRED_SUFFIX),),
            ),
        ],

        [sg.Text(
            "Only files ending in _polSkeletonMask.tif may be selected. "
            "Ctrl/Shift-click can be used in the file browser.",
            size=(80, 1),
        )],

        [sg.HorizontalSeparator()],

        [sg.Text("Combination parameters", font=("Arial", 11, "bold"))],

        [
            sg.Checkbox(
                "Exclude depolymerized mask (remove overlap from polymerized mask)",
                key="-EXCLDEPOL-",
                default=bool(settings["excludeDepolMask"]),
            )
        ],

        [
            sg.Checkbox(
                "Exclude nuclei mask (remove overlap from polymerized and depolymerized masks)",
                key="-EXCLNUC-",
                default=bool(settings["excludeNucleiMask"]),
            )
        ],

        [sg.HorizontalSeparator()],

        [
            sg.Text("Dilate polymerized mask (px):", size=(38, 1)),
            sg.Input(str(settings["dilatePolymerizedPx"]), key="-DILPOL-", size=(14, 1)),
        ],

        [
            sg.Text("Dilate depolymerized mask (px):", size=(38, 1)),
            sg.Input(str(settings["dilateDepolymerizedPx"]), key="-DILDEPOL-", size=(14, 1)),
        ],

        [
            sg.Text("Dilate nuclei mask (px):", size=(38, 1)),
            sg.Input(str(settings["dilateNucleiPx"]), key="-DILNUC-", size=(14, 1)),
        ],

        [sg.Text(
            "Dilation is applied as N iterations of a 3x3 structuring element "
            "(mirrors ImageJ's Binary > Dilate run N times).",
            size=(80, 2),
        )],

        [sg.HorizontalSeparator()],

        [sg.Frame(
            "How this works (processing steps, for reference)",
            [[sg.Multiline(
                PROCESSING_STEPS_TEXT,
                size=(90, 12),
                disabled=True,
                no_scrollbar=False,
                background_color=sg.theme_background_color(),
                border_width=0,
            )]],
        )],

        [sg.HorizontalSeparator()],

        [
            sg.Push(),
            sg.Button("Run", key="-RUN-", bind_return_key=True, size=(18, 1)),
            sg.Button("Cancel", key="-CANCEL-", size=(12, 1)),
            sg.Push(),
        ],
    ]

    window = sg.Window("Mask Combination & Summary", layout, resizable=True, finalize=True)

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
                sg.popup_error("Please select at least one *_polSkeletonMask.tif image.", title="No images selected")
                continue

            file_paths = [p for p in selected.split(";") if p.strip()]

            if not file_paths:
                sg.popup_error("No valid image files were selected.", title="No images selected")
                continue

            missing = [p for p in file_paths if not os.path.isfile(p)]

            if missing:
                sg.popup_error(
                    "The following selected file(s) could not be found:\n\n" + "\n".join(missing),
                    title="Missing file(s)",
                )
                continue

            invalid = [p for p in file_paths if not os.path.basename(p).endswith(REQUIRED_SUFFIX)]

            if invalid:
                sg.popup_error(
                    "The following selected file(s) do not end with "
                    f"'{REQUIRED_SUFFIX}':\n\n" + "\n".join(invalid),
                    title="Invalid file selection",
                )
                continue

            # ---------------------------------------------------------
            # Parameters
            # ---------------------------------------------------------

            try:
                excludeDepolMask = bool(values["-EXCLDEPOL-"])
                excludeNucleiMask = bool(values["-EXCLNUC-"])

                dilatePolymerizedPx = float(values["-DILPOL-"])
                dilateDepolymerizedPx = float(values["-DILDEPOL-"])
                dilateNucleiPx = float(values["-DILNUC-"])

                if dilatePolymerizedPx < 0:
                    raise ValueError("Dilate polymerized mask (px) must be 0 or greater.")

                if dilateDepolymerizedPx < 0:
                    raise ValueError("Dilate depolymerized mask (px) must be 0 or greater.")

                if dilateNucleiPx < 0:
                    raise ValueError("Dilate nuclei mask (px) must be 0 or greater.")

            except (ValueError, TypeError) as e:
                sg.popup_error(f"Invalid parameter:\n\n{e}", title="Invalid parameters")
                continue

            params = {
                "excludeDepolMask": excludeDepolMask,
                "excludeNucleiMask": excludeNucleiMask,
                "dilatePolymerizedPx": dilatePolymerizedPx,
                "dilateDepolymerizedPx": dilateDepolymerizedPx,
                "dilateNucleiPx": dilateNucleiPx,
            }

            save_settings(params)

            window.close()
            return file_paths, params


# %% Step 3: path derivation

def derive_related_paths(pol_mask_path):
    """
    Given a *_polSkeletonMask.tif path, derive:
        - base_name (image name with the pol-mask suffix stripped)
        - depol_mask_path (<base>_depolMask.tif)
        - nuclei_mask_path (<base>_nucleiMask.tif)
    """

    directory = os.path.dirname(pol_mask_path) or os.getcwd()
    filename = os.path.basename(pol_mask_path)

    base_name = filename[: -len(REQUIRED_SUFFIX)]

    depol_mask_path = os.path.join(directory, base_name + "_depolMask.tif")
    nuclei_mask_path = os.path.join(directory, base_name + "_nucleiMask.tif")

    return base_name, depol_mask_path, nuclei_mask_path


# %% Step 4: mask loading

def to_2d_bool_mask(arr):
    """Squeezes a possibly multi-dimensional mask array down to 2D and thresholds it."""

    arr = np.squeeze(np.asarray(arr))

    if arr.ndim > 2:
        # Fall back to the first 2D plane if the mask was saved with extra
        # singleton-like dimensions that didn't fully squeeze away.
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]

    if arr.ndim != 2:
        raise ValueError(f"Could not interpret mask array with shape {arr.shape} as 2D.")

    return arr > 0


def load_mask(path, shape):
    """
    Loads a boolean mask from disk if it exists, otherwise returns a blank
    (all-False) mask of the given shape.

    Returns:
        (mask, available) where `available` is False if a blank mask was used.
    """

    if path is not None and os.path.isfile(path):
        raw = tifffile.imread(path)
        mask = to_2d_bool_mask(raw)

        if mask.shape != shape:
            raise ValueError(
                f"Mask at {path} has shape {mask.shape}, expected {shape} to match the polymerized mask."
            )

        return mask, True

    return np.zeros(shape, dtype=bool), False


# %% Step 5: dilation helper

def dilate_mask(mask, px):
    """
    Dilates a boolean mask by `px` iterations of a 3x3 structuring element,
    mirroring ImageJ's Binary > Dilate run N times.
    """

    px = int(round(px))

    if px <= 0:
        return mask.copy()

    structure = np.ones((3, 3), dtype=bool)

    return binary_dilation(mask, structure=structure, iterations=px)


# %% Step 6: nuclei counting

def count_nuclei(nuclei_mask):
    """
    Counts connected nuclei objects (8-connectivity) and the total nuclei area,
    both measured on the RAW (pre-dilation) mask.
    """

    labeled = measure.label(nuclei_mask, connectivity=2)

    NucCounts = int(labeled.max())
    TotalNucArea = int(nuclei_mask.sum())

    return NucCounts, TotalNucArea


# %% Step 7: RGB overlay

def make_rgb_overlay(depol_mask, pol_mask, nuclei_mask):
    """
    Builds an RGB overlay image:
        depolymerized -> red
        polymerized   -> green
        nuclei        -> cyan (green + blue)
    """

    h, w = pol_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    rgb[..., 0] = depol_mask.astype(np.uint8) * 255
    rgb[..., 1] = (pol_mask | nuclei_mask).astype(np.uint8) * 255
    rgb[..., 2] = nuclei_mask.astype(np.uint8) * 255

    return rgb


# %% Step 8: per-image processing

def process_single_image(pol_mask_path, params):
    """
    Runs the full mask-combination pipeline for a single image.

    Returns a dict with the summary row and, for optional plotting/saving,
    the intermediate masks.
    """

    excludeDepolMask = params["excludeDepolMask"]
    excludeNucleiMask = params["excludeNucleiMask"]
    dilatePolymerizedPx = params["dilatePolymerizedPx"]
    dilateDepolymerizedPx = params["dilateDepolymerizedPx"]
    dilateNucleiPx = params["dilateNucleiPx"]

    base_name, depol_mask_path, nuclei_mask_path = derive_related_paths(pol_mask_path)

    # ---- Load masks ----

    raw = tifffile.imread(pol_mask_path)
    pol_mask = to_2d_bool_mask(raw)
    shape = pol_mask.shape

    # Total neurite length before any subtraction (depol, nuclei, or dilation).
    neuriteLength_raw = int(pol_mask.sum())

    depol_mask, depol_available = load_mask(depol_mask_path, shape)
    nuclei_mask, nuclei_available = load_mask(nuclei_mask_path, shape)

    # ---- Nuclei counting (before dilation) ----

    NucCounts, TotalNucArea = count_nuclei(nuclei_mask)

    # ---- Dilate nuclei mask ----

    nuclei_mask_dilated = dilate_mask(nuclei_mask, dilateNucleiPx)

    # ---- Subtract nuclei mask from polymerized / depolymerized masks ----

    if excludeNucleiMask:
        pol_mask = pol_mask & ~nuclei_mask_dilated
        depol_mask = depol_mask & ~nuclei_mask_dilated

    # ---- Dilate depolymerized mask ----

    depol_mask_dilated = dilate_mask(depol_mask, dilateDepolymerizedPx)

    # ---- Subtract depolymerized mask from polymerized mask ----

    if excludeDepolMask:
        pol_mask = pol_mask & ~depol_mask_dilated

    # ---- Raw polymerized pixel count (post-subtraction, pre-dilation) ----

    polymerizedMT_raw = int(pol_mask.sum())

    # ---- Dilate polymerized mask ----

    pol_mask_dilated = dilate_mask(pol_mask, dilatePolymerizedPx)

    # ---- Final pixel counts ----

    polymerizedMT_dilated = int(pol_mask_dilated.sum())
    depolMT_dilated = int(depol_mask_dilated.sum())

    # ---- RGB overlay ----

    rgb = make_rgb_overlay(depol_mask_dilated, pol_mask_dilated, nuclei_mask_dilated)

    row = {
        "image": base_name,
        "neuriteLength_raw": neuriteLength_raw,
        "polymerizedMT_raw": polymerizedMT_raw,
        "polymerizedMT_dilated": polymerizedMT_dilated,
        "depolMT_dilated": depolMT_dilated,
        "NucCounts": NucCounts,
        "TotalNucArea": TotalNucArea,
        "depolMaskAvailable": depol_available,
        "nucleiMaskAvailable": nuclei_available,
        "excludeDepolMask": excludeDepolMask,
        "excludeNucleiMask": excludeNucleiMask,
        "dilatePolymerizedPx": dilatePolymerizedPx,
        "dilateDepolymerizedPx": dilateDepolymerizedPx,
        "dilateNucleiPx": dilateNucleiPx,
    }

    intermediates = {
        "pol_mask_dilated": pol_mask_dilated,
        "depol_mask_dilated": depol_mask_dilated,
        "nuclei_mask_dilated": nuclei_mask_dilated,
        "rgb": rgb,
    }

    return row, intermediates


# %% Step 9: column descriptions (for the Excel "Column Descriptions" sheet)

def build_column_descriptions():
    descriptions = [
        ("image", "Base image name, shared by the pol/depol/nuclei mask files (suffixes stripped)."),
        ("neuriteLength_raw", "Total positive pixels in the polymerized skeleton mask as loaded, "
                               "before subtracting nuclei or depolymerized MTs, and before any dilation."),
        ("polymerizedMT_raw", "Total polymerized pixels after subtracting nuclei/depolymerized overlap "
                               "(if those options were checked), but before dilating the polymerized mask."),
        ("polymerizedMT_dilated", "Total polymerized pixels after the final dilation step (dilatePolymerizedPx)."),
        ("depolMT_dilated", "Total depolymerized pixels after nuclei subtraction (if checked) and its own "
                             "dilation (dilateDepolymerizedPx). 0 if no depolymerized mask file was found."),
        ("NucCounts", "Number of distinct nuclei (8-connected components) in the raw, pre-dilation nuclei mask."),
        ("TotalNucArea", "Total nuclei pixel count in the raw, pre-dilation nuclei mask."),
        ("depolMaskAvailable", "True if a matching _depolMask.tif file was found; False if a blank mask was used."),
        ("nucleiMaskAvailable", "True if a matching _nucleiMask.tif file was found; False if a blank mask was used."),
        ("excludeDepolMask", "Parameter used: whether depolymerized-overlapping pixels were removed from the polymerized mask."),
        ("excludeNucleiMask", "Parameter used: whether nuclei-overlapping pixels were removed from the polymerized/depolymerized masks."),
        ("dilatePolymerizedPx", "Parameter used: number of 3x3-structuring-element dilation iterations applied to the polymerized mask."),
        ("dilateDepolymerizedPx", "Parameter used: number of 3x3-structuring-element dilation iterations applied to the depolymerized mask."),
        ("dilateNucleiPx", "Parameter used: number of 3x3-structuring-element dilation iterations applied to the nuclei mask."),
        ("overlay_saved_to", "File path of the saved RGB overlay PNG (depolymerized=red, polymerized=green, nuclei=cyan)."),
        ("error", "Populated only if this image failed to process; contains the exception message."),
    ]

    return pd.DataFrame(descriptions, columns=["Column", "Description"])


# %% Step 10: batch processing

def process_images(file_paths, params):
    """
    Processes one or multiple *_polSkeletonMask.tif images.

    For every image (single or batch):
        - the RGB overlay PNG is saved next to the source masks as
          <base>_overlay.png
    For a single-image run, the overlay is also displayed inline.
    """

    summary_rows = []
    last_intermediates = None
    last_base_name = None

    total_files = len(file_paths)

    for idx, pol_mask_path in enumerate(file_paths, start=1):

        image_name = os.path.basename(pol_mask_path)
        image_dir = os.path.dirname(pol_mask_path) or os.getcwd()

        print(f"\nProcessing ({idx}/{total_files}): {image_name}")
        print(
            "  Parameters: "
            f"excludeDepolMask={params['excludeDepolMask']}, "
            f"excludeNucleiMask={params['excludeNucleiMask']}, "
            f"dilatePolymerizedPx={params['dilatePolymerizedPx']}, "
            f"dilateDepolymerizedPx={params['dilateDepolymerizedPx']}, "
            f"dilateNucleiPx={params['dilateNucleiPx']}"
        )

        try:
            row, intermediates = process_single_image(pol_mask_path, params)

            base_name, depol_mask_path, nuclei_mask_path = derive_related_paths(pol_mask_path)

            if not row["depolMaskAvailable"]:
                print(f"  NOTE: depolymerized mask not found ({os.path.basename(depol_mask_path)}); using blank mask.")

            if not row["nucleiMaskAvailable"]:
                print(f"  NOTE: nuclei mask not found ({os.path.basename(nuclei_mask_path)}); using blank mask.")

            print(f"  NucCounts: {row['NucCounts']}, TotalNucArea: {row['TotalNucArea']}")
            print(f"  neuriteLength_raw: {row['neuriteLength_raw']}")
            print(f"  polymerizedMT_raw: {row['polymerizedMT_raw']}")
            print(f"  polymerizedMT_dilated: {row['polymerizedMT_dilated']}, depolMT_dilated: {row['depolMT_dilated']}")

            overlay_path = os.path.join(image_dir, base_name + "_overlay.png")
            plt.imsave(overlay_path, intermediates["rgb"])

            row["overlay_saved_to"] = overlay_path

            print(f"  Overlay saved: {overlay_path}")

            summary_rows.append(row)

            if total_files == 1:
                last_intermediates = intermediates
                last_base_name = base_name

        except Exception as e:

            print(f"  ERROR processing {image_name}: {e}")

            summary_rows.append({
                "image": image_name,
                "neuriteLength_raw": np.nan,
                "polymerizedMT_raw": np.nan,
                "polymerizedMT_dilated": np.nan,
                "depolMT_dilated": np.nan,
                "NucCounts": np.nan,
                "TotalNucArea": np.nan,
                "depolMaskAvailable": np.nan,
                "nucleiMaskAvailable": np.nan,
                "excludeDepolMask": params["excludeDepolMask"],
                "excludeNucleiMask": params["excludeNucleiMask"],
                "dilatePolymerizedPx": params["dilatePolymerizedPx"],
                "dilateDepolymerizedPx": params["dilateDepolymerizedPx"],
                "dilateNucleiPx": params["dilateNucleiPx"],
                "error": str(e),
            })

    return summary_rows, last_intermediates, last_base_name


# %% Step 11: main

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

    summary_rows, last_intermediates, last_base_name = process_images(file_paths, params)

    summary_df = pd.DataFrame(summary_rows)

    print("\n=== Summary (includes parameters used) ===")
    print(summary_df.to_string(index=False))

    summary_xlsx_path = os.path.join(
        os.path.dirname(file_paths[0]) or os.getcwd(),
        "mask_combination_summary.xlsx",
    )

    column_descriptions_df = build_column_descriptions()

    with pd.ExcelWriter(summary_xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        column_descriptions_df.to_excel(writer, sheet_name="Column Descriptions", index=False)

    print(f"\nSummary saved to:\n{summary_xlsx_path}")
    print("  (includes a 'Column Descriptions' sheet explaining each column)")

    if len(file_paths) > 1:

        successful = int(summary_df["polymerizedMT_raw"].notna().sum())
        failed = len(summary_df) - successful

        message = (
            "Processing complete.\n\n"
            f"Images selected: {len(file_paths)}\n"
            f"Successfully processed: {successful}\n"
            f"Failed: {failed}\n\n"
            "Overlay PNGs were saved next to their source masks.\n\n"
            "Summary Excel file (with a Column Descriptions sheet):\n"
            f"{summary_xlsx_path}"
        )

        sg.popup(message, title="Mask Combination Complete")

    elif len(file_paths) == 1 and last_intermediates is not None:

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

        axes[0].imshow(last_intermediates["pol_mask_dilated"], cmap="gray")
        axes[0].set_title("Polymerized (final)")
        axes[0].axis("off")

        axes[1].imshow(last_intermediates["depol_mask_dilated"], cmap="gray")
        axes[1].set_title("Depolymerized (final)")
        axes[1].axis("off")

        axes[2].imshow(last_intermediates["nuclei_mask_dilated"], cmap="gray")
        axes[2].set_title("Nuclei (final)")
        axes[2].axis("off")

        axes[3].imshow(last_intermediates["rgb"])
        axes[3].set_title("Overlay (depol=red, pol=green, nuclei=cyan)")
        axes[3].axis("off")

        fig.suptitle(last_base_name)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
