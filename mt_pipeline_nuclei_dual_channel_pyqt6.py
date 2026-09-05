# =====================================================================================
# Dual-Channel Nuclei Detection Pipeline (PyQt6, batch-capable)
# =====================================================================================
#
# New version of mt_pipeline_nuclei_dual_channel.py. Changes from that file:
#   1. PyQt6 instead of PySimpleGUI for all dialogs.
#   2. Multiple-file selection supported (processed in a batch loop).
#   3. Number inputs use PyQt6 spin boxes (QSpinBox / QDoubleSpinBox).
#   4. No per-step prints during processing -- just the image name being
#      processed -- with a summary table printed once at the end for all
#      images.
#   5. The final nuclei (AND) mask is saved as a TIFF next to each original
#      image.
#   6. The sanity-check plot is only shown when exactly one image was
#      selected; skipped entirely for batch runs.
#
# All core detection logic (rolling-ball background flattening, Otsu/fixed
# threshold, area+circularity filtering, h-maxima/watershed cluster
# splitting, object-level 50%-overlap AND) is unchanged from the previous
# version.
#
# DEPENDENCIES (install what you don't already have):
#   conda install -c conda-forge "scikit-image>=0.21" tifffile numpy scipy matplotlib pandas
#   pip install PyQt6
#
#   NOTE: restoration.rolling_ball() and morphology.h_maxima() both require
#   scikit-image >= 0.21. Update if you hit an AttributeError/ImportError.
# =====================================================================================

# %% Imports

import sys
import json
import os
import numpy as np
import pandas as pd                                   # conda install -c conda-forge pandas
import tifffile                                        # conda install -c conda-forge tifffile
from scipy.ndimage import gaussian_filter              # conda install -c conda-forge scipy
from skimage import measure, restoration, morphology, segmentation, filters  # conda install -c conda-forge "scikit-image>=0.21"
import matplotlib.pyplot as plt                         # conda install -c conda-forge matplotlib

from PyQt6.QtWidgets import (                            # pip install PyQt6
    QApplication, QFileDialog, QDialog, QFormLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox,
)


# %% Step 1: file picker + parameter dialog (PyQt6)

# -------------------- persistent settings file --------------------
# Shared across all pipeline scripts.
SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".mt_pipeline_settings.json")

def load_settings():
    defaults = {
        "nucChannel1": 1,
        "nucChannel2": 2,
        "NUCradius": 11,
        "NUCthreshold1": 0,   # 0 = auto (Otsu)
        "NUCthreshold2": 0,   # 0 = auto (Otsu)
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                defaults.update(json.load(f))
        except Exception:
            pass  # fall back to defaults if file is corrupt/missing keys
    return defaults

def save_settings(settings):
    """Merges into the shared settings file rather than overwriting it, so
    parameters saved by the other pipeline scripts aren't wiped out."""
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


# QApplication.instance() check avoids a crash if a Qt app already exists in
# this process (e.g. re-running the script in the same Spyder/IPython kernel).
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


# -------------------- pick input TIF(s) --------------------
file_paths, _ = QFileDialog.getOpenFileNames(
    None,
    "Select TIF image(s)",
    "",
    "TIFF files (*.tif *.tiff);;All files (*.*)",
)

if not file_paths:
    print("No files selected. Exiting.")
    sys.exit()

print(f"Selected {len(file_paths)} file(s).")


# -------------------- parameter dialog --------------------
class ParamDialog(QDialog):
    def __init__(self, defaults):
        super().__init__()
        self.setWindowTitle("Dual-Channel Nuclei Detection Parameters")
        layout = QFormLayout(self)

        info = QLabel(
            "Runs nuclei detection independently on two channels, then keeps\n"
            "nuclei (full shape) that overlap the other channel's mask by at\n"
            "least 50% of their own area. Same radius/threshold used for both."
        )
        layout.addRow(info)

        self.nucChannel1 = QSpinBox()
        self.nucChannel1.setRange(1, 20)
        self.nucChannel1.setValue(defaults["nucChannel1"])

        self.nucChannel2 = QSpinBox()
        self.nucChannel2.setRange(1, 20)
        self.nucChannel2.setValue(defaults["nucChannel2"])

        self.NUCradius = QSpinBox()
        self.NUCradius.setRange(1, 1000)
        self.NUCradius.setValue(defaults["NUCradius"])

        self.NUCthreshold1 = QDoubleSpinBox()
        self.NUCthreshold1.setRange(0, 65535)
        self.NUCthreshold1.setDecimals(2)
        self.NUCthreshold1.setValue(defaults["NUCthreshold1"])
        self.NUCthreshold1.setSpecialValueText("0 (auto Otsu)")

        self.NUCthreshold2 = QDoubleSpinBox()
        self.NUCthreshold2.setRange(0, 65535)
        self.NUCthreshold2.setDecimals(2)
        self.NUCthreshold2.setValue(defaults["NUCthreshold2"])
        self.NUCthreshold2.setSpecialValueText("0 (auto Otsu)")

        layout.addRow("Nuclei channel 1 (1-based):", self.nucChannel1)
        layout.addRow("Nuclei channel 2 (1-based):", self.nucChannel2)
        layout.addRow("Nuclear radius (px):", self.NUCradius)
        layout.addRow("Threshold - channel 1 (0 = auto Otsu):", self.NUCthreshold1)
        layout.addRow("Threshold - channel 2 (0 = auto Otsu):", self.NUCthreshold2)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)

    def get_values(self):
        return {
            "nucChannel1": self.nucChannel1.value(),
            "nucChannel2": self.nucChannel2.value(),
            "NUCradius": self.NUCradius.value(),
            "NUCthreshold1": self.NUCthreshold1.value(),
            "NUCthreshold2": self.NUCthreshold2.value(),
        }


settings = load_settings()
dialog = ParamDialog(settings)

if dialog.exec() == QDialog.DialogCode.Accepted:
    params = dialog.get_values()
else:
    print("No parameters entered. Exiting.")
    sys.exit()

save_settings(params)  # persist for next run

# unpack into individual variables, matching the macro's naming
nucChannel1   = params["nucChannel1"]
nucChannel2   = params["nucChannel2"]
NUCradius     = params["NUCradius"]
NUCthreshold1 = params["NUCthreshold1"]
NUCthreshold2 = params["NUCthreshold2"]

print("Parameters:")
for k, v in params.items():
    print(f"  {k}: {v}")


# %% Step 2: channel extraction

def get_channel(image, channel_index_1based):
    """
    Extract a single 2D channel from a possibly multi-dimensional hyperstack,
    mirroring Fiji's Stack.setChannel(). Uses a size heuristic to find the
    channel axis: channel counts are small (e.g. 1-5), while Y/X are large,
    so the channel axis is assumed to be the smallest non-spatial dimension.
    NOTE: verify this against your actual tif shape -- revisit this
    heuristic if your hyperstack has an unusual axis order.
    """
    arr = np.asarray(image)
    ch = channel_index_1based - 1  # 1-based -> 0-based

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        c_axis = int(np.argmin(arr.shape))  # assume Y,X >> channel count
        return np.take(arr, ch, axis=c_axis)

    if arr.ndim >= 4:
        spatial_axes = (arr.ndim - 2, arr.ndim - 1)  # assume last two = Y,X
        candidate_axes = list(range(arr.ndim - 2))
        if not candidate_axes:
            raise ValueError(f"Cannot determine channel axis for shape {arr.shape}")
        c_axis = min(candidate_axes, key=lambda a: arr.shape[a])
        idx = [0] * arr.ndim
        idx[c_axis] = ch
        idx[spatial_axes[0]] = slice(None)
        idx[spatial_axes[1]] = slice(None)
        return arr[tuple(idx)]

    raise ValueError(f"Unsupported image shape: {arr.shape}")


# %% Step 3: nuclei detection (single channel) + object-level AND

# -------------------- background flattening (rolling ball only) --------------------
def rolling_ball_background_subtract(image, radius):
    """
    Estimates and removes a spatially varying background using the
    rolling-ball algorithm (Sternberg's algorithm -- the same method behind
    ImageJ's "Subtract Background"). Returns a non-negative float image.
    """
    img = image.astype(np.float64)
    background = restoration.rolling_ball(img, radius=radius)
    return np.clip(img - background, 0, None)


# -------------------- shape helper --------------------
def compute_circularity(perimeter, area):
    perimeter = perimeter if perimeter > 0 else 1e-9
    return min(4 * np.pi * area / (perimeter ** 2), 1.0)


# -------------------- modal area (binned mode) --------------------
def compute_modal_area(areas, bin_size):
    """
    Approximates the mode of a set of areas via histogram binning. Python
    port of the macro's getBinnedMode(): finds the bin with the highest
    count and returns its center as the approximate modal value.
    """
    if len(areas) == 0:
        return float("nan")
    areas = np.asarray(areas, dtype=np.float64)
    min_val, max_val = areas.min(), areas.max()
    if bin_size <= 0:
        bin_size = 1.0
    n_bins = max(int(np.ceil((max_val - min_val) / bin_size)) + 1, 1)
    counts, _ = np.histogram(areas, bins=n_bins, range=(min_val, min_val + n_bins * bin_size))
    mode_bin = int(np.argmax(counts))
    return min_val + (mode_bin + 0.5) * bin_size


# -------------------- results table from a labeled mask --------------------
def build_results_df(labels):
    rows = []
    for p in measure.regionprops(labels):
        rows.append({
            "area": p.area,
            "centroid_row": p.centroid[0],
            "centroid_col": p.centroid[1],
            "perimeter": p.perimeter,
            "circularity": compute_circularity(p.perimeter, p.area),
        })
    cols = ["area", "centroid_row", "centroid_col", "perimeter", "circularity"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# -------------------- single-channel nuclei detection --------------------
def detect_nuclei(nuc_img, NUCradius, NUCthreshold, verbose=False):
    """
    Python port of nuclei_segmentation.ijm:
      1. Rolling-ball background flattening.
      2. Otsu (or fixed) threshold -> initial binary mask.
      3. Area + circularity filtered particle detection.
      4. Objects > 3x expected single-nucleus area flagged as clusters.
      5. Clusters peak-split via h-maxima + intensity-marker watershed,
         with prominence set adaptively from the foreground noise SD.
      6. Recombine, optionally drop border-touching nuclei, measure.

    Returns (nuclei_mask, nuclei_labels, results_df). Step-by-step prints
    are gated behind verbose (default off, for quiet batch processing).
    """
    bgRollingRadius  = NUCradius * 2                    # rolling-ball radius for background flattening
    seedBlurSigma    = NUCradius / 6                    # smoothing before peak-finding (limits over-splitting)
    minArea          = np.pi * (NUCradius * 0.5) ** 2   # discard fragments smaller than this (px^2)
    expectedArea     = np.pi * NUCradius ** 2            # area of one "typical" nucleus
    clusterFactor    = 3.0                               # objects > clusterFactor x expectedArea get split
    minCirc          = 0.10                              # loose circularity filter, drops fibers/debris
    prominenceFactor = 1.0                               # x noise SD; controls watershed seed sensitivity
    excludeEdges     = False                             # set True to drop nuclei touching the image border
    clusterThresh    = expectedArea * clusterFactor

    if verbose:
        print(f"  radius={NUCradius}  bgRollingRadius={bgRollingRadius}  seedBlurSigma={seedBlurSigma:.2f}")
        print(f"  expectedArea={expectedArea:.1f}  clusterThresh={clusterThresh:.1f}")

    empty_df = pd.DataFrame(columns=["area", "centroid_row", "centroid_col", "perimeter", "circularity"])

    # ---- 1. flatten background (rolling ball only, no flat-field) ----
    flattened = rolling_ball_background_subtract(nuc_img, bgRollingRadius)

    # ---- 2. threshold ----
    if NUCthreshold == 0:
        thresh_val = filters.threshold_otsu(flattened)
        if verbose:
            print(f"  threshold=auto (Otsu) -> {thresh_val:.2f}")
    else:
        thresh_val = NUCthreshold
        if verbose:
            print(f"  threshold={thresh_val} (user-supplied)")

    mask = flattened > thresh_val

    if not mask.any():
        if verbose:
            print("  No nuclei detected above threshold.")
        return mask, np.zeros(mask.shape, dtype=np.int32), empty_df

    # ---- 3. seed image, used later only for splitting oversized clusters ----
    seed_img = gaussian_filter(flattened, sigma=seedBlurSigma)

    noise_sd = float(seed_img[mask].std())
    prominence = noise_sd * prominenceFactor
    if prominence <= 0:
        prominence = 10.0
    if verbose:
        print(f"  prominence={prominence:.2f} (noiseSD={noise_sd:.2f})")

    # ---- 4. initial object finding (size + circularity filtered) ----
    labeled_initial = measure.label(mask, connectivity=2)
    normal_masks = []
    cluster_masks = []

    for p in measure.regionprops(labeled_initial):
        circ = compute_circularity(p.perimeter, p.area)
        if p.area < minArea or circ < minCirc:
            continue  # discard debris
        blob_mask = labeled_initial == p.label
        if p.area > clusterThresh:
            cluster_masks.append(blob_mask)
        else:
            normal_masks.append(blob_mask)

    if verbose:
        print(f"  Initial objects found: {len(normal_masks) + len(cluster_masks)}")
        print(f"  Objects flagged as clusters (> {clusterFactor}x expected area): {len(cluster_masks)}")

    # ---- 5. peak-split each flagged cluster, isolated from the rest of the mask ----
    split_masks = []
    for blob_mask in cluster_masks:
        blob_seed = np.where(blob_mask, seed_img, 0)
        maxima = morphology.h_maxima(blob_seed, h=prominence) & blob_mask
        markers = measure.label(maxima, connectivity=2)

        if markers.max() == 0:
            split_masks.append(blob_mask)  # no distinguishable peaks -- keep as one object
            continue

        ws_labels = segmentation.watershed(-blob_seed, markers=markers, mask=blob_mask)

        for lbl in range(1, ws_labels.max() + 1):
            sub_mask = ws_labels == lbl
            area = int(sub_mask.sum())
            if area < minArea:
                continue
            perim = measure.regionprops(sub_mask.astype(np.uint8))[0].perimeter
            if compute_circularity(perim, area) < minCirc:
                continue
            split_masks.append(sub_mask)

    # ---- 6. recombine ----
    final_masks = normal_masks + split_masks

    # ---- 7. optionally drop nuclei touching the image border ----
    if excludeEdges:
        h, w = mask.shape
        final_masks = [
            m for m in final_masks
            if not (np.nonzero(m)[0].min() <= 0 or np.nonzero(m)[1].min() <= 0
                    or np.nonzero(m)[0].max() >= h - 1 or np.nonzero(m)[1].max() >= w - 1)
        ]

    if verbose:
        print(f"  Nuclei detected (final): {len(final_masks)}")

    # ---- 8. build labeled image + combined mask + results table ----
    nuclei_labels = np.zeros(mask.shape, dtype=np.int32)
    for i, m in enumerate(final_masks, start=1):
        nuclei_labels[m] = i

    results_df = build_results_df(nuclei_labels)
    nuclei_mask = nuclei_labels > 0

    return nuclei_mask, nuclei_labels, results_df


# -------------------- object-level AND --------------------
def object_level_and(labels_a, mask_b, min_overlap_frac):
    """
    Keeps each object in labels_a at its FULL original shape if at least
    min_overlap_frac of that object's own area overlaps with mask_b.
    Returns a boolean mask (union of all qualifying full-shape objects).
    """
    kept = np.zeros(labels_a.shape, dtype=bool)
    for p in measure.regionprops(labels_a):
        obj_mask = labels_a == p.label
        overlap_pixels = np.count_nonzero(obj_mask & mask_b)
        if p.area > 0 and (overlap_pixels / p.area) >= min_overlap_frac:
            kept |= obj_mask
    return kept


# keep whole nuclei (from either channel) that overlap the other channel's
# mask by at least this fraction of their own area
minOverlapFraction = 0.5


# %% Step 4: batch processing loop

summary_rows = []
last_result = None  # only used for the single-image sanity-check plot

for idx, image_path in enumerate(file_paths, start=1):
    image_name = os.path.basename(image_path)
    image_dir = os.path.dirname(image_path)
    print(f"Processing ({idx}/{len(file_paths)}): {image_name}")

    image = tifffile.imread(image_path)
    nuc_img1 = get_channel(image, nucChannel1)
    nuc_img2 = get_channel(image, nucChannel2)

    mask1, labels1, df1 = detect_nuclei(nuc_img1, NUCradius, NUCthreshold1, verbose=False)
    mask2, labels2, df2 = detect_nuclei(nuc_img2, NUCradius, NUCthreshold2, verbose=False)

    kept_from_1 = object_level_and(labels1, mask2, minOverlapFraction)
    kept_from_2 = object_level_and(labels2, mask1, minOverlapFraction)
    and_mask = kept_from_1 | kept_from_2
    and_labels = measure.label(and_mask, connectivity=2)
    and_df = build_results_df(and_labels)

    nuc_count = len(and_df)
    nuc_total_area = float(and_mask.sum())
    nuc_modal_area = (
        compute_modal_area(and_df["area"].values, NUCradius / 5) if nuc_count > 0 else float("nan")
    )

    # save the final nuclei mask next to the original image (0/255 uint8 for
    # standard TIFF-viewer/ImageJ compatibility)
    mask_out_name = os.path.splitext(image_name)[0] + "_nucleiMask.tif"
    mask_out_path = os.path.join(image_dir, mask_out_name)
    tifffile.imwrite(mask_out_path, (and_mask.astype(np.uint8) * 255))

    summary_rows.append({
        "image": image_name,
        "channel1_count": len(df1),
        "channel2_count": len(df2),
        "and_count": nuc_count,
        "and_total_area": nuc_total_area,
        "and_modal_area": nuc_modal_area,
        "mask_saved_to": mask_out_path,
    })

    if len(file_paths) == 1:
        last_result = {
            "nuc_img1": nuc_img1,
            "nuc_img2": nuc_img2,
            "and_mask": and_mask,
            "nuc_count": nuc_count,
        }


# %% Step 5: summary table

summary_df = pd.DataFrame(summary_rows)
print("\n=== Summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(file_paths[0], "nuclei_summary.csv"), index=False)

# %% Step 6: sanity check (single image only)

if len(file_paths) == 1 and last_result is not None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(last_result["nuc_img1"], cmap='gray')
    axes[0].set_title(f'Channel {nucChannel1} (raw)')
    axes[0].axis('off')

    axes[1].imshow(last_result["nuc_img2"], cmap='gray')
    axes[1].set_title(f'Channel {nucChannel2} (raw)')
    axes[1].axis('off')

    axes[2].imshow(last_result["and_mask"], cmap='gray')
    axes[2].set_title(f'AND mask (n={last_result["nuc_count"]})')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
else:
    print("\nMultiple images processed -- skipping sanity check plot.")
