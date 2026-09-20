# =====================================================================================
# Single-Channel Nuclei Detection Pipeline
# PySimpleGUI, batch-capable
# Fiji-style Moments/Otsu thresholding + improved EDM watershed
# =====================================================================================
#
# FEATURES
#
#   1. Single nuclei channel.
#   2. PySimpleGUI interface.
#   3. File selection and all parameters are in the SAME window.
#   4. Remembers the last image selected.
#   5. Multiple TIF/TIFF files can be selected for batch processing.
#   6. Threshold options:
#        -1 = Fiji Moments threshold
#         0 = Fiji Otsu threshold
#        >0 = manual threshold
#   7. Reports the actual threshold used for each image.
#   8. Cluster detection based on:
#        expectedArea * clusterFactor
#   9. Minimum object filtering based on:
#        expectedArea * minAreaFactor
#  10. Improved EDM watershed for oversized nuclear clusters.
#  11. Watershed boundaries are explicitly preserved.
#  12. Minimum-area filtering is performed AGAIN after watershed.
#  13. Single-pixel stripe removal is performed after watershed.
#  14. Final nuclei mask is saved as TIFF.
#  15. Summary CSV is saved next to the images.
#  16. Single-image sanity-check plot is displayed.
#  17. Advanced settings can be expanded/collapsed.
#
# DEPENDENCIES
#
# conda install -c conda-forge numpy scipy scikit-image tifffile pandas matplotlib
# pip install PySimpleGUI
#
# =====================================================================================


# %% Imports

import time
import os
import sys
import json

import numpy as np
import pandas as pd
import tifffile
import PySimpleGUI as sg
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_dilation

from skimage import measure
from skimage import restoration
from skimage import segmentation
from skimage.feature import peak_local_max


# =====================================================================================
# SETTINGS
# =====================================================================================

SETTINGS_FILE = os.path.join(
    os.path.expanduser("~"),
    ".mt_pipeline_settings.json"
)


def load_settings():

    defaults = {
        "last_nuclei_files": [],
        "nucChannel": 1,
        "NUCradius": 9,
        "NUCthreshold": -1.0,
        "clusterFactor": 3.0,
        "minAreaFactor": 0.5,
        "fijiWatershedTolerance": 0.5,
        "stripeIterations": 3,
        "stripeHorizontal": True,
        "stripeVertical": True,
        "stripeDiagonal": False,
        "minCirc": 0.10,
        "excludeEdges": False
    }

    if os.path.exists(SETTINGS_FILE):

        try:

            with open(
                SETTINGS_FILE,
                "r"
            ) as f:

                saved = json.load(f)

            defaults.update(saved)

        except Exception:
            pass

    return defaults


def save_settings(settings):

    existing = {}

    if os.path.exists(SETTINGS_FILE):

        try:

            with open(
                SETTINGS_FILE,
                "r"
            ) as f:

                existing = json.load(f)

        except Exception:
            pass

    existing.update(settings)

    try:

        with open(
            SETTINGS_FILE,
            "w"
        ) as f:

            json.dump(
                existing,
                f,
                indent=2
            )

    except Exception:
        pass


settings = load_settings()


# =====================================================================================
# CHANNEL EXTRACTION
# =====================================================================================

def get_channel(
    image,
    channel_index_1based
):

    arr = np.asarray(image)

    ch = channel_index_1based - 1

    # -------------------------------------------------------------------------
    # 2D IMAGE
    # -------------------------------------------------------------------------

    if arr.ndim == 2:

        if ch != 0:

            raise ValueError(
                f"Image is 2D, so only channel 1 is available. "
                f"Requested channel {channel_index_1based}."
            )

        return arr

    # -------------------------------------------------------------------------
    # 3D IMAGE
    # -------------------------------------------------------------------------

    if arr.ndim == 3:

        c_axis = int(
            np.argmin(arr.shape)
        )

        if ch >= arr.shape[c_axis]:

            raise ValueError(
                f"Requested channel {channel_index_1based}, "
                f"but only {arr.shape[c_axis]} channels were detected."
            )

        return np.take(
            arr,
            ch,
            axis=c_axis
        )

    # -------------------------------------------------------------------------
    # 4D+ IMAGE
    # -------------------------------------------------------------------------

    if arr.ndim >= 4:

        spatial_axes = (
            arr.ndim - 2,
            arr.ndim - 1
        )

        candidate_axes = list(
            range(arr.ndim - 2)
        )

        if not candidate_axes:

            raise ValueError(
                f"Cannot determine channel axis for image shape {arr.shape}"
            )

        c_axis = min(
            candidate_axes,
            key=lambda a: arr.shape[a]
        )

        if ch >= arr.shape[c_axis]:

            raise ValueError(
                f"Requested channel {channel_index_1based}, "
                f"but only {arr.shape[c_axis]} channels were detected."
            )

        idx = [0] * arr.ndim

        idx[c_axis] = ch

        idx[spatial_axes[0]] = slice(None)
        idx[spatial_axes[1]] = slice(None)

        return arr[
            tuple(idx)
        ]

    raise ValueError(
        f"Unsupported image shape: {arr.shape}"
    )


# =====================================================================================
# BACKGROUND SUBTRACTION
# =====================================================================================

def rolling_ball_background_subtract(
    image,
    radius
):

    img = image.astype(
        np.float64
    )

    background = restoration.rolling_ball(
        img,
        radius=radius
    )

    return np.clip(
        img - background,
        0,
        None
    )


# =====================================================================================
# CIRCULARITY
# =====================================================================================

def compute_circularity(
    perimeter,
    area
):

    if perimeter <= 0:

        return 0.0

    value = (
        4.0
        * np.pi
        * area
        / (perimeter ** 2)
    )

    return min(
        value,
        1.0
    )


# =====================================================================================
# MODAL AREA
# =====================================================================================

def compute_modal_area(
    areas,
    bin_size
):

    if len(areas) == 0:

        return float("nan")

    areas = np.asarray(
        areas,
        dtype=np.float64
    )

    min_val = areas.min()
    max_val = areas.max()

    if bin_size <= 0:

        bin_size = 1.0

    n_bins = max(
        int(
            np.ceil(
                (max_val - min_val)
                / bin_size
            )
        ) + 1,
        1
    )

    counts, _ = np.histogram(
        areas,
        bins=n_bins,
        range=(
            min_val,
            min_val + n_bins * bin_size
        )
    )

    mode_bin = int(
        np.argmax(counts)
    )

    return (
        min_val
        + (mode_bin + 0.5) * bin_size
    )


# =====================================================================================
# RESULTS TABLE
# =====================================================================================

def build_results_df(
    labels
):

    rows = []

    for p in measure.regionprops(
        labels
    ):

        rows.append({

            "area":
                p.area,

            "centroid_row":
                p.centroid[0],

            "centroid_col":
                p.centroid[1],

            "perimeter":
                p.perimeter,

            "circularity":
                compute_circularity(
                    p.perimeter,
                    p.area
                )

        })

    cols = [

        "area",

        "centroid_row",

        "centroid_col",

        "perimeter",

        "circularity"

    ]

    if rows:

        return pd.DataFrame(
            rows
        )

    return pd.DataFrame(
        columns=cols
    )


# =====================================================================================
# FIJI MOMENTS THRESHOLD
# =====================================================================================

def fiji_moments_threshold(
    image
):

    img = np.asarray(
        image
    )

    finite = img[
        np.isfinite(img)
    ]

    if finite.size == 0:

        return 0.0

    min_val = float(
        finite.min()
    )

    max_val = float(
        finite.max()
    )

    if max_val <= min_val:

        return min_val

    if np.issubdtype(
        image.dtype,
        np.integer
    ):

        if image.dtype == np.uint8:

            hist = np.bincount(
                image.astype(
                    np.uint8
                ).ravel(),
                minlength=256
            )

            offset = 0
            scale = 1.0

        elif image.dtype == np.uint16:

            hist_full = np.bincount(
                image.astype(
                    np.uint16
                ).ravel(),
                minlength=65536
            )

            first = np.nonzero(
                hist_full
            )[0][0]

            last = np.nonzero(
                hist_full
            )[0][-1]

            hist = hist_full[
                first:last + 1
            ]

            offset = first
            scale = 1.0

        else:

            hist, edges = np.histogram(
                finite,
                bins=256,
                range=(
                    min_val,
                    max_val
                )
            )

            offset = min_val

            scale = (
                max_val
                - min_val
            ) / 256.0

    else:

        hist, edges = np.histogram(
            finite,
            bins=256,
            range=(
                min_val,
                max_val
            )
        )

        offset = min_val

        scale = (
            max_val
            - min_val
        ) / 256.0

    hist = np.asarray(
        hist,
        dtype=np.float64
    )

    total = hist.sum()

    if total <= 0:

        return min_val

    histo = hist / total

    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    m0 = 1.0

    for i in range(
        len(histo)
    ):

        m1 += (
            i
            * histo[i]
        )

        m2 += (
            i
            * i
            * histo[i]
        )

        m3 += (
            i
            * i
            * i
            * histo[i]
        )

    cd = (
        m0 * m2
        - m1 * m1
    )

    if cd == 0:

        return min_val

    c0 = (
        -m2 * m2
        + m1 * m3
    ) / cd

    c1 = (
        -m0 * m3
        + m2 * m1
    ) / cd

    discriminant = (
        c1 * c1
        - 4.0 * c0
    )

    if discriminant < 0:

        return min_val

    sqrt_disc = np.sqrt(
        discriminant
    )

    z0 = 0.5 * (
        -c1
        - sqrt_disc
    )

    z1 = 0.5 * (
        -c1
        + sqrt_disc
    )

    denominator = (
        z1 - z0
    )

    if denominator == 0:

        return min_val

    p0 = (
        z1 - m1
    ) / denominator

    p0 = np.clip(
        p0,
        0.0,
        1.0
    )

    cumulative = 0.0

    threshold_bin = (
        len(histo) - 1
    )

    for i in range(
        len(histo)
    ):

        cumulative += histo[i]

        if cumulative > p0:

            threshold_bin = i

            break

    if np.issubdtype(
        image.dtype,
        np.integer
    ):

        return float(
            offset
            + threshold_bin
        )

    return float(
        offset
        + (
            threshold_bin
            + 0.5
        ) * scale
    )


# =====================================================================================
# FIJI OTSU THRESHOLD
# =====================================================================================

def fiji_otsu_threshold(
    image
):

    img = np.asarray(
        image
    )

    finite = img[
        np.isfinite(img)
    ]

    if finite.size == 0:

        return 0.0

    min_val = float(
        finite.min()
    )

    max_val = float(
        finite.max()
    )

    if max_val <= min_val:

        return min_val

    if np.issubdtype(
        img.dtype,
        np.integer
    ):

        if img.dtype == np.uint8:

            hist = np.bincount(
                img.astype(
                    np.uint8
                ).ravel(),
                minlength=256
            )

            offset = 0.0
            scale = 1.0

        elif img.dtype == np.uint16:

            hist = np.bincount(
                img.astype(
                    np.uint16
                ).ravel(),
                minlength=65536
            )

            offset = 0.0
            scale = 1.0

        else:

            hist, _ = np.histogram(
                finite,
                bins=256,
                range=(
                    min_val,
                    max_val
                )
            )

            offset = min_val

            scale = (
                max_val
                - min_val
            ) / 256.0

    else:

        hist, _ = np.histogram(
            finite,
            bins=256,
            range=(
                min_val,
                max_val
            )
        )

        offset = min_val

        scale = (
            max_val
            - min_val
        ) / 256.0

    hist = hist.astype(
        np.float64
    )

    total = hist.sum()

    if total <= 0:

        return min_val

    probability = (
        hist / total
    )

    cumulative_probability = (
        np.cumsum(
            probability
        )
    )

    cumulative_mean = (
        np.cumsum(
            probability
            * np.arange(
                len(hist)
            )
        )
    )

    total_mean = (
        cumulative_mean[-1]
    )

    between_class_variance = (
        np.zeros_like(
            cumulative_probability
        )
    )

    valid = (

        (
            cumulative_probability
            > 0
        )

        &

        (
            cumulative_probability
            < 1
        )

    )

    between_class_variance[
        valid
    ] = (

        (

            total_mean
            * cumulative_probability[
                valid
            ]

            -

            cumulative_mean[
                valid
            ]

        ) ** 2

        /

        (

            cumulative_probability[
                valid
            ]

            *

            (
                1.0
                - cumulative_probability[
                    valid
                ]
            )

        )

    )

    threshold_bin = int(
        np.argmax(
            between_class_variance
        )
    )

    if np.issubdtype(
        img.dtype,
        np.integer
    ):

        return float(
            offset
            + threshold_bin
        )

    return float(
        offset
        + (
            threshold_bin
            + 0.5
        ) * scale
    )


# =====================================================================================
# THRESHOLD SELECTION
# =====================================================================================

def determine_threshold(
    flattened,
    threshold_input
):

    if threshold_input < 0:

        threshold = (
            fiji_moments_threshold(
                flattened
            )
        )

        return (
            threshold,
            "Moments"
        )

    if threshold_input == 0:

        threshold = (
            fiji_otsu_threshold(
                flattened
            )
        )

        return (
            threshold,
            "Otsu"
        )

    return (
        float(threshold_input),
        "Manual"
    )


# =====================================================================================
# SINGLE-PIXEL STRIPE REMOVAL
# =====================================================================================

def remove_1px_stripes(
    mask,
    iterations=3,
    horizontal=True,
    vertical=True,
    diagonal=False
):

    m = mask.copy()

    for _ in range(
        iterations
    ):

        keep = np.ones_like(
            m,
            dtype=bool
        )

        # ---------------------------------------------------------------------
        # HORIZONTAL
        # ---------------------------------------------------------------------

        if horizontal:

            left = np.roll(
                m,
                1,
                axis=1
            )

            left[:, 0] = False

            right = np.roll(
                m,
                -1,
                axis=1
            )

            right[:, -1] = False

            keep &= (
                left
                | right
            )

        # ---------------------------------------------------------------------
        # VERTICAL
        # ---------------------------------------------------------------------

        if vertical:

            up = np.roll(
                m,
                1,
                axis=0
            )

            up[0, :] = False

            down = np.roll(
                m,
                -1,
                axis=0
            )

            down[-1, :] = False

            keep &= (
                up
                | down
            )

        # ---------------------------------------------------------------------
        # DIAGONAL
        # ---------------------------------------------------------------------

        if diagonal:

            upleft = np.roll(
                np.roll(
                    m,
                    1,
                    axis=0
                ),
                1,
                axis=1
            )

            upleft[0, :] = False
            upleft[:, 0] = False

            downright = np.roll(
                np.roll(
                    m,
                    -1,
                    axis=0
                ),
                -1,
                axis=1
            )

            downright[-1, :] = False
            downright[:, -1] = False

            upright = np.roll(
                np.roll(
                    m,
                    1,
                    axis=0
                ),
                -1,
                axis=1
            )

            upright[0, :] = False
            upright[:, -1] = False

            downleft = np.roll(
                np.roll(
                    m,
                    -1,
                    axis=0
                ),
                1,
                axis=1
            )

            downleft[-1, :] = False
            downleft[:, 0] = False

            keep &= (

                (
                    upleft
                    | downright
                )

                &

                (
                    upright
                    | downleft
                )

            )

        m = m & keep

    return m


# =====================================================================================
# IMPROVED EDM WATERSHED
# =====================================================================================

def fiji_edm_watershed(
    binary_mask,
    intensity_image,
    tolerance=0.5,
    min_area=0,
    nuc_radius=11
):
    """
    Improved EDM-based watershed for separating touching nuclei.

    The intensity image argument is retained for compatibility with the
    existing pipeline. The watershed itself is based on the Euclidean
    distance map of the binary nuclear object.

    Key points:

    1. Distance transform is calculated inside each cluster.
    2. Local maxima of the EDM are used as nuclear centers.
    3. Minimum peak separation scales with NUCradius.
    4. Small EDM peaks are rejected.
    5. watershed_line=True preserves the dividing ridge.
    6. Watershed regions remain physically separated in the final mask.
    """

    # -------------------------------------------------------------------------
    # EMPTY MASK
    # -------------------------------------------------------------------------

    if not binary_mask.any():

        return binary_mask.copy()

    # -------------------------------------------------------------------------
    # 1. EUCLIDEAN DISTANCE MAP
    # -------------------------------------------------------------------------

    distance = distance_transform_edt(
        binary_mask
    ).astype(
        np.float64
    )

    if distance.max() <= 0:

        return binary_mask.copy()

    # -------------------------------------------------------------------------
    # 2. VERY LIGHT EDM SMOOTHING
    #
    # sigma=0.5 suppresses pixel-level noise while preserving the shape
    # of the distance map.
    # -------------------------------------------------------------------------

    smooth_distance = gaussian_filter(
        distance,
        sigma=0.5
    )

    # -------------------------------------------------------------------------
    # 3. FIND EDM PEAKS
    #
    # With NUCradius = 11:
    #
    #     min_peak_distance = 6.6 px -> 7 px
    #
    # This prevents several nearby maxima from being interpreted as
    # separate nuclei while still allowing touching nuclei to have
    # independent centers.
    # -------------------------------------------------------------------------

    min_peak_distance = max(
        3,
        int(
            round(
                nuc_radius * 0.6
            )
        )
    )

    peak_coordinates = peak_local_max(
        smooth_distance,
        min_distance=min_peak_distance,
        labels=binary_mask,
        exclude_border=False
    )

    # -------------------------------------------------------------------------
    # If fewer than two peaks exist, there is nothing to split.
    # -------------------------------------------------------------------------

    if len(
        peak_coordinates
    ) < 2:

        return binary_mask.copy()

    # -------------------------------------------------------------------------
    # 4. GET PEAK HEIGHTS
    # -------------------------------------------------------------------------

    peak_values = smooth_distance[
        peak_coordinates[:, 0],
        peak_coordinates[:, 1]
    ]

    # -------------------------------------------------------------------------
    # 5. REMOVE VERY SMALL EDM PEAKS
    #
    # tolerance is retained as a user-controlled lower limit.
    #
    # The radius-dependent floor prevents tiny edge irregularities from
    # becoming artificial nuclei.
    #
    # For NUCradius = 11:
    #
    #     0.35 × 11 = 3.85 pixels
    #
    # Therefore peaks smaller than ~3.85 px are rejected.
    # -------------------------------------------------------------------------

    minimum_peak_height = max(
        float(tolerance),
        nuc_radius * 0.35
    )

    valid_peaks = (
        peak_values
        >= minimum_peak_height
    )

    peak_coordinates = (
        peak_coordinates[
            valid_peaks
        ]
    )

    # -------------------------------------------------------------------------
    # Again, require at least two valid peaks.
    # -------------------------------------------------------------------------

    if len(
        peak_coordinates
    ) < 2:

        return binary_mask.copy()

    # -------------------------------------------------------------------------
    # 6. CREATE WATERSHED MARKERS
    # -------------------------------------------------------------------------

    markers = np.zeros(
        binary_mask.shape,
        dtype=np.int32
    )

    for marker_number, coord in enumerate(
        peak_coordinates,
        start=1
    ):

        markers[
            coord[0],
            coord[1]
        ] = marker_number

    if markers.max() < 2:

        return binary_mask.copy()

    # -------------------------------------------------------------------------
    # 7. WATERSHED
    #
    # watershed_line=True is important.
    #
    # It leaves the dividing ridge at label 0 instead of assigning every
    # pixel to one of the nuclei.
    # -------------------------------------------------------------------------

    watershed_labels = segmentation.watershed(
        -distance,
        markers,
        mask=binary_mask,
        connectivity=1,
        watershed_line=True
    )

    # -------------------------------------------------------------------------
    # 8. PRESERVE WATERSHED BOUNDARIES
    #
    # Do NOT simply reconnect all positive pixels.
    #
    # We explicitly retain the zero-valued watershed ridge.
    # -------------------------------------------------------------------------

    result = np.zeros_like(
        binary_mask,
        dtype=bool
    )
    
    # Expand watershed boundary by 1 pixel on each side
    watershed_line = watershed_labels == 0
    watershed_line = binary_dilation(
        watershed_line,
        structure=np.ones((3, 3), dtype=bool)
    )
    
    labeled_regions = measure.label(
        watershed_labels,
        connectivity=1
    )

    for region in measure.regionprops(
        labeled_regions
    ):

        if region.area < min_area:

            continue

        result[
            labeled_regions == region.label
        ] = True

    return result


# =====================================================================================
# POST-WATERSHED MINIMUM AREA FILTER
# =====================================================================================

def minimum_area_filter(
    mask,
    min_area
):

    labeled_initial = measure.label(
        mask,
        connectivity=2
    )

    filtered_mask = np.zeros_like(
        mask,
        dtype=bool
    )

    for region in measure.regionprops(
        labeled_initial
    ):

        if region.area >= min_area:

            filtered_mask[
                labeled_initial == region.label
            ] = True

    return filtered_mask


# =====================================================================================
# NUCLEI DETECTION
# =====================================================================================

def detect_nuclei(
    nuc_img,
    NUCradius,
    NUCthreshold,
    clusterFactor,
    minAreaFactor,
    fijiWatershedTolerance,
    stripeIterations,
    stripeHorizontal,
    stripeVertical,
    stripeDiagonal,
    minCirc,
    excludeEdges,
    verbose=False
):

    # -------------------------------------------------------------------------
    # EXPECTED NUCLEAR PARAMETERS
    # -------------------------------------------------------------------------

    bgRollingRadius = (
        NUCradius * 2
    )

    expectedArea = (
        np.pi
        * NUCradius ** 2
    )

    minArea = (
        expectedArea
        * minAreaFactor
    )

    clusterThresh = (
        expectedArea
        * clusterFactor
    )

    # -------------------------------------------------------------------------
    # BACKGROUND SUBTRACTION
    # -------------------------------------------------------------------------

    flattened = (
        rolling_ball_background_subtract(
            nuc_img,
            bgRollingRadius
        )
    )

    # -------------------------------------------------------------------------
    # THRESHOLD
    # -------------------------------------------------------------------------

    final_threshold, threshold_method = (
        determine_threshold(
            flattened,
            NUCthreshold
        )
    )

    if verbose:

        print(
            f"  Threshold method: "
            f"{threshold_method}"
        )

        print(
            f"  Final threshold: "
            f"{final_threshold:.2f}"
        )

        print(
            f"  Expected nuclear area: "
            f"{expectedArea:.2f}"
        )

        print(
            f"  Minimum area: "
            f"{minArea:.2f}"
        )

        print(
            f"  Cluster threshold: "
            f"{clusterThresh:.2f}"
        )

    # -------------------------------------------------------------------------
    # INITIAL BINARY MASK
    # -------------------------------------------------------------------------

    mask = (
        flattened
        > final_threshold
    )

    empty_df = pd.DataFrame(
        columns=[
            "area",
            "centroid_row",
            "centroid_col",
            "perimeter",
            "circularity"
        ]
    )

    if not mask.any():

        return (

            mask,

            np.zeros(
                mask.shape,
                dtype=np.int32
            ),

            empty_df,

            final_threshold,

            threshold_method

        )

    # -------------------------------------------------------------------------
    # INITIAL CONNECTED COMPONENTS
    # -------------------------------------------------------------------------

    labeled_initial = measure.label(
        mask,
        connectivity=2
    )

    normal_masks = []
    cluster_masks = []

    for region in measure.regionprops(
        labeled_initial
    ):

        area = region.area

        circularity = (
            compute_circularity(
                region.perimeter,
                region.area
            )
        )

        # ---------------------------------------------------------------------
        # REMOVE SMALL OBJECTS
        # ---------------------------------------------------------------------

        if area < minArea:

            continue

        # ---------------------------------------------------------------------
        # CIRCULARITY FILTER
        # ---------------------------------------------------------------------

        if circularity < minCirc:

            continue

        # ---------------------------------------------------------------------
        # EXTRACT INDIVIDUAL OBJECT
        # ---------------------------------------------------------------------

        object_mask = (
            labeled_initial
            == region.label
        )

        # ---------------------------------------------------------------------
        # CLASSIFY AS NORMAL OBJECT OR CLUSTER
        # ---------------------------------------------------------------------

        if area > clusterThresh:

            cluster_masks.append(
                object_mask
            )

        else:

            normal_masks.append(
                object_mask
            )

    if verbose:

        print(
            f"  Initial normal objects: "
            f"{len(normal_masks)}"
        )

        print(
            f"  Initial clusters: "
            f"{len(cluster_masks)}"
        )

    # -------------------------------------------------------------------------
    # NORMAL OBJECTS ARE ALREADY ACCEPTED
    # -------------------------------------------------------------------------

    final_masks = list(
        normal_masks
    )

    # =========================================================================
    # EDM WATERSHED
    # =========================================================================

    for cluster_mask in cluster_masks:

        split_mask = (
            fiji_edm_watershed(
                cluster_mask,
                flattened,
                tolerance=fijiWatershedTolerance,
                min_area=minArea,
                nuc_radius=NUCradius
            )
        )

        # ---------------------------------------------------------------------
        # LABEL WATERSHED OUTPUT
        # ---------------------------------------------------------------------

        split_labels = measure.label(
            split_mask,
            connectivity=2
        )

        # ---------------------------------------------------------------------
        # FILTER EACH WATERSHED OBJECT
        # ---------------------------------------------------------------------

        for region in measure.regionprops(
            split_labels
        ):

            if region.area < minArea:

                continue

            object_mask = (
                split_labels
                == region.label
            )

            circularity = (
                compute_circularity(
                    region.perimeter,
                    region.area
                )
            )

            if circularity < minCirc:

                continue

            final_masks.append(
                object_mask
            )

    # =========================================================================
    # RECOMBINE AFTER WATERSHED
    # =========================================================================

    watershed_mask = np.zeros_like(
        mask,
        dtype=bool
    )

    for object_mask in final_masks:

        watershed_mask |= object_mask

    # =========================================================================
    # SECOND MINIMUM-AREA FILTER
    # =========================================================================

    watershed_mask = (
        minimum_area_filter(
            watershed_mask,
            minArea
        )
    )

    # =========================================================================
    # SINGLE-PIXEL STRIPE REMOVAL
    # =========================================================================

    watershed_mask = (
        remove_1px_stripes(
            watershed_mask,
            iterations=stripeIterations,
            horizontal=stripeHorizontal,
            vertical=stripeVertical,
            diagonal=stripeDiagonal
        )
    )

    # =========================================================================
    # MINIMUM AREA FILTER AGAIN
    # =========================================================================

    watershed_mask = (
        minimum_area_filter(
            watershed_mask,
            minArea
        )
    )

    # =========================================================================
    # OPTIONAL BORDER EXCLUSION
    # =========================================================================

    if excludeEdges:

        labeled_edges = measure.label(
            watershed_mask,
            connectivity=2
        )

        edge_filtered = np.zeros_like(
            watershed_mask,
            dtype=bool
        )

        h, w = watershed_mask.shape

        for region in measure.regionprops(
            labeled_edges
        ):

            coords = region.coords

            touches_edge = (

                np.any(
                    coords[:, 0] == 0
                )

                or

                np.any(
                    coords[:, 0] == h - 1
                )

                or

                np.any(
                    coords[:, 1] == 0
                )

                or

                np.any(
                    coords[:, 1] == w - 1
                )

            )

            if not touches_edge:

                edge_filtered[
                    labeled_edges
                    == region.label
                ] = True

        watershed_mask = (
            edge_filtered
        )

    # =========================================================================
    # FINAL LABELING
    # =========================================================================

    nuclei_labels = measure.label(
        watershed_mask,
        connectivity=2
    )

    results_df = (
        build_results_df(
            nuclei_labels
        )
    )

    nuclei_mask = (
        nuclei_labels > 0
    )

    if verbose:

        print(
            f"  Final nuclei count: "
            f"{len(results_df)}"
        )

    return (

        nuclei_mask,

        nuclei_labels,

        results_df,

        final_threshold,

        threshold_method

    )


# =====================================================================================
# PY SIMPLE GUI
# =====================================================================================

def make_window(
    defaults
):

    sg.theme(
        "SystemDefault"
    )

    last_files = defaults.get(
        "last_nuclei_files",
        []
    )

    if isinstance(
        last_files,
        str
    ):

        last_files = [
            last_files
        ]

    existing_files = [

        p

        for p in last_files

        if os.path.exists(p)

    ]

    default_file_string = (

        ";".join(
            existing_files
        )

        if existing_files

        else ""

    )

    advanced_layout = [

        [

            sg.Text(
                "Minimum area factor:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "minAreaFactor"
                    ]
                ),
                key="-MINAREAFACTOR-",
                size=(10, 1)
            ),

            sg.Text(
                "min area = expected area × factor"
            )

        ],

        [

            sg.Text(
                "Cluster area factor:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "clusterFactor"
                    ]
                ),
                key="-CLUSTERFACTOR-",
                size=(10, 1)
            ),

            sg.Text(
                "cluster if area > expected area × factor"
            )

        ],

        [

            sg.Text(
                "Fiji EDM tolerance:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "fijiWatershedTolerance"
                    ]
                ),
                key="-EDMTOL-",
                size=(10, 1)
            ),

            sg.Text(
                "minimum EDM peak height"
            )

        ],

        [

            sg.Text(
                "Minimum circularity:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "minCirc"
                    ]
                ),
                key="-MINCIRC-",
                size=(10, 1)
            ),

            sg.Text(
                "rejects very non-circular objects"
            )

        ],

        [

            sg.Text(
                "Stripe-removal iterations:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "stripeIterations"
                    ]
                ),
                key="-STRIPEITER-",
                size=(10, 1)
            ),

            sg.Text(
                "number of cleanup passes"
            )

        ],

        [

            sg.Checkbox(
                "Remove horizontal 1-pixel stripes",
                default=defaults[
                    "stripeHorizontal"
                ],
                key="-STRIPEH-"
            )

        ],

        [

            sg.Checkbox(
                "Remove vertical 1-pixel stripes",
                default=defaults[
                    "stripeVertical"
                ],
                key="-STRIPEV-"
            )

        ],

        [

            sg.Checkbox(
                "Remove diagonal 1-pixel stripes",
                default=defaults[
                    "stripeDiagonal"
                ],
                key="-STRIPED-"
            )

        ],

        [

            sg.Checkbox(
                "Exclude nuclei touching image edges",
                default=defaults[
                    "excludeEdges"
                ],
                key="-EXCLUDEEDGES-"
            )

        ]

    ]

    layout = [

        [

            sg.Text(
                "Input TIF/TIFF image(s):",
                size=(22, 1)
            ),

            sg.Input(
                default_file_string,
                key="-FILES-",
                size=(55, 1)
            ),

            sg.FilesBrowse(
                "Browse",
                file_types=(

                    (
                        "TIFF Files",
                        "*.tif;*.tiff"
                    ),

                    (
                        "All Files",
                        "*.*"
                    )

                )
            )

        ],

        [

            sg.Text(
                "Nuclei channel (1-based):",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "nucChannel"
                    ]
                ),
                key="-CHANNEL-",
                size=(10, 1)
            ),

            sg.Text(
                "1 = first channel"
            )

        ],

        [

            sg.Text(
                "Nuclear radius (px):",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "NUCradius"
                    ]
                ),
                key="-RADIUS-",
                size=(10, 1)
            ),

            sg.Text(
                "approximate nuclear radius"
            )

        ],

        [

            sg.Text(
                "Threshold input:",
                size=(27, 1)
            ),

            sg.Input(
                str(
                    defaults[
                        "NUCthreshold"
                    ]
                ),
                key="-THRESHOLD-",
                size=(10, 1)
            )

        ],

        [

            sg.Text(
                "Threshold instructions:",
                size=(27, 1)
            ),

            sg.Text(
                "-1 = Moments    0 = Otsu    >0 = manual"
            )

        ],

        [

            sg.HorizontalSeparator()

        ],

        [

            sg.pin(

                sg.Column(
                    advanced_layout,
                    key="-ADVANCED-",
                    visible=False
                )

            )

        ],

        [

            sg.Button(
                "Advanced Settings",
                key="-ADVBTN-"
            ),

            sg.Push(),

            sg.Button(
                "Run",
                bind_return_key=True
            ),

            sg.Button(
                "Cancel"
            )

        ]

    ]

    return sg.Window(

        "Single-Channel Nuclei Detection",

        layout,

        finalize=True,

        resizable=True

    )


# =====================================================================================
# GUI
# =====================================================================================

window = make_window(
    settings
)

advanced_visible = False

while True:

    event, values = window.read()

    if event in (

        sg.WIN_CLOSED,

        "Cancel"

    ):

        window.close()

        sys.exit()

    if event == "-ADVBTN-":

        advanced_visible = (
            not advanced_visible
        )

        window[
            "-ADVANCED-"
        ].update(
            visible=advanced_visible
        )

        window[
            "-ADVBTN-"
        ].update(

            "Hide Advanced Settings"

            if advanced_visible

            else "Advanced Settings"

        )

    if event == "Run":

        try:

            file_string = (
                values[
                    "-FILES-"
                ].strip()
            )

            if not file_string:

                sg.popup_error(
                    "Please select at least one TIF/TIFF image."
                )

                continue

            file_paths = [

                p

                for p in file_string.split(";")

                if p

            ]

            nucChannel = int(
                values[
                    "-CHANNEL-"
                ]
            )

            NUCradius = int(
                values[
                    "-RADIUS-"
                ]
            )

            NUCthreshold = float(
                values[
                    "-THRESHOLD-"
                ]
            )

            clusterFactor = float(
                values[
                    "-CLUSTERFACTOR-"
                ]
            )

            minAreaFactor = float(
                values[
                    "-MINAREAFACTOR-"
                ]
            )

            fijiWatershedTolerance = float(
                values[
                    "-EDMTOL-"
                ]
            )

            stripeIterations = int(
                values[
                    "-STRIPEITER-"
                ]
            )

            stripeHorizontal = bool(
                values[
                    "-STRIPEH-"
                ]
            )

            stripeVertical = bool(
                values[
                    "-STRIPEV-"
                ]
            )

            stripeDiagonal = bool(
                values[
                    "-STRIPED-"
                ]
            )

            minCirc = float(
                values[
                    "-MINCIRC-"
                ]
            )

            excludeEdges = bool(
                values[
                    "-EXCLUDEEDGES-"
                ]
            )

            # -----------------------------------------------------------------
            # VALIDATION
            # -----------------------------------------------------------------

            if nucChannel < 1:

                raise ValueError(
                    "Nuclei channel must be 1 or greater."
                )

            if NUCradius < 1:

                raise ValueError(
                    "Nuclear radius must be at least 1."
                )

            if clusterFactor <= 0:

                raise ValueError(
                    "Cluster area factor must be greater than 0."
                )

            if minAreaFactor <= 0:

                raise ValueError(
                    "Minimum area factor must be greater than 0."
                )

            if fijiWatershedTolerance < 0:

                raise ValueError(
                    "Fiji EDM tolerance cannot be negative."
                )

            if stripeIterations < 0:

                raise ValueError(
                    "Stripe iterations cannot be negative."
                )

            if minCirc < 0:

                raise ValueError(
                    "Minimum circularity cannot be negative."
                )

            # -----------------------------------------------------------------
            # SAVE SETTINGS
            # -----------------------------------------------------------------

            settings.update({

                "last_nuclei_files":
                    file_paths,

                "nucChannel":
                    nucChannel,

                "NUCradius":
                    NUCradius,

                "NUCthreshold":
                    NUCthreshold,

                "clusterFactor":
                    clusterFactor,

                "minAreaFactor":
                    minAreaFactor,

                "fijiWatershedTolerance":
                    fijiWatershedTolerance,

                "stripeIterations":
                    stripeIterations,

                "stripeHorizontal":
                    stripeHorizontal,

                "stripeVertical":
                    stripeVertical,

                "stripeDiagonal":
                    stripeDiagonal,

                "minCirc":
                    minCirc,

                "excludeEdges":
                    excludeEdges

            })

            save_settings(
                settings
            )

            break

        except Exception as e:

            sg.popup_error(
                "Invalid parameter:",
                str(e)
            )

window.close()


# =====================================================================================
# PRINT PARAMETERS
# =====================================================================================

print(
    f"Selected {len(file_paths)} file(s)."
)

threshold_method_input = (

    "Moments"

    if NUCthreshold < 0

    else

    "Otsu"

    if NUCthreshold == 0

    else

    "Manual"

)

print(
    "Parameters:"
)

print(
    f"  nucChannel: {nucChannel}"
)

print(
    f"  NUCradius: {NUCradius}"
)

print(
    f"  threshold method: "
    f"{threshold_method_input}"
)

print(
    f"  threshold input: "
    f"{NUCthreshold}"
)

print(
    f"  clusterFactor: "
    f"{clusterFactor}"
)

print(
    f"  minAreaFactor: "
    f"{minAreaFactor}"
)

print(
    f"  Fiji EDM tolerance: "
    f"{fijiWatershedTolerance}"
)

print(
    f"  minCirc: "
    f"{minCirc}"
)

print(
    f"  stripeIterations: "
    f"{stripeIterations}"
)

print(
    f"  stripeHorizontal: "
    f"{stripeHorizontal}"
)

print(
    f"  stripeVertical: "
    f"{stripeVertical}"
)

print(
    f"  stripeDiagonal: "
    f"{stripeDiagonal}"
)

print(
    f"  excludeEdges: "
    f"{excludeEdges}"
)


# =====================================================================================
# BATCH PROCESSING
# =====================================================================================

summary_rows = []

last_result = None

for idx, image_path in enumerate(
    file_paths,
    start=1
):

    image_name = os.path.basename(
        image_path
    )

    image_dir = os.path.dirname(
        image_path
    )

    print(
        f"\nProcessing ({idx}/{len(file_paths)}): "
        f"{image_name}"
    )

    try:

        # ---------------------------------------------------------------------
        # READ IMAGE
        # ---------------------------------------------------------------------

        image = tifffile.imread(
            image_path
        )

        # ---------------------------------------------------------------------
        # EXTRACT NUCLEAR CHANNEL
        # ---------------------------------------------------------------------

        nuc_img = get_channel(
            image,
            nucChannel
        )

        # ---------------------------------------------------------------------
        # DETECT NUCLEI
        # ---------------------------------------------------------------------

        (

            nuclei_mask,

            nuclei_labels,

            nuclei_df,

            final_threshold,

            threshold_method

        ) = detect_nuclei(

            nuc_img,

            NUCradius,

            NUCthreshold,

            clusterFactor,

            minAreaFactor,

            fijiWatershedTolerance,

            stripeIterations,

            stripeHorizontal,

            stripeVertical,

            stripeDiagonal,

            minCirc,

            excludeEdges,

            verbose=False

        )

        # ---------------------------------------------------------------------
        # NUCLEAR COUNT
        # ---------------------------------------------------------------------

        nuclei_count = len(
            nuclei_df
        )

        # ---------------------------------------------------------------------
        # TOTAL NUCLEAR AREA
        # ---------------------------------------------------------------------

        nuclei_total_area = float(
            nuclei_mask.sum()
        )

        # ---------------------------------------------------------------------
        # MODAL AREA
        # ---------------------------------------------------------------------

        nuclei_modal_area = (

            compute_modal_area(

                nuclei_df[
                    "area"
                ].values,

                NUCradius / 5.0

            )

            if nuclei_count > 0

            else float("nan")

        )

        # ---------------------------------------------------------------------
        # SAVE FINAL MASK
        # ---------------------------------------------------------------------

        mask_out_name = (

            os.path.splitext(
                image_name
            )[0]

            + "_nucleiMask.tif"

        )

        mask_out_path = os.path.join(

            image_dir,

            mask_out_name

        )

        tifffile.imwrite(

            mask_out_path,

            nuclei_mask.astype(
                np.uint8
            ) * 255

        )

        # ---------------------------------------------------------------------
        # SUMMARY
        # ---------------------------------------------------------------------

        summary_rows.append({

            "image":
                image_name,

            "nuclei_channel":
                nucChannel,

            "threshold_method":
                threshold_method,

            "threshold_input":
                NUCthreshold,

            "final_threshold":
                final_threshold,

            "nuclear_radius_px":
                NUCradius,

            "expected_area":
                np.pi * NUCradius ** 2,

            "min_area_factor":
                minAreaFactor,

            "minimum_area":
                (
                    np.pi
                    * NUCradius ** 2
                    * minAreaFactor
                ),

            "cluster_factor":
                clusterFactor,

            "cluster_area_threshold":
                (
                    np.pi
                    * NUCradius ** 2
                    * clusterFactor
                ),

            "fiji_watershed_tolerance":
                fijiWatershedTolerance,

            "nuclei_count":
                nuclei_count,

            "nuclei_total_area":
                nuclei_total_area,

            "nuclei_modal_area":
                nuclei_modal_area,

            "mask_saved_to":
                mask_out_path

        })

        # ---------------------------------------------------------------------
        # SINGLE IMAGE SANITY CHECK DATA
        # ---------------------------------------------------------------------

        if len(file_paths) == 1:

            last_result = {

                "nuc_img":
                    nuc_img,

                "flattened":
                    rolling_ball_background_subtract(
                        nuc_img,
                        NUCradius * 2
                    ),

                "nuclei_mask":
                    nuclei_mask,

                "nuclei_count":
                    nuclei_count,

                "final_threshold":
                    final_threshold,

                "threshold_method":
                    threshold_method

            }

        # ---------------------------------------------------------------------
        # PRINT RESULTS
        # ---------------------------------------------------------------------

        print(
            f"  Threshold method: "
            f"{threshold_method}"
        )

        print(
            f"  Final threshold: "
            f"{final_threshold:.2f}"
        )

        print(
            f"  Expected nuclear area: "
            f"{np.pi * NUCradius ** 2:.2f} px²"
        )

        print(
            f"  Minimum area: "
            f"{np.pi * NUCradius ** 2 * minAreaFactor:.2f} px²"
        )

        print(
            f"  Cluster threshold: "
            f"{np.pi * NUCradius ** 2 * clusterFactor:.2f} px²"
        )

        print(
            f"  Nuclei detected: "
            f"{nuclei_count}"
        )

    except Exception as e:

        print(
            f"  ERROR processing "
            f"{image_name}:"
        )

        print(
            f"  {type(e).__name__}: {e}"
        )

        summary_rows.append({

            "image":
                image_name,

            "nuclei_channel":
                nucChannel,

            "threshold_method":
                "",

            "threshold_input":
                NUCthreshold,

            "final_threshold":
                np.nan,

            "nuclear_radius_px":
                NUCradius,

            "expected_area":
                np.pi * NUCradius ** 2,

            "min_area_factor":
                minAreaFactor,

            "minimum_area":
                (
                    np.pi
                    * NUCradius ** 2
                    * minAreaFactor
                ),

            "cluster_factor":
                clusterFactor,

            "cluster_area_threshold":
                (
                    np.pi
                    * NUCradius ** 2
                    * clusterFactor
                ),

            "fiji_watershed_tolerance":
                fijiWatershedTolerance,

            "nuclei_count":
                np.nan,

            "nuclei_total_area":
                np.nan,

            "nuclei_modal_area":
                np.nan,

            "mask_saved_to":
                ""

        })


# =====================================================================================
# SUMMARY CSV
# =====================================================================================

summary_df = pd.DataFrame(
    summary_rows
)

print(
    "\n=== Summary ==="
)

print(
    summary_df.to_string(
        index=False
    )
)

if file_paths:

    summary_dir = os.path.dirname(
        file_paths[0]
    )

    summary_path = os.path.join(

        summary_dir,

        "nuclei_summary.csv"

    )

    summary_df.to_csv(

        summary_path,

        index=False

    )

    print(
        "\nSummary CSV saved to:"
    )

    print(
        summary_path
    )


# =====================================================================================
# SINGLE IMAGE SANITY CHECK
# =====================================================================================

if (

    len(file_paths) == 1

    and

    last_result is not None

):

    fig, axes = plt.subplots(

        1,

        3,

        figsize=(15, 5)

    )

    # -------------------------------------------------------------------------
    # RAW NUCLEAR CHANNEL
    # -------------------------------------------------------------------------

    axes[0].imshow(

        last_result[
            "nuc_img"
        ],

        cmap="gray"

    )

    axes[0].set_title(

        f"Channel {nucChannel} (raw)"

    )

    axes[0].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # BACKGROUND SUBTRACTED
    # -------------------------------------------------------------------------

    axes[1].imshow(

        last_result[
            "flattened"
        ],

        cmap="gray"

    )

    axes[1].set_title(

        "Background-subtracted"

    )

    axes[1].axis(
        "off"
    )

    # -------------------------------------------------------------------------
    # FINAL NUCLEAR MASK
    # -------------------------------------------------------------------------

    axes[2].imshow(

        last_result[
            "nuclei_mask"
        ],

        cmap="gray"

    )

    axes[2].set_title(

        f"{last_result['threshold_method']} "
        f"threshold = "
        f"{last_result['final_threshold']:.2f}\n"
        f"Nuclei = "
        f"{last_result['nuclei_count']}"

    )

    axes[2].axis(
        "off"
    )

    plt.tight_layout()

    plt.show()

else:

    print(

        "\nMultiple images processed -- "
        "skipping sanity check plot."

    )
