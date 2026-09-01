import os
import glob
from datetime import datetime
import numpy as np
import tifffile as tiff
import PySimpleGUI as sg


FILE_PATTERNS = ("*.tif", "*.tiff")


# ============================================================
# GUI
# ============================================================

def select_input_folder():
    """Open a folder-selection dialog."""

    layout = [
        [
            sg.Text(
                "Select the folder containing your hyperstack TIFFs:"
            )
        ],
        [
            sg.Input(key="-FOLDER-"),
            sg.FolderBrowse()
        ],
        [
            sg.Button("OK"),
            sg.Button("Cancel")
        ],
    ]

    window = sg.Window(
        "Select input folder",
        layout
    )

    folder = None

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Cancel"):
            break

        if event == "OK":

            if values["-FOLDER-"]:
                folder = values["-FOLDER-"]
                break

            sg.popup(
                "Please select a folder first."
            )

    window.close()

    if not folder:
        raise SystemExit(
            "No folder selected — exiting."
        )

    return folder


def get_projection_settings(max_channels):
    """Ask the user for projection mode and channel thresholds."""

    layout = [
        [
            sg.Text("Select Projection Mode:")
        ],

        [
            sg.Combo(
                ["max", "mean"],
                default_value="max",
                readonly=True,
                key="PROJECTION_MODE"
            )
        ],

        [
            sg.Text(
                f"Minimum intensity for each of the "
                f"{max_channels} channels:"
            )
        ],

        [
            sg.Input(
                ",".join(["0"] * max_channels),
                key="MIN_INTENSITY",
                size=(50, 1)
            )
        ],

        [
            sg.Text(
                "Example: 0, 0, 100, 500"
            )
        ],

        [
            sg.Button("OK"),
            sg.Button("Cancel")
        ],
    ]

    window = sg.Window(
        "Projection Settings",
        layout
    )

    while True:

        event, values = window.read()

        if event in (sg.WIN_CLOSED, "Cancel"):

            window.close()

            raise SystemExit(
                "No projection settings selected — exiting."
            )

        if event == "OK":

            try:

                mode = values["PROJECTION_MODE"]

                intensity_list = [
                    float(value.strip())
                    for value in
                    values["MIN_INTENSITY"].split(",")
                ]

                if len(intensity_list) != max_channels:

                    sg.popup(
                        f"Please enter exactly "
                        f"{max_channels} intensity values."
                    )

                    continue

                min_intensity = {
                    channel: intensity_list[channel]
                    for channel in range(max_channels)
                }

                window.close()

                return mode, min_intensity

            except ValueError:

                sg.popup(
                    "Intensity values must be numbers."
                )


# ============================================================
# AXIS HANDLING
# ============================================================

def move_axis_to(data, axes, axis_name, destination):
    """
    Move one axis to a specified position.

    Returns:
        data
        updated axes string
    """

    if axis_name not in axes:
        return data, axes

    source = axes.index(axis_name)

    data = np.moveaxis(
        data,
        source,
        destination
    )

    axes_list = list(axes)

    axis = axes_list.pop(source)
    axes_list.insert(destination, axis)

    return data, "".join(axes_list)


# ============================================================
# TIFF METADATA
# ============================================================

def get_tiff_resolution(tf):
    """
    Read X/Y resolution information from the original TIFF.

    Returns:
        x_resolution
        y_resolution
        resolution_unit
    """

    page = tf.pages[0]

    x_resolution = None
    y_resolution = None
    resolution_unit = None

    if "XResolution" in page.tags:
        x_resolution = page.tags["XResolution"].value

    if "YResolution" in page.tags:
        y_resolution = page.tags["YResolution"].value

    if "ResolutionUnit" in page.tags:
        resolution_unit = page.tags["ResolutionUnit"].value

    return (
        x_resolution,
        y_resolution,
        resolution_unit
    )


def get_imagej_metadata(tf):
    """
    Return ImageJ metadata from the source TIFF.

    Only metadata that can be safely transferred to the
    projected image is retained.
    """

    try:

        metadata = tf.imagej_metadata

        if metadata is None:
            return {}

        return dict(metadata)

    except Exception:

        return {}


# ============================================================
# PROJECTION
# ============================================================

def project_z(stack, mode, min_intensity):
    """
    Project a stack along its first axis.

    Input:
        Z, ...

    Output:
        ...
    """

    original_dtype = stack.dtype

    stack_float = stack.astype(
        np.float32,
        copy=False
    )

    if min_intensity > 0:

        mask = (
            stack_float >= min_intensity
        )

        stack_float = np.where(
            mask,
            stack_float,
            np.nan
        )

    if mode == "max":

        with np.errstate(
            invalid="ignore"
        ):

            projected = np.nanmax(
                stack_float,
                axis=0
            )

    elif mode == "mean":

        with np.errstate(
            invalid="ignore"
        ):

            projected = np.nanmean(
                stack_float,
                axis=0
            )

    else:

        raise ValueError(
            f"Unknown projection mode: {mode}"
        )

    # Pixels for which every Z slice was below
    # threshold become NaN.
    projected = np.nan_to_num(
        projected,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Restore original data type.
    if np.issubdtype(
        original_dtype,
        np.integer
    ):

        limits = np.iinfo(
            original_dtype
        )

        projected = np.clip(
            projected,
            limits.min,
            limits.max
        )

    return projected.astype(
        original_dtype
    )


# ============================================================
# DISPLAY RANGE
# ============================================================

def calculate_display_ranges(image, percentile=99.7):
    """
    Calculate ImageJ display ranges.

    Each non-YX plane receives its own range.
    """

    if image.ndim == 2:

        upper = np.percentile(
            image,
            percentile
        )

        return [(0, float(upper))]

    planes = image.reshape(
        (-1,) +
        image.shape[-2:]
    )

    ranges = []

    for plane in planes:

        upper = np.percentile(
            plane,
            percentile
        )

        if not np.isfinite(upper):
            upper = 0

        ranges.append(
            (0, float(upper))
        )

    return ranges


# ============================================================
# CHANNEL INFORMATION
# ============================================================

def get_channel_labels(imagej_metadata, n_channels):
    """
    Extract ImageJ channel labels when available.

    If they are unavailable, return generic labels.
    """

    labels = imagej_metadata.get(
        "Labels"
    )

    if labels is not None:

        labels = list(labels)

        if len(labels) == n_channels:
            return labels

    return [
        f"Channel {i + 1}"
        for i in range(n_channels)
    ]


# ============================================================
# CHANNEL COUNT
# ============================================================

def find_max_channels(files):
    """Find the largest channel count in the input folder."""

    max_channels = 1

    for file_path in files:

        try:

            with tiff.TiffFile(file_path) as tf:

                series = tf.series[0]

                if "C" in series.axes:

                    channel_axis = (
                        series.axes.index("C")
                    )

                    n_channels = (
                        series.shape[channel_axis]
                    )

                    max_channels = max(
                        max_channels,
                        n_channels
                    )

        except Exception as error:

            print(
                f"Could not inspect "
                f"{os.path.basename(file_path)}: "
                f"{error}"
            )

    return max_channels


# ============================================================
# FILE PROCESSING
# ============================================================

def process_file(
    tif_path,
    projection_mode,
    min_intensity_by_channel
):

    print("\n" + "=" * 70)

    print(
        f"Processing: "
        f"{os.path.basename(tif_path)}"
    )

    # --------------------------------------------------------
    # Read source TIFF
    # --------------------------------------------------------

    with tiff.TiffFile(tif_path) as tf:

        series = tf.series[0]

        data = series.asarray()

        original_axes = series.axes

        original_shape = data.shape

        original_dtype = data.dtype

        imagej_metadata = (
            get_imagej_metadata(tf)
        )

        (
            x_resolution,
            y_resolution,
            resolution_unit
        ) = get_tiff_resolution(tf)

        channel_count = (
            data.shape[
                original_axes.index("C")
            ]
            if "C" in original_axes
            else 1
        )

    print(
        f"Original shape: "
        f"{original_shape}"
    )

    print(
        f"Original axes:  "
        f"{original_axes}"
    )

    print(
        f"Original dtype: "
        f"{original_dtype}"
    )

    print(
        f"Channels: "
        f"{channel_count}"
    )

    # --------------------------------------------------------
    # Require Z
    # --------------------------------------------------------

    if "Z" not in original_axes:

        print(
            "No Z axis found — skipping."
        )

        return

    # --------------------------------------------------------
    # Move Z to first position
    # --------------------------------------------------------

    data, axes = move_axis_to(
        data,
        original_axes,
        "Z",
        0
    )

    # --------------------------------------------------------
    # Move C next to Z
    # --------------------------------------------------------

    if "C" in axes:

        data, axes = move_axis_to(
            data,
            axes,
            "C",
            1
        )

    print(
        f"Processing axes: "
        f"{axes}"
    )

    # --------------------------------------------------------
    # Project
    # --------------------------------------------------------

    if "C" in axes:

        # After reordering:
        #
        # Z C [T/other dimensions] Y X

        n_channels = data.shape[1]

        projected_channels = []

        for channel in range(n_channels):

            threshold = (
                min_intensity_by_channel.get(
                    channel,
                    0
                )
            )

            print(
                f"  Channel {channel + 1}: "
                f"threshold = {threshold}"
            )

            channel_stack = (
                data[:, channel, ...]
            )

            projected = project_z(
                channel_stack,
                projection_mode,
                threshold
            )

            projected_channels.append(
                projected
            )

        output = np.stack(
            projected_channels,
            axis=0
        )

        output_axes = (
            "C" + axes[2:]
        )

    else:

        threshold = (
            min_intensity_by_channel.get(
                0,
                0
            )
        )

        output = project_z(
            data,
            projection_mode,
            threshold
        )

        output_axes = axes[1:]

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if len(output.shape) != len(output_axes):

        raise RuntimeError(
            "Output dimension mismatch: "
            f"shape={output.shape}, "
            f"axes={output_axes}"
        )

    print(
        f"Output shape: "
        f"{output.shape}"
    )

    print(
        f"Output axes:  "
        f"{output_axes}"
    )

    # --------------------------------------------------------
    # Prepare metadata
    # --------------------------------------------------------
    
    metadata = {
        "axes": output_axes
    }
    
    # --------------------------------------------------------
    # Processing information
    # --------------------------------------------------------
    
    processing_info = (
        "Z Projection\n"
        f"Projection type: {projection_mode}\n"
        f"Original filename: {os.path.basename(tif_path)}\n"
        f"Original shape: {original_shape}\n"
        f"Original axes: {original_axes}\n"
        f"Original data type: {original_dtype}\n"
        f"Number of Z slices: {original_shape[original_axes.index('Z')]}\n"
        f"Number of channels: {channel_count}\n"
        "Minimum intensity thresholds:\n"
    )
    
    for channel in range(channel_count):
    
        threshold = min_intensity_by_channel.get(
            channel,
            0
        )
    
        processing_info += (
            f"  Channel {channel + 1}: {threshold}\n"
        )
    
    processing_info += (
        f"Processing date: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "Processing software: "
        "Polymerized Microtubule Z Projection Pipeline"
    )
    
    metadata["ProcessingInfo"] = processing_info

    # Preserve physical unit.
    if "unit" in imagej_metadata:

        metadata["unit"] = (
            imagej_metadata["unit"]
        )

    elif resolution_unit == 2:

        metadata["unit"] = "inch"

    elif resolution_unit == 3:

        metadata["unit"] = "cm"

    # Preserve Z spacing if it exists.
    if "spacing" in imagej_metadata:

        metadata["spacing"] = (
            imagej_metadata["spacing"]
        )

    # Preserve time interval.
    if "finterval" in imagej_metadata:

        metadata["finterval"] = (
            imagej_metadata["finterval"]
        )

    # Preserve FPS.
    if "fps" in imagej_metadata:

        metadata["fps"] = (
            imagej_metadata["fps"]
        )

    # Preserve channel labels.
    if "C" in output_axes:

        channel_labels = get_channel_labels(
            imagej_metadata,
            channel_count
        )

        metadata["Labels"] = channel_labels

    # --------------------------------------------------------
    # Display range
    # --------------------------------------------------------

    display_ranges = calculate_display_ranges(
        output
    )

    metadata["DisplayRangeMin"] = [
        minimum
        for minimum, maximum
        in display_ranges
    ]

    metadata["DisplayRangeMax"] = [
        maximum
        for minimum, maximum
        in display_ranges
    ]

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

    suffix = (
        "_MAXproj"
        if projection_mode == "max"
        else "_MEANproj"
    )

    output_path = (
        os.path.splitext(tif_path)[0]
        + suffix
        + ".tif"
    )

    # --------------------------------------------------------
    # Preserve X/Y resolution
    # --------------------------------------------------------

    resolution = None

    if (
        x_resolution is not None
        and y_resolution is not None
    ):

        resolution = (
            x_resolution,
            y_resolution
        )

    # --------------------------------------------------------
    # BigTIFF if necessary
    # --------------------------------------------------------

    estimated_size = (
        output.size *
        output.dtype.itemsize
    )

    use_bigtiff = (
        estimated_size > 3.5 * 1024**3
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    tiff.imwrite(
        output_path,
        output,
        imagej=True,
        bigtiff=use_bigtiff,
        resolution=resolution,
        metadata=metadata,
        description=processing_info
    )

    # --------------------------------------------------------
    # Verify written file
    # --------------------------------------------------------

    with tiff.TiffFile(output_path) as tf:

        written_series = tf.series[0]

        print(
            "\nWritten file verification:"
        )

        print(
            f"  Shape: "
            f"{written_series.shape}"
        )

        print(
            f"  Axes:  "
            f"{written_series.axes}"
        )

        print(
            f"  Dtype: "
            f"{written_series.dtype}"
        )

        page = tf.pages[0]

        if "XResolution" in page.tags:

            print(
                f"  XResolution: "
                f"{page.tags['XResolution'].value}"
            )

        if "YResolution" in page.tags:

            print(
                f"  YResolution: "
                f"{page.tags['YResolution'].value}"
            )

        if "ResolutionUnit" in page.tags:

            print(
                f"  ResolutionUnit: "
                f"{page.tags['ResolutionUnit'].value}"
            )

    print(
        f"\nSaved: "
        f"{os.path.basename(output_path)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    input_folder = (
        select_input_folder()
    )

    # --------------------------------------------------------
    # Find TIFF files
    # --------------------------------------------------------

    files = []

    for pattern in FILE_PATTERNS:

        files.extend(
            glob.glob(
                os.path.join(
                    input_folder,
                    pattern
                )
            )
        )

    # Do not process files already generated
    # by this script.
    files = sorted(
        file_path
        for file_path in files
        if "_MAXproj" not in file_path
        and "_MEANproj" not in file_path
    )

    if not files:

        print(
            f"No TIFF files found in "
            f"{input_folder}"
        )

        return

    print(
        f"Found {len(files)} TIFF files."
    )

    # --------------------------------------------------------
    # Determine maximum channel count
    # --------------------------------------------------------

    max_channels = (
        find_max_channels(files)
    )

    print(
        f"Maximum number of channels: "
        f"{max_channels}"
    )

    # --------------------------------------------------------
    # User settings
    # --------------------------------------------------------

    projection_mode, min_intensity = (
        get_projection_settings(
            max_channels
        )
    )

    print(
        f"Projection mode: "
        f"{projection_mode}"
    )

    print(
        f"Minimum intensity: "
        f"{min_intensity}"
    )

    # --------------------------------------------------------
    # Process files
    # --------------------------------------------------------

    successful = 0
    failed = 0
    skipped = 0

    for file_path in files:

        try:

            process_file(
                file_path,
                projection_mode,
                min_intensity
            )

            successful += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR: "
                f"{os.path.basename(file_path)}"
            )

            print(
                f"{type(error).__name__}: "
                f"{error}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print("Finished.")

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed:     {failed}"
    )

    print(
        f"Skipped:    {skipped}"
    )


if __name__ == "__main__":
    main()