#!/usr/bin/env python3

import argparse
from pathlib import Path

import nrrd


def extract_nrrd_stack(
    input_path,
    output_path,
    x_range=None,
    y_range=None,
    z_range=None,
    axis_order="xyz",
):
    data, header = nrrd.read(input_path)

    if data.ndim != 3:
        raise ValueError(f"Input must be 3D. Current ndim={data.ndim}")

    axis_map = {a: i for i, a in enumerate(axis_order)}

    slices = [slice(None)] * 3

    for axis_name, selected_range in {
        "x": x_range,
        "y": y_range,
        "z": z_range,
    }.items():

        if selected_range is None:
            continue

        start, end = selected_range

        axis = axis_map[axis_name]
        size = data.shape[axis]

        if start < 0 or end >= size:
            raise ValueError(
                f"{axis_name} range {start}-{end} exceeds "
                f"0-{size-1}"
            )

        if start > end:
            raise ValueError(
                f"{axis_name} start must be <= end"
            )

        slices[axis] = slice(start, end + 1)

    cropped = data[tuple(slices)]

    # output_path = Path(output_path)
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    nrrd.write(output_path, cropped, header=header)

    print("===================================")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Original shape : {data.shape}")
    print(f"Cropped shape  : {cropped.shape}")
    print("Done.")
    print("===================================")


def main():

    parser = argparse.ArgumentParser(
        description="Crop a 3D NRRD stack."
    )

    parser.add_argument(
        "input",
        help="Input NRRD file"
    )

    parser.add_argument(
        "output",
        help="Output NRRD file"
    )

    parser.add_argument(
        "--axis-order",
        default="zyx",
        choices=["zyx", "xyz"],
        help="Axis order of the NRRD array (default: zyx)"
    )

    parser.add_argument(
        "--x",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Crop x range"
    )

    parser.add_argument(
        "--y",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Crop y range"
    )

    parser.add_argument(
        "--z",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Crop z range"
    )

    args = parser.parse_args()

    extract_nrrd_stack(
        input_path=args.input,
        output_path=args.output,
        x_range=args.x,
        y_range=args.y,
        z_range=args.z,
        axis_order=args.axis_order,
    )


if __name__ == "__main__":
    main()