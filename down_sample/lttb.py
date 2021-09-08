# -*- coding: utf-8 -*-
import numpy as np


def _area_of_triangles(this_bin, pre_point, next_point):
    """Area of a triangle from duples of vertex coordinates
    Uses implicit numpy boradcasting along first axis of this_bin"""
    bin_pre = this_bin - pre_point
    pre_bin = pre_point - this_bin
    return 0.5 * abs((pre_point[0] - next_point[0]) * (bin_pre[:, 1])
                     - (pre_bin[:, 0]) * (next_point[1] - pre_point[1]))


def lttb(data, n_out):
    """Downsample ``data`` to ``n_out`` points using the LTTB algorithm.

    Reference
    ---------

    Sveinn Steinarsson. 2013. Downsampling Time Series for Visual
    Representation. MSc thesis. University of Iceland.

    Constraints
    -----------
      - ncols(data) == 2
      - 3 <= n_out <= nrows(data)
      - ``data`` should be sorted on the first column.

    Returns
    -------

    numpy.array of shape (n_out, 2).
    """
    # Validate input
    if data.shape[1] != 2:
        raise ValueError('data should have 2 columns')

    if any(data[:, 0] != np.sort(data[:, 0])):
        raise ValueError('data should be sorted on first column')

    if n_out > data.shape[0]:
        raise ValueError('n_out must be <= number of rows in data')

    if n_out == data.shape[0]:
        return data

    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')

    # Split data into bins
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((n_out, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # Keep the max point and min point in output.
    max_point = data[np.argmax(data, axis=0)[1]].tolist()
    min_point = data[np.argmin(data, axis=0)[1]].tolist()

    # Largest Triangle Three Buckets (LTTB):
    # In each bin, find the point that makes the largest triangle
    # with the point saved in the previous bin
    # and the centroid of the points in the next bin.
    for i in range(len(data_bins)):
        this_bin = data_bins[i]
        this_bin_lists = this_bin.tolist()

        if max_point in this_bin_lists:
            out[i + 1] = max_point
        elif min_point in this_bin_lists:
            out[i + 1] = min_point
        else:
            if i < n_bins - 1:
                next_bin = data_bins[i + 1]
            else:
                next_bin = data[len(data) - 1:]

            pre_point = out[i]
            next_point = next_bin.mean(axis=0)

            areas = _area_of_triangles(this_bin, pre_point, next_point)

            out[i + 1] = this_bin[np.argmax(areas)]

    return out
