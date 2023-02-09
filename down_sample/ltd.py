# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import linregress


def _area_of_triangles(this_bin, pre_point, next_point):
    """Area of a triangle from duples of vertex coordinates
    Uses implicit numpy boradcasting along first axis of this_bin"""
    bin_pre = this_bin - pre_point
    pre_bin = pre_point - this_bin
    return 0.5 * abs((pre_point[0] - next_point[0]) * (bin_pre[:, 1])
                     - (pre_bin[:, 0]) * (next_point[1] - pre_point[1]))


def _split_data(data, n_out):
    """
    Split data into bins
    Dynamically adjust bucket size according to SSE (the sum of squared errors)
    :param data:
    :return:
    """
    iteration_num = int(len(data)/(n_out*10))
    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)
    curr_iteration = 1
    while curr_iteration <= iteration_num:
        sse_list = []
        for i in range(len(data_bins)):
            if i == 0:
                pre_point = np.array([data[0]])
                next_bin = data_bins[i+1]
                next_point = np.array([next_bin[0]])
            elif i == len(data_bins)-1:
                pre_bin = data_bins[i-1]
                pre_point = np.array([pre_bin[-1]])
                next_point = np.array([data[-1]])
            else:
                pre_bin = data_bins[i - 1]
                pre_point = np.array([pre_bin[-1]])
                next_bin = data_bins[i + 1]
                next_point = np.array([next_bin[0]])

            this_bin = np.append(pre_point, data_bins[i], axis=0)
            this_bin = np.append(this_bin, next_point, axis=0)
            sse_list.append(linregress(this_bin).stderr)
        data_bins = _resize_bins(data_bins, sse_list)
        curr_iteration += 1
    return data_bins


def _find_max_bin(data_bins, sse_list):
    """
    return max_index in SSE list, and the data_bin with max SSE must has at least two data points.
    :param data_bins:
    :param sse_list:
    :return:
    """
    sort_sse = np.argsort(sse_list)
    for i in range(-1, -len(sort_sse)-1, -1):
        if len(data_bins[sort_sse[i]]) >= 2:
            return sort_sse[i]
    return sort_sse[i]


def _find_min_bin(sse_list):
    """
    return min_index in SEE list
    :param sse_list:
    :return:
    """
    sum_pairs = [sse_list[i] + sse_list[i + 1] for i in range(len(sse_list) - 1)]
    return sum_pairs.index(min(sum_pairs))


def _resize_bins(data_bins, sse_list):
    """
    resize bins according to SSE (the sum of squared errors)
    :param data_bins:
    :param sse_list:
    :return:
    """
    resize_bins = []
    max_index = _find_max_bin(data_bins, sse_list)
    min_index = _find_min_bin(sse_list)

    for i in range(len(data_bins)):
        if i == max_index:
            resize_bins += np.array_split(data_bins[i], 2)
        elif i == min_index:
            resize_bins.append(np.concatenate([data_bins[i], data_bins[i+1]], axis=0))
        elif i == min_index + 1:
            continue
        else:
            resize_bins.append(data_bins[i])
    return resize_bins


def ltd(data, n_out):
    """Downsample ``data`` to ``n_out`` points using the LTD algorithm.

    Reference
    ---------

    Sveinn Steinarsson. 2013. Downsampling Time Series for Visual
    Representation. MSc thesis. University of Iceland.

    Constraints
    -----------
      - ncols(data) == 2
      - 3 <= n_out <= nrows(data)
      - ``data`` should be sorted on the first column.is_overseas

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
    # Dynamically adjust bucket size
    n_bins = n_out - 2
    data_bins = _split_data(data, n_out)

    # Prepare output array
    # First and last points are the same as in the input.
    out = np.zeros((n_out, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]

    # Keep the max point and min point in output.
    max_point = data[np.argmax(data, axis=0)[1]].tolist()
    min_point = data[np.argmin(data, axis=0)[1]].tolist()

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
