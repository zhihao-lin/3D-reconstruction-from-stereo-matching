from scipy.ndimage.filters import median_filter, sobel
import numpy as np
import cv2
import time

# Author Andy Cheng
# The pipeline  and algorithms in references are implemented in this code.
# References:
# [1] Cross-Based Local Stereo Matching Using Orthogonal Integral Images 
#      Ke Zhang et.al 2009
# [2] On Building an Accurate Stereo Matching System on Graphics Hardware
# 	   Xing Mei et.al 2011


def AD_Census_Cost(Il, Ir, max_disp):
    h, w, ch = Il.shape
    ad = np.zeros((max_disp, h, w), dtype=np.float64)
    #  window size for census cost computation is 7x9
    census = np.zeros((max_disp, h, w), dtype=np.float64)
    padded_Il = np.pad(Il, ([3, 3], [4, 4], (0, 0)), 'reflect')
    padded_Ir = np.pad(Ir, ([3, 3], [4, 4], (0, 0)), 'reflect')
    # Ad
    for d in range(max_disp):
        for x in range(w):
            if x - d < 0:
                    ad[d, :, x] = np.ones_like(ad[d, :, x]) * np.inf
            else:
                ad[d, :, x] = np.sum(np.absolute(Il[:, x] - Ir[:, x - d]), axis=1) / 3.0
    # Census
    # In the paper, the window size for census cost computation is 7x9. As we compute hamming distance, we have to consider 3 channels.
    census_window_left = np.zeros((h, w, 7, 9, 3), dtype=np.float64)
    census_window_right = np.zeros((h, w, 7, 9, 3), dtype=np.float64)
    
    for y in range(h):
        for x in range(w):
            window_center_y, window_center_x = y + 3, x + 4
            census_window_left[y, x] = padded_Il[window_center_y - 3: window_center_y + 4, window_center_x - 4: window_center_x + 5] < padded_Il[window_center_y, window_center_x]
            census_window_right[y, x] = padded_Ir[window_center_y - 3: window_center_y + 4, window_center_x - 4: window_center_x + 5] < padded_Ir[window_center_y, window_center_x]
    # We only have to consider census on the same horizontal line. 
    for d in range(max_disp):
        for x in range(w):    
            if x - d < 0:
                census[d, :, x] = np.ones_like(census[d, :, x]) * np.inf
            else:
                bit_string = (census_window_left[:, x] != census_window_right[:, x - d])
                census[d, :, x] = np.sum(bit_string.reshape(h, -1), axis=1)
    # Normalize census by division by 3, beacause of RGB 3 channels.
    result = (1 - np.exp(-ad / 10)) + (1 - np.exp(-census / 90))
    return result

# Construct upright cross for each pixel
def construct_cross(I):
    # 3 rules presented in the paper
    def criteria(center,  arm, edge, I):
        p_x, p_y = center
        q_x, q_y = arm
        if not (0 <= q_y < h and 0 <= q_x < w): 
            return False
        if abs(q_y - p_y) == 1 or abs(q_x - p_x) == 1: 
            return True

        if abs((I[p_y, p_x, 0]) - (I[q_y, q_x, 0])) >= 20: 
            return False
        if abs((I[p_y, p_x, 1]) - (I[q_y, q_x, 1])) >= 20: 
            return False
        if abs((I[p_y, p_x, 2]) - (I[q_y, q_x, 2])) >= 20: 
            return False

        if abs((I[q_y, q_x, 0]) - (I[q_y + edge[0], q_x + edge[1], 0])) >= 20: 
            return False
        if abs((I[q_y, q_x, 1]) - (I[q_y + edge[0], q_x + edge[1], 1]) ) >= 20: 
            return False
        if abs((I[q_y, q_x, 2]) - (I[q_y + edge[0], q_x + edge[1], 2])) >= 20: 
            return False

        if abs(q_y - p_y) >= 34 or abs(q_x - p_x) >= 34: 
            return False

        if abs(q_y - p_y) >= 17 or abs(q_x - p_x) >= 17:
            if abs((I[p_y, p_x, 0]) - (I[q_y, q_x, 0])) >= 6: 
                return False
            if abs((I[p_y, p_x, 1]) - (I[q_y, q_x, 1])) >= 6: 
                return False
            if abs((I[p_y, p_x, 2]) - (I[q_y, q_x, 2])) >= 6: 
                return False
        return True

    h, w, ch = I.shape        
    cross = np.empty((h, w, 4), dtype=np.int)
    # Cross arm order: (top, bottom, left, right)
    for y in range(h):
        for x in range(w):
            cross[y, x, 0] = y - 1
            cross[y, x, 1] = y + 1
            cross[y, x, 2] = x - 1
            cross[y, x, 3] = x + 1
            # Expand the arms continually until it breaks the rules
            while criteria((x, y), (x, cross[y, x, 0]), [1, 0], I):
                cross[y, x, 0] -= 1
            while criteria((x, y), (x, cross[y, x, 1]), [-1, 0], I):
                cross[y, x, 1] += 1
            while criteria((x, y), (cross[y, x, 2], y), [0, 1], I):
                cross[y, x, 2] -= 1
            while criteria((x, y), (cross[y, x, 3], y), [0, -1], I):
                cross[y, x, 3] += 1
    return cross

def cross_based_aggregation(cross_l, cross_r, prev_cost_volume, isHorzontal, max_disp, h, w):
    cost_volume = np.empty_like(prev_cost_volume)
    horizontal_aggregation = np.empty_like(prev_cost_volume)
    vertical_aggregation = np.empty_like(prev_cost_volume)
    # Agrregation for fast computation with OII method
    for x in range(w):
        horizontal_aggregation[:, :, x] = np.sum(prev_cost_volume[:, :, :x + 1], axis=2)
    for y in range(h):
        vertical_aggregation[:, y, :] = np.sum(prev_cost_volume[:, :y + 1, :], axis=1)
    for d in range(max_disp):
        for y in range(h):
            for x in range(w):
                if x - d < 0:
                    cost_volume[d, y, x] = prev_cost_volume[d, y, x]
                    continue
                cost_per_pixel, pixel_count = 0, 0
                if isHorzontal:
                    up_edge = max(cross_l[y, x, 0], cross_r[y, x - d, 0]) + 1
                    down_edge = min(cross_l[y, x, 1], cross_r[y, x - d, 1])
                    for vertical_seg in range(up_edge, down_edge):
                        left_edge = max(cross_l[vertical_seg, x, 2], cross_r[vertical_seg, x - d, 2] + d) + 1
                        right_edge = min(cross_l[vertical_seg, x, 3], cross_r[vertical_seg, x - d, 3] + d)
                        assert(left_edge >= 0)
                        if left_edge - 1 < 0:
                            cost_per_pixel += horizontal_aggregation[d, vertical_seg, right_edge - 1]
                            pixel_count += right_edge
                        else:
                            cost_per_pixel += horizontal_aggregation[d, vertical_seg, right_edge - 1] - horizontal_aggregation[d, vertical_seg, left_edge - 1]
                            pixel_count += right_edge - left_edge
                else:
                    left_edge = max(cross_l[y, x, 2], cross_r[y, x - d, 2] + d) + 1
                    right_edge = min(cross_l[y, x, 3], cross_r[y, x - d, 3] + d)
                    for horizontal_seg in range(left_edge, right_edge):
                        up_edge = max(cross_l[y, horizontal_seg, 0], cross_r[y, horizontal_seg - d, 0]) + 1
                        down_edge = min(cross_l[y, horizontal_seg, 1], cross_r[y, horizontal_seg - d, 1])
                        assert(up_edge >=0)
                        if up_edge - 1 < 0:
                            cost_per_pixel += vertical_aggregation[d, down_edge - 1, horizontal_seg]
                            pixel_count += down_edge
                        else:
                            cost_per_pixel += vertical_aggregation[d, down_edge - 1, horizontal_seg] - vertical_aggregation[d, up_edge - 1, horizontal_seg]
                            pixel_count += down_edge - up_edge
                assert (pixel_count > 0)
                cost_volume[d, y, x] = cost_per_pixel / pixel_count
    return cost_volume
    
def scanline_optimization(cost_volume, max_disp, Il, Ir):
    h, w, ch = Il.shape
    # The following parameters are used as ones in the paper.
    T_so = 15
    Pi_1, Pi_2 = 1.0, 3.0

    # up down scan
    scanned_filtered_volume = np.empty_like(cost_volume)
    for x in range(w):
        for y in range(h):
            current_min = np.inf
            for d in range(max_disp):
                if y - 1 < 0 or x - d < 0:
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x]
                else:
                    d_1 = max(abs(Il[y, x, 0] - Il[y - 1, x, 0]),
                                abs(Il[y, x, 1] - Il[y - 1, x, 1]),
                                abs(Il[y, x, 2] - Il[y - 1, x, 2]))
                    d_2 = max(abs(Ir[y, x - d, 0] - Ir[y - 1, x - d, 0]),
                                abs(Ir[y, x - d, 1] - Ir[y - 1, x - d, 1]),
                                abs(Ir[y, x - d, 2] - Ir[y - 1, x - d, 2]))
                    if d_1 < T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1, Pi_2
                    elif d_1 < T_so and d_2 >= T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    elif d_1 >= T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    else:
                        p_1, p_2 = Pi_1 / 10., Pi_2 / 10.
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x] - prev_min + min(
                        scanned_filtered_volume[d, y - 1, x],
                        scanned_filtered_volume[d - 1, y - 1, x] + p_1 if d - 1 >= 0 else np.inf,
                        scanned_filtered_volume[d + 1, y - 1, x] + p_1 if d + 1 < max_disp else np.inf,
                        prev_min + p_2
                    )
                if scanned_filtered_volume[d, y, x] < current_min:
                    current_min = scanned_filtered_volume[d, y, x]
            prev_min = current_min
    up_down = scanned_filtered_volume

    # down up scan
    scanned_filtered_volume = np.empty_like(cost_volume)
    for x in range(w):
        for y in range(h - 1, -1, -1):
            current_min = np.inf
            for d in range(max_disp):
                if y + 1 >= h or x - d < 0:
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x]
                else:
                    d_1 = max(abs(Il[y, x, 0] - Il[y + 1, x, 0]),
                                abs(Il[y, x, 1] - Il[y + 1, x, 1]),
                                abs(Il[y, x, 2] - Il[y + 1, x, 2]))
                    d_2 = max(abs(Ir[y, x - d, 0] - Ir[y + 1, x - d, 0]),
                                abs(Ir[y, x - d, 1] - Ir[y + 1, x - d, 1]),
                                abs(Ir[y, x - d, 2] - Ir[y + 1, x - d, 2]))
                    if d_1 < T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1, Pi_2
                    elif d_1 < T_so and d_2 >= T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    elif d_1 >= T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    else:
                        p_1, p_2 = Pi_1 / 10., Pi_2 / 10.
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x] - prev_min + min(
                        scanned_filtered_volume[d, y + 1, x],
                        scanned_filtered_volume[d - 1, y + 1, x] + p_1 if d - 1 >= 0 else np.inf,
                        scanned_filtered_volume[d + 1, y + 1, x] + p_1 if d + 1 < max_disp else np.inf,
                        prev_min + p_2
                    )
                if scanned_filtered_volume[d, y, x] < current_min:
                    current_min = scanned_filtered_volume[d, y, x]
            prev_min = current_min
    down_up = scanned_filtered_volume

    # left right scan
    scanned_filtered_volume = np.empty_like(cost_volume)
    prev_min = 0
    for y in range(h):
        for x in range(w):
            current_min = np.inf
            for d in range(max_disp):
                if x - d - 1 < 0:
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x]
                else:
                    d_1 = max(abs(Il[y, x, 0] - Il[y, x - 1, 0]), abs(Il[y, x, 1] - Il[y, x - 1, 1]), abs(Il[y, x, 2] - Il[y, x - 1, 2]))
                    d_2 = max(abs(Ir[y, x - d, 0] - Ir[y, x - d - 1, 0]), abs(Ir[y, x - d, 1] - Ir[y, x - d - 1, 1]), abs(Ir[y, x - d, 2] - Ir[y, x - d - 1, 2]))
                    if d_1 < T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1, Pi_2
                    elif d_1 < T_so and d_2 >= T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    elif d_1 >= T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1 / 4., Pi_2 / 4.
                    else:
                        p_1, p_2 = Pi_1 / 10., Pi_2 / 10.
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x] - prev_min + min(
                        scanned_filtered_volume[d, y, x - 1],
                        scanned_filtered_volume[d - 1, y, x - 1] + p_1 if d - 1 >= 0 else np.inf,
                        scanned_filtered_volume[d + 1, y, x - 1] + p_1 if d + 1 < max_disp else np.inf,
                        prev_min + p_2
                    )
                if scanned_filtered_volume[d, y, x] < current_min:
                    current_min = scanned_filtered_volume[d, y, x]
            prev_min = current_min
    left_right = scanned_filtered_volume

    # right left scan
    scanned_filtered_volume = np.empty_like(cost_volume)
    for y in range(h):
        for x in range(w-1, -1, -1):
            current_min = np.inf
            for d in range(max_disp):
                if x + 1 >= w or x - d < 0:
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x]
                else:
                    d_1 = max(abs(Il[y, x, 0] - Il[y, x + 1, 0]),
                                abs(Il[y, x, 1] - Il[y, x + 1, 1]),
                                abs(Il[y, x, 2] - Il[y, x + 1, 2]))
                    d_2 = max(abs(Ir[y, x - d, 0] - Ir[y, x - d + 1, 0]),
                                abs(Ir[y, x - d, 1] - Ir[y, x - d + 1, 1]),
                                abs(Ir[y, x - d, 2] - Ir[y, x - d + 1, 2]))
                    if d_1 < T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1, Pi_2
                    elif d_1 < T_so and d_2 >= T_so:
                        p_1, p_2 = Pi_1/4., Pi_2/4.
                    elif d_1 >= T_so and d_2 < T_so:
                        p_1, p_2 = Pi_1/4., Pi_2/4.
                    else:
                        p_1, p_2 = Pi_1/10., Pi_2/10.
                    scanned_filtered_volume[d, y, x] = cost_volume[d, y, x] - prev_min + min(
                        scanned_filtered_volume[d, y, x + 1],
                        scanned_filtered_volume[d - 1, y, x + 1] + p_1 if d - 1 >= 0 else np.inf,
                        scanned_filtered_volume[d + 1, y, x + 1] + p_1 if d + 1 < max_disp else np.inf,
                        prev_min + p_2
                    )
                if scanned_filtered_volume[d, y, x] < current_min:
                    current_min = scanned_filtered_volume[d, y, x]
            prev_min = current_min
    right_left = scanned_filtered_volume
    # Average the scanned volumes in four directions
    cost_volume = (left_right + right_left + up_down + down_up) / 4
    return cost_volume

def cost_volume_construction(Il, Ir, max_disp, isRight):
    # Flip the ordder of Il and Ir if the initial cost volume is constreucted for the right image.
    # Note: Close one of your eye and look at an object, you'll see why is that.
    if isRight:
        Il = np.flip(Il, axis=1)
        Ir = np.flip(Ir, axis=1)
    h, w, ch = Il.shape
    # >>> Cost computation
    tic = time.time()
    # Compute Ad-Census cost for each pixel 
    cost_volume = AD_Census_Cost(Il, Ir, max_disp)
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))

    # >>> Cost aggregation
    tic = time.time()
    # Cross-based Cost Aggregation
    cross_l = construct_cross(Il)
    cross_r = construct_cross(Ir)

    # In the paper, the algorithm uses 4 times of aggregation, with horizontal one  and vertical one alternatively.
    cost_volume = cross_based_aggregation(cross_l, cross_r, cost_volume, True, max_disp, h, w)
    cost_volume = cross_based_aggregation(cross_l, cross_r, cost_volume, False, max_disp, h, w)
    cost_volume = cross_based_aggregation(cross_l, cross_r, cost_volume, True, max_disp, h, w)
    cost_volume = cross_based_aggregation(cross_l, cross_r, cost_volume, False, max_disp, h, w)
    
    cost_volume = scanline_optimization(cost_volume, max_disp, Il, Ir)
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))
    if isRight:
        return cost_volume
    return cost_volume, cross_l

def outlier_detection(Il, Ir,labels, labels_r):
    h, w, ch = Il.shape
    # 0: not an outlier, 1: outkier
    outliers = np.zeros_like(labels)
    # Left-right consistency check
    for y in range(h):
        for x in range(w):
            if abs(labels[y, x]- labels_r[y, x-labels[y, x]]) > 1:
                outliers[y, x] = 1
            elif x - labels[y, x] < 0:
                outliers[y, x] = 1
    return outliers


def interpolation(outliers, labels, h, w, cross, cost_volume):
    for y in range(h):
        for x in range(w):
            if outliers[y, x] != 0:
                min_cost = np.inf
                point = (-1, -1)
                for y_in_cross in range(cross[y, x, 0], cross[y, x, 1]):
                    cost = cost_volume[int(labels[y_in_cross, x]), y_in_cross, x]
                    if cost < min_cost and outliers[y_in_cross, x] == 0:
                        point = (y_in_cross, x)
                for x_in_cross in range(cross[y, x, 2], cross[y, x, 3]):
                    cost = cost_volume[int(labels[y, x_in_cross]), y, x_in_cross]
                    if cost < min_cost and outliers[y, x_in_cross] == 0:
                        point = (y, x_in_cross)
                labels[y, x] = labels[point[0], point[1]]
    return labels

# Detect the edges using sobel
def depth_discontinuity_adjustment(labels, cost_volume, h, w):
    # Horizontally
    refined_labels = np.copy(labels)
    sobel_filter = sobel(labels, axis=0)
    for y in range(h):
        for x in range(1, w-1):
            if sobel_filter[y, x] > 10 :
                d = labels[y, x]
                # If smaller, update
                if cost_volume[labels[y, x - 1], y, x] < cost_volume[d, y, x]:
                    d = labels[y, x - 1]
                if cost_volume[labels[y, x + 1], y, x] < cost_volume[d, y, x]:
                    d = labels[y, x + 1]
                refined_labels[y, x] = d
    labels = refined_labels    
    # Vertically do again
    refined_labels = np.copy(labels)
    sobel_filter = sobel(labels, axis=1)
    for y in range(1, h-1):
        for x in range(w):
            if sobel_filter[y, x] > 10:
                d = labels[y, x]
                # If smaller, update
                if cost_volume[labels[y - 1, x], y, x] < cost_volume[d, y, x]:
                    d = labels[y - 1, x]
                if cost_volume[labels[y + 1, x], y, x] < cost_volume[d, y, x]:
                    d = labels[y + 1, x]
                refined_labels[y, x] = d
    return refined_labels

def subpixel_enhancement(labels, cost_volume, max_disp, h, w):
    refined_labels = np.copy(labels)
    for y in range(h):
        for x in range(w):
            d = refined_labels[y, x]
            if 1 <= d < max_disp - 1:
                C_n = cost_volume[d - 1, y, x]
                C_z = cost_volume[d, y, x]
                C_p = cost_volume[d + 1, y, x]
                factor = 2 * (C_p + C_n - 2 * C_z)
                if factor > 1e-5:
                    refined_labels[y, x] = d - min(1, max(-1, (C_p - C_n) / factor))
    return refined_labels


def computeDisp(Il, Ir, max_disp, name):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    labels_r = np.zeros((h, w), dtype=np.uint8)
    # Transform dtype in case of overflow
    Il = Il.astype(np.float64)
    Ir = Ir.astype(np.float64)
    cost_volume, cross = cost_volume_construction(Il, Ir, max_disp, isRight=False)
    # order along x-axis is reversed
    cost_volume_r = cost_volume_construction(Ir, Il, max_disp, isRight=True)
    # >>> Disparity optimization
    tic = time.time()
    # Using WTA
    labels = cost_volume.argmin(0)
    labels_r = cost_volume_r.argmin(0)
    # flip the order along x-axis back
    labels_r = np.flip(labels_r, axis=1)
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))

    # >>> Disparity refinement
    tic = time.time()
    # ex: Left-right consistency check + hole filling + weighted median filtering
    outliers = outlier_detection(Il, Ir, labels, labels_r)
    labels = interpolation(outliers, labels, h, w, cross, cost_volume)
    labels = depth_discontinuity_adjustment(labels, cost_volume, h, w)
    labels = subpixel_enhancement(labels, cost_volume, max_disp, h, w)
    labels = median_filter(labels, size=3, mode='nearest')
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    return labels

def main():
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp, 'tsukuba_cross.png')
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))

    
    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp, 'venus_cross.png')
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))
    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, 'teddy_cross.png')
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))
    
    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp, 'cones_cross.png')
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))

if __name__ == '__main__':
    main()