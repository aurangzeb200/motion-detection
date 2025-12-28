import numpy as np
from collections import deque

def find_connected_components(mask, connectivity=8, min_area=50, max_area_ratio=0.5):
    H, W = mask.shape
    max_area = int(H * W * max_area_ratio)

    bin_mask = (mask != 0).astype(np.uint8)
    labeled = np.zeros((H, W), dtype=np.int32)
    label = 1
    comp_info = []
    filtered_mask = np.zeros((H, W), dtype=np.uint8)

    if connectivity == 4:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for i in range(H):
        for j in range(W):
            if bin_mask[i, j] == 1 and labeled[i, j] == 0:
                q = deque()
                q.append((i, j))
                labeled[i, j] = label
                pixels = [(i, j)]
                min_r, min_c, max_r, max_c = i, j, i, j
                sum_r, sum_c = i, j

                while q:
                    r, c = q.popleft()
                    for dr, dc in neighbors:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if bin_mask[nr, nc] == 1 and labeled[nr, nc] == 0:
                                labeled[nr, nc] = label
                                q.append((nr, nc))
                                pixels.append((nr, nc))
                                if nr < min_r: min_r = nr
                                if nc < min_c: min_c = nc
                                if nr > max_r: max_r = nr
                                if nc > max_c: max_c = nc
                                sum_r += nr
                                sum_c += nc

                area = len(pixels)
                centroid = (sum_r / area, sum_c / area)
                bbox = (min_r, min_c, max_r, max_c)

                comp_info.append({"label": label, "area": area, "centroid": centroid, "bbox": bbox})

                if min_area <= area <= max_area:
                    for (r, c) in pixels:
                        filtered_mask[r, c] = 1

                label += 1

    num_components = label - 1
    return num_components, labeled, comp_info, filtered_mask

