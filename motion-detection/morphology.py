import numpy as np

def create_kernel(kernel_size=3):
    if kernel_size <= 0:
        raise ValueError("kernel_size must be positive")
    return np.ones((kernel_size, kernel_size), dtype=np.uint8)

def _pad_mask(mask, pad_top, pad_bottom, pad_left, pad_right):
    H, W = mask.shape
    out = np.zeros((H + pad_top + pad_bottom, W + pad_left + pad_right), dtype=np.uint8)
    out[pad_top:pad_top+H, pad_left:pad_left+W] = mask
    return out

def erode(mask, kernel, anchor=None, iterations=1):
    if anchor is None:
        ar = kernel.shape[0] // 2
        ac = kernel.shape[1] // 2
    else:
        ar, ac = anchor

    kH, kW = kernel.shape
    pad_top = ar
    pad_left = ac
    pad_bottom = kH - 1 - ar
    pad_right = kW - 1 - ac

    out = mask.copy().astype(np.uint8)
    H, W = out.shape

    for _ in range(iterations):
        padded = _pad_mask(out, pad_top, pad_bottom, pad_left, pad_right)
        new = np.zeros_like(out, dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                keep = True
                for m in range(kH):
                    if not keep:
                        break
                    for n in range(kW):
                        if kernel[m, n] == 0:
                            continue
                        pi = i + m
                        pj = j + n
                        if padded[pi, pj] == 0:
                            keep = False
                            break
                new[i, j] = 1 if keep else 0
        out = new
    return out

def dilate(mask, kernel, anchor=None, iterations=1):
    if anchor is None:
        ar = kernel.shape[0] // 2
        ac = kernel.shape[1] // 2
    else:
        ar, ac = anchor

    kH, kW = kernel.shape
    pad_top = ar
    pad_left = ac
    pad_bottom = kH - 1 - ar
    pad_right = kW - 1 - ac

    out = mask.copy().astype(np.uint8)
    H, W = out.shape

    for _ in range(iterations):
        padded = _pad_mask(out, pad_top, pad_bottom, pad_left, pad_right)
        new = np.zeros_like(out, dtype=np.uint8)
        for i in range(H):
            for j in range(W):
                set_one = False
                for m in range(kH):
                    if set_one:
                        break
                    for n in range(kW):
                        if kernel[m, n] == 0:
                            continue
                        pi = i + m
                        pj = j + n
                        if padded[pi, pj] == 1:
                            set_one = True
                            break
                new[i, j] = 1 if set_one else 0
        out = new
    return out

def morphological_operations(mask, kernel_size=3):
    kernel = create_kernel(kernel_size)
    anchor = (kernel_size // 2, kernel_size // 2)
    er = erode(mask, kernel, anchor=anchor, iterations=1)
    cleaned = dilate(er, kernel, anchor=anchor, iterations=1)
    return cleaned

