import numpy as np

# Step 3: Mean & Variance 
def compute_mean(frames: np.ndarray) -> np.ndarray:
    F, H, W = frames.shape
    mean = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            s = 0.0  
            for f in range(F):
                s += float(frames[f, i, j])
            mean[i, j] = s / float(F)
    return mean

def compute_variance(frames: np.ndarray, mean_frame: np.ndarray) -> np.ndarray:
    F, H, W = frames.shape
    variance = np.zeros((H, W), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            s = 0.0
            mu = float(mean_frame[i, j])
            for f in range(F):
                d = float(frames[f, i, j]) - mu
                s += d * d
            variance[i, j] = s / float(F)
    return variance

def build_background_model(frames: np.ndarray, t: int):
    bg = frames[:t]
    mean = compute_mean(bg)
    var = compute_variance(bg, mean)
    return mean, var


# Step 4: Mahalanobis mask
def compute_mask(frame: np.ndarray, mean_frame: np.ndarray, variance_frame: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    eps = 1e-6
    safe_var = np.where(variance_frame <= 0.0, eps, variance_frame)
    M = np.sqrt(((frame - mean_frame) ** 2) / safe_var)
    mask = (M > threshold).astype(np.uint8)
    return mask
