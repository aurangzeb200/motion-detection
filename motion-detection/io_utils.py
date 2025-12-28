import struct
import zlib
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def _png_pack(tag, data):
    tagb = tag.encode("ascii")
    return struct.pack("!I", len(data)) + tagb + data + struct.pack("!I", zlib.crc32(tagb + data) & 0xffffffff)

def write_png(path, arr):
    arr = np.asarray(arr)
    if arr.dtype != np.uint8:
        raise ValueError("write_png requires uint8 array")
    if arr.ndim == 2:
        H, W = arr.shape
        color_type = 0  
        channels = 1
    elif arr.ndim == 3 and arr.shape[2] == 3:
        H, W, _ = arr.shape
        color_type = 2  
        channels = 3
    else:
        raise ValueError("write_png supports 2D (gray) or 3-channel RGB arrays")

    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack("!IIBBBBB", W, H, 8, color_type, 0, 0, 0)
        f.write(_png_pack("IHDR", ihdr))

        raw = b""
        for y in range(H):
            raw += b"\x00" + arr[y].tobytes()
        comp = zlib.compress(raw, level=6)
        f.write(_png_pack("IDAT", comp))
        f.write(_png_pack("IEND", b""))

def ensure_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def scale_to_uint8(arr):
    a = np.asarray(arr, dtype=np.float32)
    amin = a.min()
    amax = a.max()
    if amax - amin < 1e-8:
        return np.clip(a, 0, 255).astype(np.uint8)
    scaled = 255.0 * (a - amin) / (amax - amin)
    return np.clip(scaled, 0, 255).astype(np.uint8)

def save_binary_masks(frames_masks, out_folder):
    ensure_dir(out_folder)
    for idx, mask in enumerate(frames_masks):
        mask_img = (mask * 255).astype(np.uint8)
        write_png(os.path.join(out_folder, f"mask_{idx:03d}.png"), mask_img)

def create_video_from_masks(frames_masks, video_path, fps=10, codec=None):
    F, H, W = frames_masks.shape
    ensure_dir(os.path.dirname(video_path) or ".")
    if codec is None:
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ('.mp4', '.m4v'):
            codec = 'mp4v'
        elif ext in ('.avi',):
            codec = 'XVID'
        else:
            codec = 'mp4v'
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(video_path, fourcc, fps, (W, H), isColor=True)
    for mask in frames_masks:
        if mask.dtype != np.uint8:
            gray = (mask * 255).astype(np.uint8)
        else:
            if mask.max() <= 1:
                gray = (mask * 255).astype(np.uint8)
            else:
                gray = mask
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        out.write(bgr)
    out.release()
    print("Video saved at", video_path)

def plot_frames(frames, num_frames, save_name):
    total_frames = len(frames)
    num_frames = min(num_frames, total_frames)
    cols = min(5, num_frames)
    rows = max(1, (num_frames + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for i in range(num_frames):
        axes_arr[i].imshow(frames[i], cmap="gray", vmin=0, vmax=255)
        axes_arr[i].axis("off")
        axes_arr[i].set_title(f"Frame {i}")
    for j in range(num_frames, len(axes_arr)):
        axes_arr[j].axis("off")
    plt.tight_layout()
    plt.savefig(save_name, format="pdf")
    plt.close(fig)
