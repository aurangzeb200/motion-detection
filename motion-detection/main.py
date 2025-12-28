import argparse
import os
import numpy as np
import imageio.v2 as iio

from io_utils import write_png, ensure_dir, scale_to_uint8, create_video_from_masks
from background import compute_mean, compute_variance, compute_mask
from morphology import create_kernel, erode, dilate, morphological_operations
from components import find_connected_components

def read_frames(input_folder: str, input_ext: str = "png") -> np.ndarray:
    ext = input_ext.lower().lstrip('.')
    files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(f".{ext}")])
    if len(files) == 0:
        raise RuntimeError(f"No image files with extension .{ext} found in {input_folder}")
    frames = []
    for fn in files:
        path = os.path.join(input_folder, fn)
        img = iio.imread(path)
        if img.ndim == 2:
            rgb = np.stack([img]*3, axis=-1)
        else:
            rgb = img[..., :3]
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2])
        frames.append(gray)
    return np.stack(frames, axis=0).astype(np.float32)


def remove_person(frames_with_person, background_frame, mask_sequence=None, start_alpha=0.0, end_alpha=1.0):
    F = frames_with_person.shape[0]
    H, W = frames_with_person.shape[1:]
    if background_frame.ndim == 2:
        bg_stack = np.stack([background_frame] * F, axis=0)
    else:
        bg_stack = background_frame
    if mask_sequence is None:
        masks = np.zeros((F, H, W), dtype=np.uint8)
    else:
        if mask_sequence.ndim == 2:
            masks = np.stack([mask_sequence] * F, axis=0)
        else:
            masks = mask_sequence.astype(np.uint8)

    out = np.zeros((F, H, W), dtype=np.uint8)
    alphas = np.linspace(start_alpha, end_alpha, F)
    for f in range(F):
        k = frames_with_person[f].astype(np.float32)
        t = bg_stack[f].astype(np.float32)
        maskf = masks[f].astype(np.float32)  # 0 or 1
        a = alphas[f]
        blended = k * (1 - a * maskf) + t * (a * maskf)
        out[f] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def changeDetection(input_folder, output_folder, input_ext='png', output_ext='png', video_format='mp4'):
    ensure_dir(output_folder)
    output_ext = (output_ext or 'png').lower()
    if output_ext != 'png':
        print(f"Note: outputs will be saved as .png regardless of requested {output_ext}")

    name = os.path.basename(input_folder).lower()
    t = 70 if 'person1' in name else 60

    print("Reading frames...")
    frames = read_frames(input_folder, input_ext)
    F, H, W = frames.shape
    print(f"Loaded {F} frames ({H}x{W})")

    use_t = min(t, F)
    print(f"Building background model with {use_t} frames...")
    mean_frame = compute_mean(frames[:use_t])
    var_frame = compute_variance(frames[:use_t], mean_frame)

    write_png(os.path.join(output_folder, "mean_frame.png"), scale_to_uint8(mean_frame))
    write_png(os.path.join(output_folder, "variance_frame.png"), scale_to_uint8(var_frame))

    thresholds = [1.0, 1.5, 2.0]
    kernels = [3, 5, 7]

    cleaned_stacks = {}
    raw_masks_by_thr = {}

    for thr in thresholds:
        thr_name = str(thr).replace('.', '_')
        thr_dir = os.path.join(output_folder, f"threshold_{thr_name}")
        ensure_dir(thr_dir)

        print(f"Processing threshold {thr} ...")
        raw_masks = []
        raw_dir = os.path.join(thr_dir, "raw_masks")
        ensure_dir(raw_dir)

        for idx in range(F):
            mask = compute_mask(frames[idx], mean_frame, var_frame, threshold=thr)
            raw_masks.append(mask)
            write_png(os.path.join(raw_dir, f"mask_{idx:03d}.png"), (mask * 255).astype(np.uint8))
        raw_masks = np.stack(raw_masks, axis=0)
        raw_masks_by_thr[thr] = raw_masks

        create_video_from_masks(raw_masks, os.path.join(thr_dir, f"raw_masks_thr_{thr_name}.{video_format}"), fps=10)

        if thr == 1.0:
            for k in kernels:
                k_dir = os.path.join(thr_dir, f"kernel_{k}x{k}")
                erode_dir = os.path.join(k_dir, "eroded")
                dilate_dir = os.path.join(k_dir, "dilated")
                cleaned_dir = os.path.join(k_dir, "cleaned")
                ensure_dir(erode_dir); ensure_dir(dilate_dir); ensure_dir(cleaned_dir)

                print(f"Applying morphological ops for thr={thr}, kernel {k}x{k} ...")
                eroded_stack, dilated_stack, cleaned_stack = [], [], []
                kernel = create_kernel(k)
                anchor = (k // 2, k // 2)

                for idx in range(F):
                    raw = raw_masks[idx]
                    erd = erode(raw, kernel, anchor=anchor)
                    dil = dilate(raw, kernel, anchor=anchor)
                    cleaned = morphological_operations(raw, kernel_size=k)

                    eroded_stack.append(erd)
                    dilated_stack.append(dil)
                    cleaned_stack.append(cleaned)

                    write_png(os.path.join(erode_dir, f"mask_{idx:03d}_eroded.png"), (erd*255).astype(np.uint8))
                    write_png(os.path.join(dilate_dir, f"mask_{idx:03d}_dilated.png"), (dil*255).astype(np.uint8))
                    write_png(os.path.join(cleaned_dir, f"mask_{idx:03d}_cleaned.png"), (cleaned*255).astype(np.uint8))

                cleaned_stack = np.stack(cleaned_stack, axis=0)
                cleaned_stacks[(thr, k)] = cleaned_stack

                create_video_from_masks(np.stack(eroded_stack, axis=0), os.path.join(k_dir, f"eroded_masks_k{k}.{video_format}"), fps=10)
                create_video_from_masks(np.stack(dilated_stack, axis=0), os.path.join(k_dir, f"dilated_masks_k{k}.{video_format}"), fps=10)
                create_video_from_masks(cleaned_stack, os.path.join(k_dir, f"cleaned_masks_k{k}.{video_format}"), fps=10)

    thr = 1.0
    k = 3
    if (thr, k) in cleaned_stacks:
        print("Running Step 6 (Connected Components Filtering)...")
        cleaned_stack = cleaned_stacks[(thr, k)]
        step6_dir = os.path.join(output_folder, "step6_filtered_binary_masks")
        ensure_dir(step6_dir)
        binary_list = []
        comp_summaries = []
        for idx in range(F):
            cleaned = cleaned_stack[idx]
            numc, labeled, info, filtered = find_connected_components(cleaned, connectivity=8, min_area=50, max_area_ratio=0.5)
            binary_list.append(filtered)
            comp_summaries.append({"frame": idx, "num_components": numc, "components": info})
            write_png(os.path.join(step6_dir, f"mask_{idx:03d}_binary.png"), (filtered * 255).astype(np.uint8))
        binary_stack = np.stack(binary_list, axis=0)
        create_video_from_masks(binary_stack, os.path.join(step6_dir, f"binary_masks.{video_format}"), fps=15)
        print(f"Step 6 done: {step6_dir}")

        print("Running Step 7 (Person Removal)...")
        step7_dir = os.path.join(output_folder, "step7_removal")
        ensure_dir(step7_dir)
        removed_frames = remove_person(frames, mean_frame, mask_sequence=binary_stack, start_alpha=0.0, end_alpha=1.0)
        for idx in range(F):
            write_png(os.path.join(step7_dir, f"removed_{idx:03d}.png"), removed_frames[idx])
        create_video_from_masks(removed_frames, os.path.join(step7_dir, f"removed_video.{video_format}"), fps=15)
        print(f"Step 7 done: {step7_dir}")

        step8_dir = os.path.join(output_folder, "step8_final_masks")
        ensure_dir(step8_dir)
        for idx in range(F):
            write_png(os.path.join(step8_dir, f"final_mask_{idx:03d}.png"), (binary_stack[idx] * 255).astype(np.uint8))
        create_video_from_masks(binary_stack, os.path.join(step8_dir, f"final_masks_video.{video_format}"), fps=15)
        print(f"Step 8 done: {step8_dir}")

    else:
        print(f"No cleaned stack found for threshold {thr} and kernel {k}. Skipping Step 6/7.")



def main():
    parser = argparse.ArgumentParser(description='Motion Detection using Classical CV')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to input folder containing frames')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save masks and video')
    parser.add_argument('--input_ext', type=str, default='png', help='Extension of input images (png, jpg, jpeg)')
    parser.add_argument('--output_ext', type=str, default='png', help='Extension of output masks (png, jpg)')
    parser.add_argument('--video_format', type=str, default='mp4', help='Video format for saving output (mp4, avi)')
    args = parser.parse_args()

    changeDetection(args.input_folder, args.output_folder, args.input_ext, args.output_ext, args.video_format)


if __name__ == '__main__':
    main()