"""
========================================================================================
IMNeT: Video Inference Script for Audio-Visual Deepfake Detection
========================================================================================
Accepts an input video file (.mp4, .avi, .mov), preprocesses synchronized audio
and facial motion trajectories, loads a trained IMNeT checkpoint, and outputs
the predicted class with fine-grained probability scores and latency.

Usage:
    python inference.py --video path/to/video.mp4
    python inference.py --video path/to/video.mp4 --checkpoint results/checkpoints/best_imnet_model.pth
========================================================================================
"""

import os
import sys
import time
import json
import argparse
import torch
import numpy as np

from training import IMNeT, IMNeTConfig, AudioPreprocessor, VideoPreprocessor

CLASS_DESCRIPTIONS = {
    0: ("AFVF", "Fake Audio + Fake Video (Both modalities manipulated)"),
    1: ("AFVR", "Fake Audio + Real Video (Audio synthetic/cloned, video authentic)"),
    2: ("ARVF", "Real Audio + Fake Video (Facial reenactment / face swap)"),
    3: ("ARVR", "Real Audio + Real Video (Authentic Genuine Media)")
}

def run_inference(video_path: str, checkpoint_path: str = None, json_output: bool = False):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    t_start = time.perf_counter()
    config = IMNeTConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Preprocessing
    aproc = AudioPreprocessor(target_sr=config.sampling_rate)
    vproc = VideoPreprocessor(target_size=config.frame_size)

    raw_audio = aproc.extract_audio_from_video(video_path)
    audio_feat = aproc.extract_acoustic_features(raw_audio)
    face_tensor, lm_seq = vproc.extract_frames_and_landmarks(video_path, max_frames=50)

    # 2. Model Initialization
    model = IMNeT(config).to(device)

    # Resolve default checkpoint path if not specified
    if checkpoint_path is None:
        default_ckpts = [
            "results/checkpoints/best_imnet_model.pth",
            "checkpoints/best_imnet_model.pth"
        ]
        for p in default_ckpts:
            if os.path.exists(p):
                checkpoint_path = p
                break

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        # If no checkpoint exists yet, warn user
        if not json_output:
            print("[INFO] No pretrained checkpoint specified/found. Running inference with model base weights.")

    model.eval()

    # 3. Model Forward Pass
    audio_t = torch.tensor(audio_feat, dtype=torch.float32).unsqueeze(0).to(device)
    lm_t = torch.tensor(lm_seq, dtype=torch.float32).unsqueeze(0).to(device)
    face_t = torch.tensor(face_tensor, dtype=torch.float32).unsqueeze(0).to(device)

    t_model_start = time.perf_counter()
    with torch.no_grad():
        out = model(audio_t, lm_t, face_t)
        probs = out["probabilities"][0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
    t_model_end = time.perf_counter()

    t_total = (time.perf_counter() - t_start) * 1000.0
    t_model = (t_model_end - t_model_start) * 1000.0

    class_code, class_desc = CLASS_DESCRIPTIONS[pred_idx]
    confidence_pct = float(probs[pred_idx] * 100.0)

    result_data = {
        "video_path": os.path.abspath(video_path),
        "predicted_class": class_code,
        "description": class_desc,
        "confidence_percentage": round(confidence_pct, 2),
        "probabilities": {
            "AFVF": round(float(probs[0] * 100.0), 2),
            "AFVR": round(float(probs[1] * 100.0), 2),
            "ARVF": round(float(probs[2] * 100.0), 2),
            "ARVR": round(float(probs[3] * 100.0), 2),
        },
        "model_latency_ms": round(t_model, 2),
        "total_processing_time_ms": round(t_total, 2),
        "is_authentic": (pred_idx == 3)
    }

    if json_output:
        print(json.dumps(result_data, indent=2))
        return result_data

    # Pretty CLI Printout
    print("\n" + "=" * 68)
    print("  IMNeT AUDIO-VISUAL DEEPFAKE DETECTION REPORT")
    print("=" * 68)
    print(f"  Target File          : {os.path.basename(video_path)}")
    print(f"  Prediction           : [{class_code}] {class_desc}")
    print(f"  Confidence           : {confidence_pct:.2f}%")
    print(f"  Authenticity Status  : {'[GENUINE / AUTHENTIC]' if pred_idx == 3 else '[DEEPFAKE DETECTED]'}")
    print(f"  Model Inference Time : {t_model:.2f} ms")
    print(f"  Total Pipeline Time  : {t_total:.2f} ms")
    print("-" * 68)
    print("  Class Probability Breakdown:")
    for code, idx in [("AFVF", 0), ("AFVR", 1), ("ARVF", 2), ("ARVR", 3)]:
        p_val = probs[idx] * 100.0
        bar = "#" * int(p_val / 4.0)
        marker = " <-- [PREDICTED]" if idx == pred_idx else ""
        print(f"    {code:<4}: {p_val:5.1f}% | {bar:<25}{marker}")
    print("=" * 68 + "\n")

    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMNeT Single-Video Inference CLI")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained IMNeT checkpoint")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    args = parser.parse_args()

    run_inference(args.video, checkpoint_path=args.checkpoint, json_output=args.json)
