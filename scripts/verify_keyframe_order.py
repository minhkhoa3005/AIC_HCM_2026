import csv
import json
import os
import sys
import numpy as np
from pathlib import Path

ROOT = Path("data")
KEYFRAMES_DIR = ROOT / "keyframes"
MAP_DIR = ROOT / "map-keyframes"
FEAT_DIR = ROOT / "clip-features"

def verify_video(video_id):
    print(f"=== Verifying {video_id} ===")
    
    # 1. Get JPGs using the exact logic from import_btc_data.py
    jpg_files = sorted((KEYFRAMES_DIR / video_id).rglob("*.jpg"))
    jpg_stems = [p.stem for p in jpg_files]
    
    if not jpg_stems:
        print(f"  [!] No JPGs found for {video_id}")
        return False
        
    print(f"  [+] Found {len(jpg_stems)} JPGs, sorted: {jpg_stems[0]} ... {jpg_stems[-1]}")
    
    # 2. Get the NPY shape
    npy_path = FEAT_DIR / f"{video_id}.npy"
    if npy_path.exists():
        data = np.load(npy_path)
        print(f"  [+] NPY shape: {data.shape}")
        if data.shape[0] != len(jpg_stems):
            print(f"  [ERROR] NPY vectors ({data.shape[0]}) != JPG count ({len(jpg_stems)})")
            return False
    else:
        print(f"  [?] No NPY found for {video_id}")
        
    # 3. Read MAP-KEYFRAMES CSV
    csv_path = MAP_DIR / f"{video_id}.csv"
    if not csv_path.exists():
        print(f"  [?] No CSV found for {video_id}")
        return True
        
    csv_frames = []
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Depending on CSV format, try to get the frame name or n
            key = row.get("frame_name") or row.get("filename") or row.get("n")
            if key is not None:
                # Try zero padding if it's just an integer
                try:
                    n = int(key)
                    # Often the images are zero padded like 001.jpg, 0001.jpg
                    # Let's see how many digits are used in JPG stems
                    digit_len = len(jpg_stems[0])
                    csv_frames.append(f"{n:0{digit_len}d}")
                except ValueError:
                    csv_frames.append(str(key).replace('.jpg', ''))

    if not csv_frames:
        print(f"  [?] Could not parse frame names from CSV for {video_id}")
        return True
        
    print(f"  [+] CSV has {len(csv_frames)} entries, first: {csv_frames[0]}, last: {csv_frames[-1]}")
    
    # Check lengths
    if len(csv_frames) != len(jpg_stems):
        print(f"  [ERROR] CSV length ({len(csv_frames)}) != JPG count ({len(jpg_stems)})")
        return False
        
    # Check element-by-element exact match
    mismatch_count = 0
    for i, (jpg, csv_f) in enumerate(zip(jpg_stems, csv_frames)):
        if jpg != csv_f:
            mismatch_count += 1
            if mismatch_count <= 3:
                print(f"  [ERROR] Mismatch at ordinal {i}: JPG={jpg} vs CSV={csv_f}")
                
    if mismatch_count > 0:
        print(f"  [ERROR] Total {mismatch_count} order mismatches!")
        return False
        
    print("  [OK] Order matches exactly!")
    return True

if __name__ == "__main__":
    videos_to_test = ["L21_V001", "L21_V010", "L21_V030", "L22_V001", "L22_V030"]
    all_ok = True
    for vid in videos_to_test:
        if (KEYFRAMES_DIR / vid).exists():
            ok = verify_video(vid)
            all_ok = all_ok and ok
            print()
            
    if all_ok:
        print("ALL CHECKS PASSED: The sorting assumption is SAFE for these samples.")
    else:
        print("SOME CHECKS FAILED: You have alignment issues!")
