import os
import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy.signal import welch
from src import config

def get_unique_rpms(folder_path):
    rpm_values = set()
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return []
    for f in os.listdir(folder_path):
        if f.endswith(".csv"):
            m = re.search(r'(\d+)RPM', f)
            if m:
                rpm_values.add(int(m.group(1)))
    return sorted(rpm_values)

def get_files_by_rpm(folder_path, rpm):
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path)
            if f.endswith(".csv") and re.search(rf'(\b|_){rpm}RPM', f)]

def extract_doc_sorted(files):
    info = []
    for f in files:
        m = re.search(r'(\d+)doc', f)
        if m:
            info.append((f, int(m.group(1))))
    return sorted(info, key=lambda x: x[1])

def detect_cutting_zone(force, percentile=60):
    thr = np.percentile(np.abs(force), percentile)
    idx = np.where(np.abs(force) > thr)[0]
    if len(idx) == 0:
        return slice(0, len(force))
    return slice(int(idx[0]), int(idx[-1]) + 1)

def find_force_columns(columns):
    chans = {}
    for c in columns:
        cl = c.lower()
        if "fz" in cl or "cutting force" in cl:
            chans["Fz"] = c
        elif "fx" in cl or "feed force" in cl:
            chans["Fx"] = c
        elif "fy" in cl or "radial force" in cl:
            chans["Fy"] = c
    if "Fz" not in chans:
        raise ValueError(f"Could not find a Fz/cutting-force column in {list(columns)}")
    return chans

def hand_features(fc, dt, spindle_freq):
    fs = 1.0 / dt
    mean_v   = np.mean(fc)
    rms_v    = np.sqrt(np.mean(fc**2))
    std_v    = np.std(fc)
    ptp_v    = np.ptp(fc)
    kurt_v   = kurtosis(fc)
    skew_v   = skew(fc)
    peak_v   = np.max(np.abs(fc))
    mabs     = np.mean(np.abs(fc)) + 1e-12
    crest_f  = peak_v / (rms_v + 1e-12)
    shape_f  = rms_v / mabs
    impulse_f = peak_v / mabs

    win    = np.hanning(len(fc))
    fft_v  = np.abs(np.fft.rfft(fc * win))
    freq_v = np.fft.rfftfreq(len(fc), dt)
    if len(fft_v) > 5:
        fft_v[:5] = 0
    dom_freq = freq_v[np.argmax(fft_v)] if len(fft_v) else 0.0
    spec_cen = np.sum(freq_v * fft_v**2) / (np.sum(fft_v**2) + 1e-12)
    spec_bw  = np.sqrt(np.sum((freq_v - spec_cen)**2 * fft_v**2) /
                        (np.sum(fft_v**2) + 1e-12))

    total_fft_e = np.sum(fft_v**2)
    harmonic_mask = np.zeros(len(freq_v), dtype=bool)
    for n in range(1, config.N_HARMONICS + 1):
        h = spindle_freq * n
        harmonic_mask |= (freq_v >= h - config.BW_HZ) & (freq_v <= h + config.BW_HZ)
    harmonic_e = np.sum(fft_v[harmonic_mask]**2)
    harmonic_ratio = harmonic_e / (total_fft_e + 1e-12)

    nper = min(1024, max(len(fc) // 4, 8))
    fw, Pxx = welch(fc, fs=fs, nperseg=nper)
    total_e = np.sum(Pxx)
    sp_e = sum(np.sum(Pxx[(fw >= spindle_freq * n - config.BW_HZ) &
                          (fw <= spindle_freq * n + config.BW_HZ)])
               for n in range(1, config.N_HARMONICS + 1))
    ci = float(np.clip(1.0 - sp_e / (total_e + 1e-12), 0, 1))

    feats = np.array([mean_v, rms_v, std_v, ptp_v, kurt_v, skew_v,
                       crest_f, shape_f, impulse_f,
                       dom_freq, spec_cen, spec_bw, harmonic_ratio], dtype=np.float32)
    return feats, ci

def extract_features_from_raw(raw_folder=config.RAW_DATA_DIR, output_csv=config.PROCESSED_DATA_DIR / "features.csv"):
    rows = []
    rpms = get_unique_rpms(raw_folder)
    
    if not rpms:
        print(f"No RPM data found in {raw_folder}. Ensure the CSVs are in this directory.")
        return pd.DataFrame()
        
    for rpm in rpms:
        rpm_files = get_files_by_rpm(raw_folder, rpm)
        rpm_info  = extract_doc_sorted(rpm_files)
        sp_f      = rpm / 60.0

        for file, doc in rpm_info:
            file_path = os.path.join(raw_folder, file)
            try:
                data = pd.read_csv(file_path).dropna().iloc[100:]
                chans = find_force_columns(data.columns)
                
                # Try to find time column
                time_col = [c for c in data.columns if 'time' in c.lower()]
                if time_col:
                    time = data[time_col[0]].values
                    dt = time[1] - time[0]
                else:
                    dt = 0.0001 # fallback typical sampling rate
                    
                fz = data[chans["Fz"]].values
                zone = detect_cutting_zone(fz)
                if zone.stop - zone.start < 16:
                    continue

                fz_cut = fz[zone]
                feats, ci = hand_features(fz_cut, dt, sp_f)

                rows.append({
                    "RPM": rpm, "DOC": doc, "CI": ci,
                    **dict(zip(config.FEATURE_NAMES, feats))
                })
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    df = pd.DataFrame(rows).sort_values(["RPM", "DOC"]).reset_index(drop=True)

    if not df.empty:
        # Create a Stable/Chatter label per RPM based on CI threshold
        thr = df.groupby("RPM")["CI"].apply(lambda g: (g.min() + g.max()) / 2)
        df["Threshold"] = df["RPM"].map(thr)
        df["Label"] = (df["CI"] > df["Threshold"]).astype(int)  # 1 = Chatter, 0 = Stable
        
        # Save to processed dir
        df.to_csv(output_csv, index=False)
        print(f"Successfully extracted features for {len(df)} conditions and saved to {output_csv}")
    
    return df

if __name__ == "__main__":
    print("Starting data processing pipeline...")
    extract_features_from_raw()
