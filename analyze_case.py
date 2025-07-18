import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.interpolate import griddata
from collections import defaultdict
import os

def analyze_case(filepath, label, output_folder):
    print(f"\nAnalyzing: {filepath}")

    # === 1. Load Voltage Data ===
    df = pd.read_csv(filepath)
    voltages = df["voltage"].tolist()

    # === 2. Define Grid Points ===
    manual_points = [(0, 0), (0, 1)]
    outer_x = np.arange(0, 31, 1)
    middle_x = np.arange(13, 31, 1)
    inner_x = np.arange(16, 31, 1)

    all_x = np.concatenate([[p[0] for p in manual_points], outer_x, middle_x, inner_x])
    all_y = np.concatenate([[p[1] for p in manual_points], np.full_like(outer_x, 2),
                            np.full_like(middle_x, 1), np.full_like(inner_x, 0)])
    measurement_points = list(zip(all_x, all_y))

    # === 3. King's Law ===
    A = 5.03404726390578
    B = 1.97278924979119
    n = 0.433228583309451

    def voltage_to_velocity(U):
        U_squared = U**2
        numerator = U_squared - A
        return (numerator / B) ** (1 / n) if numerator > 0 else 0

    # === 4. Process Signals ===
    values_per_point = 10000
    velocity_data = {}
    mean_velocity = {}
    TI_field = {}
    TKE_field = {}

    for i, point in enumerate(measurement_points):
        start = i * values_per_point
        end = start + values_per_point
        U = voltages[start:end]
        V = [voltage_to_velocity(u) for u in U]
        velocity_data[point] = V

        u_mean = np.mean(V)
        mean_velocity[point] = u_mean
        u_rms = np.sqrt(np.mean((np.array(V) - u_mean)**2))
        TI = u_rms / u_mean if u_mean != 0 else np.nan
        TKE = 0.5 * u_rms**2
        TI_field[point] = TI
        TKE_field[point] = TKE

    # === 5. Plot TI and TKE on Grid (No Interpolation) ===
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sc1 = axs[0].scatter(all_x, all_y, c=[TI_field[pt] for pt in measurement_points], cmap='jet', s=100)
    axs[0].set_title(f"Turbulence Intensity (TI) - {label}")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].grid(True)
    axs[0].set_aspect('equal')
    fig.colorbar(sc1, ax=axs[0], label="TI")

    sc2 = axs[1].scatter(all_x, all_y, c=[TKE_field[pt] for pt in measurement_points], cmap='jet', s=100)
    axs[1].set_title(f"Turbulent Kinetic Energy (TKE) - {label}")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].grid(True)
    axs[1].set_aspect('equal')
    fig.colorbar(sc2, ax=axs[1], label="TKE (m^2/s^2)")

    plt.tight_layout()
    plt.show()

    # === 6. FFT & PSD for Point (18, 0) ===
    fs = 2000
    target_point = (18, 0)
    V_target = np.array(velocity_data[target_point])
    fluctuations = V_target - np.mean(V_target)

    N = len(fluctuations)
    frequencies = fftfreq(N, d=1/fs)[:N//2]
    fft_vals = fft(fluctuations)
    fft_magnitude = np.abs(fft_vals[:N//2])
    psd = (2.0 / (fs * N)) * np.abs(fft_vals[:N//2])**2

    # Welch
    f_welch, psd_welch = welch(fluctuations, fs=fs, nperseg=1024, noverlap=512)

    # Averaged PSD
    points_to_average = [(18, 0), (17, 0), (18, 1), (19, 0)]
    psd_sum = 0
    for pt in points_to_average:
        V = velocity_data[pt]
        V_fluct = V - np.mean(V)
        _, psd_pt = welch(V_fluct, fs=fs, nperseg=1024, noverlap=512)
        psd_sum += psd_pt
    psd_avg = psd_sum / len(points_to_average)

    # Plot FFT and PSD
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].loglog(frequencies[1:], fft_magnitude[1:], color='orange')
    axs[0].set_title("FFT Magnitude Spectrum at (18, 0)")
    axs[0].set_xlabel("Frequency [Hz]")
    axs[0].set_ylabel("Magnitude")
    axs[0].grid(True, which='both', linestyle='--', alpha=0.5)

    axs[1].loglog(f_welch[1:], psd_avg[1:], label="Averaged PSD", color='blue')
    ref_freq = f_welch[100:500]
    ref_line = 1e-3 * (ref_freq / ref_freq[0])**(-5/3)
    axs[1].loglog(ref_freq, ref_line, 'k--', label="–5/3 Slope")
    axs[1].set_title("Smoothed PSD via Welch at (18, 0)")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("PSD [$m^2/s^2$/Hz]")
    axs[1].grid(True, which='both', linestyle='--', alpha=0.5)
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # === 7. Kolmogorov Scales ===
    dissipation_field = {}
    dx = 1.0
    rows_by_y = defaultdict(list)
    for (x, y) in TKE_field:
        rows_by_y[y].append((x, y))

    for y, row in rows_by_y.items():
        row_sorted = sorted(row, key=lambda pt: pt[0])
        for i, pt in enumerate(row_sorted):
            if i == 0:
                dK_dx = (TKE_field[row_sorted[1]] - TKE_field[pt]) / dx
            elif i == len(row_sorted) - 1:
                dK_dx = (TKE_field[pt] - TKE_field[row_sorted[i-1]]) / dx
            else:
                dK_dx = (TKE_field[row_sorted[i+1]] - TKE_field[row_sorted[i-1]]) / (2 * dx)
            U_mean = mean_velocity[pt]
            epsilon = -U_mean * dK_dx
            dissipation_field[pt] = epsilon

    nu = 1.5e-5
    sorted_eps = sorted(dissipation_field.items(), key=lambda x: x[1])
    min_pt, min_eps = sorted_eps[0]
    max_pt, max_eps = sorted_eps[-1]

    def kolmogorov_scales(eps):
        if eps > 0:
            tau = (nu / eps)**0.5
            u = (nu * eps)**0.25
            eta = (nu**3 / eps)**0.25
            return tau, u, eta
        else:
            return np.nan, np.nan, np.nan

    tau_min, u_min, eta_min = kolmogorov_scales(min_eps)
    tau_max, u_max, eta_max = kolmogorov_scales(max_eps)

    print(f"\nKolmogorov Scales for {label}:")
    print(f"  Min ε at {min_pt}: ε = {min_eps:.2e}, η = {eta_min:.4e} m")
    print(f"  Max ε at {max_pt}: ε = {max_eps:.2e}, η = {eta_max:.4e} m")

    # === 8. Interpolation and Symmetry ===
    x_top = np.array([pt[0] for pt in measurement_points])
    y_top = np.array([pt[1] for pt in measurement_points])
    TI_top = np.array([TI_field[pt] for pt in measurement_points])
    TKE_top = np.array([TKE_field[pt] for pt in measurement_points])

    x_full = np.concatenate([x_top, x_top])
    y_full = np.concatenate([y_top, -y_top])
    TI_full = np.concatenate([TI_top, TI_top])
    TKE_full = np.concatenate([TKE_top, TKE_top])

    grid_x, grid_y = np.mgrid[0:31:200j, -2:2:200j]
    ti_grid = griddata((x_full, y_full), TI_full, (grid_x, grid_y), method='cubic')
    tke_grid = griddata((x_full, y_full), TKE_full, (grid_x, grid_y), method='cubic')

    # === 9. Plot Interpolated TI and TKE ===
    plt.figure(figsize=(7, 5))
    plt.imshow(ti_grid.T, extent=(0, 31, -2, 2), origin='lower', cmap='jet', aspect='auto')
    plt.title(f"Interpolated TI Field - {label}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="TI")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(tke_grid.T, extent=(0, 31, -2, 2), origin='lower', cmap='jet', aspect='auto')
    plt.title(f"Interpolated TKE Field - {label}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="TKE (m²/s²)")
    plt.tight_layout()
    plt.show()
