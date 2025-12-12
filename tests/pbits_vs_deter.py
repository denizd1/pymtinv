import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys
import os
import time

# Kütüphane yollarını ekle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymtinv.forward import MT2DForward
from pymtinv.backward import MT2DGradient
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer

# YENİ: Otomasyon Araçları
from pymtinv.utils import (
    find_optimal_beta_fast,
    tune_pbit_hyperparameters,
    pbit_optimizer,
)


# --- 1. Tek Bloklu Basit Model (Padding Uyumlu) ---
def create_single_block_model(mesh, n_pad_y):
    """
    Homojen arkaplan ortasında tek bir iletken blok.
    """
    # Arkaplan: 100 ohm-m (log = -2.0)
    sigma_log = np.ones((mesh.Ny, mesh.Nz)) * -2.0

    # Blok Konumu (Core bölgesinin ortası)
    cy = n_pad_y + (mesh.Ny - 2 * n_pad_y) // 2
    cz = mesh.Nz // 2

    # 10 ohm-m (log = -1.0) İletken Blok (4x4 hücre)
    sigma_log[cy - 3 : cy + 3, cz - 3 : cz + 3] = -1.0

    return sigma_log


def main():
    print("=======================================================")
    print("   GRAND BENCHMARK: L-BFGS vs P-BITS (AUTO-TUNED)")
    print("=======================================================\n")

    # --- AYARLAR ---
    N_PAD = 6
    mesh = create_padded_mesh(
        core_width=10000,
        core_depth=6000,
        core_dy=500,
        core_dz=250,
        pad_factor=1.3,
        n_pad_y=N_PAD,
        n_pad_z=N_PAD,
    )

    # Frekanslar (Logaritmik)
    frequencies = np.logspace(2, -2.5, 10)

    # --- A. VERİ ÜRETİMİ ---
    print("\n [1/5] Veri Üretiliyor (Single Block)...")
    sigma_true_log = create_single_block_model(mesh, N_PAD)

    # Forward (Linear Sigma ile)
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, 10**sigma_true_log)

    # %5 Gürültü Ekle
    Z_obs, data_std = [], []
    np.random.seed(42)
    for f in frequencies:
        val = Z_true[f]
        err = 0.05 * np.abs(val)
        noise = err * (np.random.randn(*val.shape) + 1j * np.random.randn(*val.shape))
        Z_obs.append(val + noise)
        data_std.append(err)

    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    m_init = (np.ones((mesh.Ny, mesh.Nz)) * -2.0).flatten()
    bounds = [(-4, 0) for _ in range(len(m_init))]

    # --- B. OTOMATİK BETA SEÇİMİ ---
    target_phi = (len(frequencies) * mesh.Ny * 2) / 2.0
    optimal_beta = find_optimal_beta_fast(
        grad_engine, frequencies, mesh, m_init, target_phi
    )

    # --- C. L-BFGS (CPU) ---
    print(f"\n [2/5] L-BFGS (Klasik Yöntem) Çalışıyor (Beta={optimal_beta})...")
    start_cpu = time.time()

    def objective(m):
        sig = 10 ** m.reshape((mesh.Ny, mesh.Nz))
        g, p = grad_engine.compute_gradient(frequencies, sig, beta=optimal_beta)
        return p, g.flatten()

    res_lbfgs = minimize(
        objective,
        m_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": 60},
    )
    time_lbfgs = time.time() - start_cpu
    phi_lbfgs = res_lbfgs.fun
    m_lbfgs = res_lbfgs.x.reshape((mesh.Ny, mesh.Nz))
    print(f"   -> Süre: {time_lbfgs:.2f}s | Phi: {phi_lbfgs:.1f}")

    # --- D. P-BITS TUNING ---
    print(f"\n [3/5] P-bits Hiperparametreleri Ayarlanıyor...")
    best_lr, best_t_start = tune_pbit_hyperparameters(
        grad_engine, frequencies, mesh, m_init, beta=optimal_beta
    )

    # --- E. P-BITS SİMÜLASYONU ---
    # Python'da beklememek için 3.000 adım atıyoruz.
    # AVERAGING: Sonucun daha pürüzsüz olması için Averaging mekanizması P-bit içinde yoktu,
    # burada P-bit'in son 3000 adımını yaparken basit bir averaging ekleyebiliriz veya
    # direkt son sonucu alabiliriz. Görsel net olsun diye son sonucu alacağız.
    P_STEPS = 3000
    print(f"\n [4/5] P-bits Simülasyonu ({P_STEPS} adım, LR={best_lr})...")
    start_pbit = time.time()

    # Standart optimizer çağrısı
    m_pbit_flat, _, phi_pbit = pbit_optimizer(
        grad_engine,
        m_init,
        frequencies,
        mesh,
        n_steps=P_STEPS,
        lr=best_lr,
        t_start=best_t_start,
        t_end=0.0001,
        beta=optimal_beta,
    )

    time_pbit_python = time.time() - start_pbit
    m_pbit = m_pbit_flat.reshape((mesh.Ny, mesh.Nz))
    print(f"   -> Süre: {time_pbit_python:.2f}s | Best Phi: {phi_pbit:.1f}")

    # --- F. FPGA PROJEKSİYONU ---
    FPGA_CLOCK_NS = 5e-9  # 200 MHz
    REAL_P_STEPS = 100000  # Gerçek senaryo
    time_fpga_theory = REAL_P_STEPS * FPGA_CLOCK_NS

    print(f"\n [5/5] Analiz Tamamlandı.")
    print(f"   -> L-BFGS       : {time_lbfgs:.4f} s")
    print(f"   -> Python P-bit : {time_pbit_python:.4f} s")
    print(f"   -> FPGA P-bit   : {time_fpga_theory:.8f} s")

    # --- GÖRSELLEŞTİRME ---
    plt.figure(figsize=(16, 8), constrained_layout=True)
    vmin, vmax = -3, -1

    # 1. Gerçek Model
    ax1 = plt.subplot(2, 2, 1)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_true_log,
        ax=ax1,
        title="Gerçek Model",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # 2. L-BFGS Sonucu
    ax2 = plt.subplot(2, 2, 2)
    MT2DVisualizer.plot_model(
        mesh,
        m_lbfgs,
        ax=ax2,
        title=f"L-BFGS (Phi={phi_lbfgs:.0f})",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # 3. P-bits Sonucu
    ax3 = plt.subplot(2, 2, 3)
    MT2DVisualizer.plot_model(
        mesh,
        m_pbit,
        ax=ax3,
        title=f"Auto-Pbits (Phi={phi_pbit:.0f})",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # 4. Performans Bar Grafiği
    ax4 = plt.subplot(2, 2, 4)
    times = [time_lbfgs, time_pbit_python, time_fpga_theory]
    labels = ["L-BFGS (CPU)", "P-bits (Python)", "P-bits (FPGA)"]
    colors = ["gray", "blue", "red"]

    bars = ax4.barh(labels, times, color=colors)
    ax4.set_xscale("log")
    ax4.set_xlabel("Hesaplama Süresi (Saniye) - Log Scale")
    ax4.set_title("Donanım Performans Kıyaslaması")
    ax4.grid(True, which="both", ls="--", alpha=0.5)

    for bar, val in zip(bars, times):
        ax4.text(
            val * 1.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1e} s",
            va="center",
            fontweight="bold",
        )

    plt.show()


if __name__ == "__main__":
    main()
