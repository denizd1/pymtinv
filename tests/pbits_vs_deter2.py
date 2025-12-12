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


# --- 1. Tek Bloklu Basit Model ---
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


# --- 2. P-bit Optimizer (AVERAGING EKLENDİ) ---
def pbit_optimizer_benchmark(
    grad_engine, m_init, freqs, mesh, n_steps, lr, t_start, t_end
):
    m_current = m_init.copy()

    # Averaging (Ortalama Alma) için hazırlık
    # Son %20'lik kısımdaki modelleri toplayıp ortalamasını alacağız.
    avg_start_step = int(n_steps * 0.8)
    m_sum = np.zeros_like(m_current)
    count_avg = 0

    best_m = m_current.copy()
    best_phi = 1e9

    print(f"   P-bit Progress: 0/{n_steps}", end="\r")

    for step in range(n_steps):
        # Cooling
        decay = (t_end / t_start) ** (step / n_steps)
        temp = t_start * decay

        # Gradient
        sigma_lin = 10 ** m_current.reshape((mesh.Ny, mesh.Nz))
        grad, phi = grad_engine.compute_gradient(freqs, sigma_lin, beta=5.0)  # Beta=5.0
        grad = grad.flatten()

        if phi < best_phi:
            best_phi = phi
            best_m = m_current.copy()

        # Update
        noise = np.sqrt(2 * temp) * np.random.randn(*m_current.shape)
        m_current += -lr * grad + noise
        m_current = np.clip(m_current, -4.0, 0.0)

        # --- AVERAGING MEKANİZMASI ---
        # Son %20'lik dilimde modelin "fotoğraflarını" üst üste bindiriyoruz
        if step >= avg_start_step:
            m_sum += m_current
            count_avg += 1

        if step % 100 == 0:
            print(f"   P-bit Progress: {step}/{n_steps} (Phi={phi:.1f})", end="\r")

    print(f"   P-bit Progress: {n_steps}/{n_steps} (Done)        ")

    # Ortalama Modeli Hesapla
    m_mean = m_sum / count_avg

    return m_mean, best_phi


def main():
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

    frequencies = np.logspace(2, -2.5, 10)

    # --- A. VERİ ÜRETİMİ ---
    print("\n [1/4] Veri Üretiliyor (Single Block)...")
    sigma_true_log = create_single_block_model(mesh, N_PAD)

    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, 10**sigma_true_log)

    # %5 Gürültü
    Z_obs, data_std = [], []
    np.random.seed(42)
    for f in frequencies:
        val = Z_true[f]
        err = 0.05 * np.abs(val)
        Z_obs.append(
            val + err * (np.random.randn(*val.shape) + 1j * np.random.randn(*val.shape))
        )
        data_std.append(err)

    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    m_init = (np.ones((mesh.Ny, mesh.Nz)) * -2.0).flatten()
    bounds = [(-4, 0) for _ in range(len(m_init))]

    # --- B. L-BFGS (CPU) ---
    print("\n [2/4] L-BFGS (Klasik Yöntem) Çalışıyor...")
    start_cpu = time.time()

    def objective(m):
        sig = 10 ** m.reshape((mesh.Ny, mesh.Nz))
        g, p = grad_engine.compute_gradient(frequencies, sig, beta=5.0)  # Beta=5.0
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

    # --- C. P-BITS (PYTHON SIMULATION) ---
    # Python'da 3.000 adım atalım (Benchmark için)
    P_STEPS = 3000
    print(f"\n [3/4] P-bits Simülasyonu ({P_STEPS} adım)...")
    start_pbit = time.time()

    # DİKKAT: Artık ortalama alınmış (daha pürüzsüz) model dönüyor
    m_pbit_mean_flat, phi_pbit = pbit_optimizer_benchmark(
        grad_engine,
        m_init,
        frequencies,
        mesh,
        n_steps=P_STEPS,
        lr=0.002,
        t_start=0.1,
        t_end=0.001,
    )
    time_pbit_python = time.time() - start_pbit
    m_pbit = m_pbit_mean_flat.reshape((mesh.Ny, mesh.Nz))
    print(f"   -> Süre: {time_pbit_python:.2f}s | Best Phi: {phi_pbit:.1f}")

    # --- D. FPGA PROJEKSİYONU (TEORİK HESAP) ---
    # FPGA: 200 MHz Clock, Her adım 1 Cycle
    FPGA_CLOCK_NS = 5e-9

    # Adil karşılaştırma için: Gerçek senaryoda 100.000 adım
    REAL_P_STEPS = 100000
    time_fpga_theory = REAL_P_STEPS * FPGA_CLOCK_NS

    print(f"\n [4/4] Analiz Tamamlandı.")

    # --- GÖRSELLEŞTİRME ---
    plt.figure(figsize=(16, 8), constrained_layout=True)
    vmin, vmax = -3, -1

    # 1. Gerçek Model
    ax1 = plt.subplot(2, 2, 1)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_true_log,
        ax=ax1,
        title="Gerçek Model (Single Block)",
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

    # 3. P-bits Sonucu (MEAN MODEL)
    ax3 = plt.subplot(2, 2, 3)
    MT2DVisualizer.plot_model(
        mesh,
        m_pbit,
        ax=ax3,
        title=f"P-bits Simülasyonu (Phi={phi_pbit:.0f})\n(Mean Model of Last 20%)",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # 4. Performans Bar Grafiği (LOG SCALE)
    ax4 = plt.subplot(2, 2, 4)
    times = [time_lbfgs, time_pbit_python, time_fpga_theory]
    labels = ["L-BFGS (CPU)", "P-bits (Python)", "P-bits (FPGA)"]
    colors = ["gray", "blue", "red"]

    bars = ax4.barh(labels, times, color=colors)
    ax4.set_xscale("log")  # Logaritmik skala
    ax4.set_xlabel("Hesaplama Süresi (Saniye) - Log Scale")
    ax4.set_title("Donanım Performans Kıyaslaması")
    ax4.grid(True, which="both", ls="--", alpha=0.5)

    # Değerleri yaz
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
