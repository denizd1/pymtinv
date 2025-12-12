import numpy as np
import matplotlib.pyplot as plt

import time


from pymtinv.forward import MT2DForward
from pymtinv.backward import MT2DGradient
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer

# İki otomatik fonksiyonu da çağırıyoruz
from pymtinv.utils import (
    find_optimal_beta_fast,
    tune_pbit_hyperparameters,
    pbit_optimizer,
)


def create_complex_model_log(mesh, n_pad_y):
    """
    Test için LOGARİTMİK (log10) yer altı modeli oluşturur.
    """
    # Arkaplan: 100 ohm-m -> 0.01 S/m -> log10 = -2.0
    sigma_log = np.ones((mesh.Ny, mesh.Nz)) * -2.0

    # Core bölgesine iletken blok (Kırmızı)
    # 10 ohm-m -> 0.1 S/m -> log10 = -1.0
    center_y, center_z = mesh.Ny // 2, mesh.Nz // 2

    # Blok boyutları
    sigma_log[center_y - 4 : center_y + 4, center_z - 4 : center_z + 4] = -1.0

    return sigma_log


def main():
    # 1. SETUP
    N_PAD = 5
    mesh = create_padded_mesh(10000, 5000, 500, 250, 1.4, N_PAD, N_PAD)
    frequencies = np.logspace(2, -2.3, 10)

    print(" [1/5] Veri hazırlanıyor...")
    sigma_true_log = create_complex_model_log(mesh, N_PAD)
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, 10**sigma_true_log)

    Z_obs, data_std = [], []
    np.random.seed(42)
    for f in frequencies:
        val = Z_true[f]
        noise = 0.05 * np.abs(val)
        Z_obs.append(
            val
            + noise * (np.random.randn(*val.shape) + 1j * np.random.randn(*val.shape))
        )
        data_std.append(noise)

    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    m_init = (np.ones((mesh.Ny, mesh.Nz)) * -2.0).flatten()

    # 2. AUTO-BETA (Tikhonov Ayarı)
    target_phi = (len(frequencies) * mesh.Ny * 2) / 2.0
    optimal_beta = find_optimal_beta_fast(
        grad_engine, frequencies, mesh, m_init, target_phi
    )

    # 3. AUTO-PBIT (Hiperparametre Ayarı)
    # Burada optimal_beta'yı da kullanıyoruz ki şartlar eşit olsun
    best_lr, best_t_start = tune_pbit_hyperparameters(
        grad_engine, frequencies, mesh, m_init, beta=optimal_beta
    )

    # 4. FINAL RUN
    # Python'da beklememek için 3000 adım yapıyoruz (FPGA'de 1M olurdu)
    FINAL_STEPS = 3000
    print(f"\n [4/5] Final P-bit Simülasyonu Başlıyor ({FINAL_STEPS} adım)...")
    start_time = time.time()

    m_final, hist, final_phi = pbit_optimizer(
        grad_engine,
        m_init,
        frequencies,
        mesh,
        n_steps=FINAL_STEPS,
        lr=best_lr,
        t_start=best_t_start,
        t_end=0.0001,
        beta=optimal_beta,
    )

    print(
        f" İşlem Tamamlandı: {time.time() - start_time:.2f}s | Final Phi: {final_phi:.1f}"
    )

    # 5. GÖRSELLEŞTİRME
    plt.figure(figsize=(14, 6), constrained_layout=True)

    # Enerji Grafiği
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(hist, "b-", alpha=0.5, label="Anlık Enerji")
    ax1.plot(np.minimum.accumulate(hist), "r-", linewidth=2, label="Best-So-Far")
    ax1.set_title(
        f"Auto-Tuned P-bits\n(LR={best_lr}, T0={best_t_start}, Beta={optimal_beta})"
    )
    ax1.set_xlabel("Adım")
    ax1.set_ylabel("Misfit")
    ax1.legend()
    ax1.grid(True)

    # Model
    ax2 = plt.subplot(1, 2, 2)
    # m_final zaten logaritmik geliyor
    MT2DVisualizer.plot_model(
        mesh,
        m_final.reshape((mesh.Ny, mesh.Nz)),
        ax=ax2,
        title=f"Sonuç (Phi={final_phi:.1f})",
        grid_lines=True,
        vmin=-3,
        vmax=-1,
    )

    plt.show()


if __name__ == "__main__":
    main()
