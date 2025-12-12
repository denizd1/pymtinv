import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

import time


from pymtinv.forward import MT2DForward
from pymtinv.backward import MT2DGradient
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer

# YENİ: Otomatik Beta Seçiciyi çağırdık
from pymtinv.utils import find_optimal_beta_fast, InversionMonitor


# --- 2. Karmaşık Model Oluşturucu ---
def create_complex_model(mesh, n_pad_y):
    # Arkaplan: 100 ohm-m (sigma = 0.01) -> log10 = -2.0
    sigma = np.ones((mesh.Ny, mesh.Nz)) * 0.01

    # A. İletken Yüzey Tabakası
    sigma[:, 0:2] = 0.1

    # Padding kaydırması
    core_start_y = n_pad_y

    # B. Derin İletken Blok (Sol)
    y1, y2 = core_start_y + 4, core_start_y + 8
    z1, z2 = 8, 12
    sigma[y1:y2, z1:z2] = 0.1

    # C. Dirençli Blok (Sağ)
    y3, y4 = core_start_y + 12, core_start_y + 16
    z3, z4 = 6, 10
    sigma[y3:y4, z3:z4] = 0.001

    return sigma


def run_complex_inversion():
    # 1. Grid Ayarları (Paddingli)
    N_PAD_Y = 5
    N_PAD_Z = 5

    mesh = create_padded_mesh(
        core_width=10000,
        core_depth=5000,
        core_dy=500,
        core_dz=250,
        pad_factor=1.4,
        n_pad_y=N_PAD_Y,
        n_pad_z=N_PAD_Z,
    )

    # Logaritmik Frekanslar (100 Hz -> 0.01 Hz)
    frequencies = np.logspace(2, -2, 10)
    print(f"Frekanslar: {np.round(frequencies, 3)}")

    # 2. Gerçek Modeli Oluştur
    sigma_true = create_complex_model(mesh, N_PAD_Y)

    # 3. Sentetik Veri Üret
    print("Forward model hesaplaniyor (Sentetik Veri)...")
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, sigma_true)

    Z_obs = []
    data_std = []
    np.random.seed(42)

    for f in frequencies:
        z_val = Z_true[f]
        noise_std = 0.02 * np.abs(z_val)  # %2 Gürültü
        noise = noise_std * (
            np.random.randn(*z_val.shape) + 1j * np.random.randn(*z_val.shape)
        )
        Z_obs.append(z_val + noise)
        data_std.append(noise_std)

    # 4. Inversion Hazırlığı
    grad_engine = MT2DGradient(fwd, Z_obs, data_std)

    # Başlangıç Modeli: Homojen
    sigma_init = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    m_init = np.log10(sigma_init).flatten()

    # --- YENİ BÖLÜM: OTOMATİK BETA SEÇİMİ ---
    # Hedef Misfit = Veri Sayısı / 2
    # Veri Sayısı = Frekans * İstasyon Sayısı (Ny) * 2 (Reel+Sanal)
    n_data = len(frequencies) * mesh.Ny * 2
    target_phi = n_data / 2.0
    print(f"\n [Bilgi] Hedeflenen Misfit (Target Phi): {target_phi:.1f}")

    # Otomatik Beta Bulucu
    OPTIMAL_BETA = find_optimal_beta_fast(
        grad_engine, frequencies, mesh, m_init, target_phi
    )

    # 5. Ana Optimizasyon
    print(f"\n [Main] Inversion Başlıyor (Seçilen Beta={OPTIMAL_BETA})...")
    monitor = InversionMonitor()

    def objective_function(m_vec):
        sigma_curr = 10 ** m_vec.reshape((mesh.Ny, mesh.Nz))
        # Artık otomatik bulunan Beta'yı kullanıyoruz
        grad_m, phi = grad_engine.compute_gradient(
            frequencies, sigma_curr, beta=OPTIMAL_BETA
        )
        monitor.store_stats(phi, grad_m.flatten())
        return phi, grad_m.flatten()

    bounds = [(-4, 0) for _ in range(len(m_init))]

    result = minimize(
        fun=objective_function,
        x0=m_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        callback=monitor.callback,
        options={"disp": False, "maxiter": 100},
    )

    print(f"\nSonuç: {result.message}")

    # 6. Görselleştirme
    m_final = result.x.reshape((mesh.Ny, mesh.Nz))
    sigma_true_log = np.log10(sigma_true)
    sigma_init_log = np.log10(sigma_init)

    plt.figure(figsize=(16, 6), constrained_layout=True)
    vmin, vmax = -3, -1

    # A) Gerçek Model
    ax1 = plt.subplot(1, 3, 1)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_true_log,
        ax=ax1,
        title="Gerçek Model (Complex)",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # B) Başlangıç
    ax2 = plt.subplot(1, 3, 2)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_init_log,
        ax=ax2,
        title="Başlangıç (Homojen)",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # C) Inversion Sonucu
    ax3 = plt.subplot(1, 3, 3)
    cbar, _ = MT2DVisualizer.plot_model(
        mesh,
        m_final,
        ax=ax3,
        title=f"Inversion Sonucu (Beta={OPTIMAL_BETA})",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(
        cbar, ax=[ax1, ax2, ax3], label="log10(Conductivity)", fraction=0.02, pad=0.04
    )
    plt.show()


if __name__ == "__main__":
    run_complex_inversion()
