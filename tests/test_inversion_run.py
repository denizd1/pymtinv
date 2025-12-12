import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pymtinv.forward import MT2DForward
from pymtinv.backward import MT2DGradient
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer

# YENİ: Otomatik Beta bulucuyu import ediyoruz
from pymtinv.utils import find_optimal_beta_fast


def create_synthetic_model_log(mesh):
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
    # --- 1. AYARLAR ---
    # Paddingli Mesh
    mesh = create_padded_mesh(
        core_width=10000,
        core_depth=5000,
        core_dy=500,
        core_dz=250,
        pad_factor=1.4,
        n_pad_y=5,
        n_pad_z=5,
    )

    # YENİ: Frekansları Logaritmik yapıyoruz (Daha iyi derinlik algısı için)
    # 100 Hz'den 0.01 Hz'e kadar 10 adım
    frequencies = np.logspace(2, -2, 10)
    print(f"Kullanılan Frekanslar: {np.round(frequencies, 3)}")

    # --- 2. VERİ ÜRETİMİ ---
    print("\n [1/4] Sentetik veri üretiliyor...")
    sigma_true_log = create_synthetic_model_log(mesh)

    # Forward solver lineer iletkenlik (S/m) ister
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, 10**sigma_true_log)

    # Gürültü Ekleme (%5)
    Z_obs = []
    data_std = []
    np.random.seed(42)
    for f in frequencies:
        z_val = Z_true[f]
        noise_std = 0.05 * np.abs(z_val)
        noise = noise_std * (
            np.random.randn(*z_val.shape) + 1j * np.random.randn(*z_val.shape)
        )
        Z_obs.append(z_val + noise)
        data_std.append(noise_std)

    # --- 3. OTOMATİK HAZIRLIK ---
    grad_engine = MT2DGradient(fwd, Z_obs, data_std)

    # Başlangıç Modeli
    sigma_init_log = np.ones((mesh.Ny, mesh.Nz)) * -2.0
    m_init = sigma_init_log.flatten()

    # HEDEF MİSFİT HESABI (Target Phi)
    # N_data = Frekans * İstasyon Sayısı * 2 (Reel+Sanal)
    # Bizim kodumuzda istasyon sayısı Ny kadar kabul ediliyor
    n_data = len(frequencies) * mesh.Ny * 2
    target_phi = n_data / 2.0

    print(f" [Bilgi] Toplam Veri Noktası: {n_data}")
    print(f" [Bilgi] Hedef Misfit (Target Phi): {target_phi:.1f}")

    # YENİ: Otomatik Beta Seçimi
    # utils.py içindeki fonksiyonu çağırıyoruz
    optimal_beta = find_optimal_beta_fast(
        grad_engine, frequencies, mesh, m_init, target_phi
    )

    # --- 4. OPTİMİZASYON (L-BFGS) ---
    print(f"\n [2/4] Inversion başlatılıyor (Beta={optimal_beta})...")

    bounds = [(-4, 0) for _ in range(len(m_init))]

    def objective_function(m_vec):
        sigma_curr_linear = 10 ** m_vec.reshape((mesh.Ny, mesh.Nz))
        # Otomatik bulunan optimal_beta'yı kullanıyoruz
        grad_m, phi = grad_engine.compute_gradient(
            frequencies, sigma_curr_linear, beta=optimal_beta
        )
        return phi, grad_m.flatten()

    result = minimize(
        fun=objective_function,
        x0=m_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"disp": True, "maxiter": 100},
    )

    print(f" [3/4] İşlem Tamamlandı: {result.message}")

    # --- 5. GÖRSELLEŞTİRME ---
    print(" [4/4] Sonuçlar çizdiriliyor...")
    m_final = result.x.reshape((mesh.Ny, mesh.Nz))

    # constrained_layout=True ile eksen kayması uyarısını önlüyoruz
    plt.figure(figsize=(16, 5), constrained_layout=True)
    vmin, vmax = -3, -1

    # A) Gerçek Model
    ax1 = plt.subplot(1, 3, 1)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_true_log,
        ax=ax1,
        title="Gerçek Model (Ground Truth)",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    # B) Başlangıç Modeli
    ax2 = plt.subplot(1, 3, 2)
    MT2DVisualizer.plot_model(
        mesh,
        sigma_init_log,
        ax=ax2,
        title="Başlangıç Modeli",
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
        title=f"Auto-Inversion (Beta={optimal_beta})",
        grid_lines=True,
        vmin=vmin,
        vmax=vmax,
    )

    plt.colorbar(
        cbar, ax=[ax1, ax2, ax3], label="log10(Conductivity)", fraction=0.02, pad=0.04
    )
    plt.show()


if __name__ == "__main__":
    main()
