import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymtinv.forward import MT2DForward, MT2DMesh
from pymtinv.backward import MT2DGradient


# --- Model Oluşturucu (Aynı) ---
def create_complex_model(mesh):
    sigma = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    sigma[:, 0:2] = 0.1
    sigma[4:8, 8:12] = 0.1
    sigma[12:16, 6:10] = 0.001
    return sigma


def run_l_curve_analysis():
    # 1. Setup
    mesh = MT2DMesh(dy=np.ones(25) * 500, dz=np.ones(20) * 500)
    frequencies = [100.0, 30.0, 10.0, 1.0, 0.1]

    # Gerçek Model ve Veri
    sigma_true = create_complex_model(mesh)
    fwd = MT2DForward(mesh)
    print("Sentetik veri üretiliyor...")
    Z_true, _ = fwd.solve_te(frequencies, sigma_true)

    Z_obs, data_std = [], []
    np.random.seed(42)
    # Gürültü ekleyelim ki Regularization'ın etkisi netleşsin
    for f in frequencies:
        z_val = Z_true[f]
        noise_std = 0.05 * np.abs(z_val)  # %5 Gürültü
        noise = noise_std * (
            np.random.randn(*z_val.shape) + 1j * np.random.randn(*z_val.shape)
        )
        Z_obs.append(z_val + noise)
        data_std.append(noise_std)

    # Başlangıç Modeli
    sigma_init = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    m_init = np.log10(sigma_init).flatten()
    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    bounds = [(-4, 0) for _ in range(len(m_init))]

    # --- L-CURVE PARAMETRE TARAMASI ---
    # Logaritmik olarak sıralanmış beta değerleri
    # Çok küçük (sadece veri) -> Çok büyük (dümdüz model)
    betas = [1e-2, 1e-1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0]

    l_curve_phi_d = []  # Misfit (Y ekseni)
    l_curve_phi_m = []  # Roughness (X ekseni)
    models = []  # Modelleri saklayalım

    print(
        f"\n{'Beta':<10} | {'Phi_d (Misfit)':<15} | {'Phi_m (Rough)':<15} | {'Durum'}"
    )
    print("-" * 55)

    for beta in betas:
        # Her beta için inversion yap
        def objective_function(m_vec):
            sigma_curr = 10 ** m_vec.reshape((mesh.Ny, mesh.Nz))
            grad_m, phi_total = grad_engine.compute_gradient(
                frequencies, sigma_curr, beta=beta
            )
            return phi_total, grad_m.flatten()

        # Hız için iterasyonu düşük tutuyoruz (Gerçek analizde 50-100 olmalı)
        res = minimize(
            objective_function,
            m_init,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9, "gtol": 1e-5},
        )

        # Sonuçları Ayrıştır (Phi_d ve Phi_m'i ayrı ayrı hesaplamamız lazım)
        m_final = res.x.reshape((mesh.Ny, mesh.Nz))

        # 1. Phi_d'yi manuel hesapla (Regularization olmadan)
        grad_dummy, phi_d_only = grad_engine.compute_gradient(
            frequencies, 10**m_final, beta=0.0
        )

        # 2. Phi_m'i manuel hesapla
        # (grad_engine._compute_regularization private metoduna erişiyoruz veya backward.py'den çekeceğiz)
        # backward.py içinde _compute_regularization metodunu kullanıyoruz:
        phi_m_val, _ = grad_engine._compute_regularization(m_final)

        l_curve_phi_d.append(phi_d_only)
        l_curve_phi_m.append(phi_m_val)
        models.append(m_final)

        print(
            f"{beta:<10.1e} | {phi_d_only:<15.2f} | {phi_m_val:<15.2f} | {res.message}"
        )

    # --- Görselleştirme (L-Curve) ---
    plt.figure(figsize=(14, 6))

    # 1. L-Curve Plot
    plt.subplot(1, 2, 1)
    plt.loglog(l_curve_phi_m, l_curve_phi_d, "b-o", linewidth=2)
    plt.xlabel(r"Model Roughness ($\Phi_m$)")
    plt.ylabel(r"Data Misfit ($\Phi_d$)")
    plt.title("L-Curve Analysis")
    plt.grid(True, which="both", ls="-", alpha=0.5)

    # Noktaların üzerine beta değerlerini yaz
    for i, beta in enumerate(betas):
        plt.text(l_curve_phi_m[i], l_curve_phi_d[i], f" $\\beta={beta}$", fontsize=9)

    # Optimuma yakın bir model seçelim (Genellikle köşedeki)
    # Basit bir seçim mantığı: Eğrinin köşesi genellikle ortalardadır.
    # Görsel olarak 4. veya 5. elemanı seçiyoruz (Beta=5.0 veya 10.0 civarı)
    idx_opt = 4
    opt_beta = betas[idx_opt]

    # Seçilen noktayı işaretle
    plt.plot(
        l_curve_phi_m[idx_opt],
        l_curve_phi_d[idx_opt],
        "r*",
        markersize=15,
        label=f"Selected Beta={opt_beta}",
    )
    plt.legend()

    # 2. Seçilen Modelin Görüntüsü
    plt.subplot(1, 2, 2)
    plt.title(f"Optimal Model (Beta={opt_beta})")
    plt.imshow(models[idx_opt].T, cmap="jet", aspect="auto", vmin=-3, vmax=-1)
    plt.colorbar(label="log10(sigma)")

    plt.tight_layout()
    plt.show()
    # 3. Gerçek vs Optimal Karşılaştırması (Yeni Pencere)
    plt.figure(figsize=(12, 5))

    # Renk skalasını sabitleyelim (-3 ile -1 arası)
    # -1: İletken (Kırmızı)
    # -2: Arkaplan (Yeşil/Sarı)
    # -3: Dirençli (Mavi)
    vmin, vmax = -3, -1

    plt.subplot(1, 2, 1)
    plt.title("Gerçek Model (Ground Truth)")
    plt.imshow(np.log10(sigma_true).T, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="log10(sigma)")

    plt.subplot(1, 2, 2)
    plt.title(f"Tikhonov Sonucu (Beta={opt_beta})")
    plt.imshow(models[idx_opt].T, cmap="jet", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(label="log10(sigma)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_l_curve_analysis()
