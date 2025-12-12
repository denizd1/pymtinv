import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pymtinv.forward import MT2DForward
from pymtinv.mesh import MT2DMesh
from pymtinv.backward import MT2DGradient
import time


# --- Modeller ve Yardımcı Fonksiyonlar (Önceki dosyadan kopyalanabilir) ---
# (Burada tekrar tanımlıyorum ki tek başına çalışsın)
def create_complex_model(mesh):
    sigma = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    sigma[:, 0:2] = 0.1  # Overburden
    sigma[4:8, 8:12] = 0.1  # Conductive Block
    sigma[12:16, 6:10] = 0.001  # Resistive Block
    return sigma


def run_tikhonov_test():
    mesh = MT2DMesh(dy=np.ones(25) * 500, dz=np.ones(20) * 500)
    frequencies = [100.0, 30.0, 10.0, 1.0, 0.1]

    # 1. Veri Üret
    sigma_true = create_complex_model(mesh)
    fwd = MT2DForward(mesh)
    print("Veri üretiliyor...")
    Z_true, _ = fwd.solve_te(frequencies, sigma_true)

    Z_obs, data_std = [], []
    np.random.seed(42)
    for f in frequencies:
        z_val = Z_true[f]
        # Biraz yüksek gürültü ekleyelim ki Tikhonov'un faydası belli olsun (%5)
        noise_std = 0.05 * np.abs(z_val)
        noise = noise_std * (
            np.random.randn(*z_val.shape) + 1j * np.random.randn(*z_val.shape)
        )
        Z_obs.append(z_val + noise)
        data_std.append(noise_std)

    # 2. Inversion Ayarları
    sigma_init = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    m_init = np.log10(sigma_init).flatten()
    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    bounds = [(-4, 0) for _ in range(len(m_init))]

    # --- TIKHONOV PARAMETRESİ (BETA) ---
    # Bu değeri değiştirerek denemeler yapabilirsin.
    # 0.0 = Düzgünleştirme yok (Gürültülü çıkar)
    # 5.0 = Dengeli
    # 50.0 = Çok pürüzsüz (Detaylar kaybolabilir)
    BETA = 1.0

    def objective_function(m_vec):
        sigma_curr = 10 ** m_vec.reshape((mesh.Ny, mesh.Nz))
        # Beta parametresini buraya gönderiyoruz
        grad_m, phi = grad_engine.compute_gradient(frequencies, sigma_curr, beta=BETA)
        return phi, grad_m.flatten()

    print(f"Inversion Basliyor (Beta={BETA})...")
    start_time = time.time()

    result = minimize(
        fun=objective_function,
        x0=m_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"disp": True, "maxiter": 1000},
    )

    print(f"Bitti! Süre: {time.time() - start_time:.2f}s")

    # Çizim
    m_final = result.x.reshape((mesh.Ny, mesh.Nz))

    plt.figure(figsize=(10, 5))
    plt.suptitle(f"Tikhonov Regularization (Beta={BETA})")

    plt.subplot(1, 2, 1)
    plt.title("Gerçek Model")
    plt.imshow(np.log10(sigma_true).T, cmap="jet", aspect="auto", vmin=-3, vmax=-1)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.title("Inversion Sonucu")
    plt.imshow(m_final.T, cmap="jet", aspect="auto", vmin=-3, vmax=-1)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_tikhonov_test()
