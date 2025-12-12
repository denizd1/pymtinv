import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymtinv.forward import MT2DForward
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer


# --- 1. Model ---
def create_single_block_model(mesh, n_pad_y):
    sigma_log = np.ones((mesh.Ny, mesh.Nz)) * -2.0
    cy = n_pad_y + (mesh.Ny - 2 * n_pad_y) // 2
    cz = mesh.Nz // 2
    sigma_log[cy - 3 : cy + 3, cz - 3 : cz + 3] = -1.0
    return sigma_log


# --- 2. MCMC Motoru (Metropolis-Hastings) ---
def run_metropolis_hastings(
    fwd, Z_obs, data_std, freqs, mesh, m_init, n_iter, step_size, temp
):
    """
    Basit Metropolis-Hastings Algoritması.
    Gradyan kullanmaz, sadece Forward Solver ile çalışır.
    """
    m_curr = m_init.copy()

    # İlk Misfit Hesabı
    sigma_curr = 10 ** m_curr.reshape((mesh.Ny, mesh.Nz))
    Z_pred, _ = fwd.solve_te(freqs, sigma_curr)

    phi_curr = 0
    for i, f in enumerate(freqs):
        diff = Z_obs[i] - Z_pred[f]
        phi_curr += np.sum(np.abs(diff) ** 2 / data_std[i] ** 2) / 2.0

    # Kayıtlar
    phi_history = [phi_curr]
    accepted_count = 0

    print(f"{'Iter':<10} | {'Phi':<15} | {'Accept Rate':<15}")
    print("-" * 45)

    for i in range(n_iter):
        # 1. Öneri (Propose): Rastgele küçük bir değişiklik yap
        perturbation = np.random.randn(*m_curr.shape) * step_size
        m_prop = m_curr + perturbation
        m_prop = np.clip(m_prop, -4.0, 0.0)  # Sınırlar

        # 2. Forward Run (Gradyan yok!)
        sigma_prop = 10 ** m_prop.reshape((mesh.Ny, mesh.Nz))
        Z_prop, _ = fwd.solve_te(freqs, sigma_prop)

        # 3. Misfit Hesapla
        phi_prop = 0
        for k, f in enumerate(freqs):
            diff = Z_obs[k] - Z_prop[f]
            phi_prop += np.sum(np.abs(diff) ** 2 / data_std[k] ** 2) / 2.0

        # 4. Metropolis Kriteri
        delta_phi = phi_prop - phi_curr

        if delta_phi < 0:
            # Daha iyi model -> KESİN KABUL
            accept = True
        else:
            # Daha kötü model -> Olasılıkla Kabul
            prob = np.exp(-delta_phi / temp)
            if np.random.rand() < prob:
                accept = True
            else:
                accept = False

        # 5. Güncelleme
        if accept:
            m_curr = m_prop
            phi_curr = phi_prop
            accepted_count += 1

        phi_history.append(phi_curr)

        if i % 100 == 0:
            rate = accepted_count / (i + 1) * 100
            print(f"{i:<10} | {phi_curr:<15.1f} | {rate:<15.1f}%", end="\r")

    return m_curr, phi_history


def main():
    # Setup (Hızlı olması için küçük mesh ve az frekans)
    N_PAD = 4
    mesh = create_padded_mesh(8000, 4000, 500, 250, 1.3, N_PAD, N_PAD)
    freqs = [10.0, 1.0, 0.1]  # Sadece 3 frekans (MCMC çok yavaştır!)

    print(" [1/3] Veri Hazırlanıyor...")
    sigma_true = create_single_block_model(mesh, N_PAD)
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(freqs, 10**sigma_true)

    Z_obs, data_std = [], []
    for f in freqs:
        val = Z_true[f]
        err = 0.05 * np.abs(val)
        Z_obs.append(
            val + err * (np.random.randn(*val.shape) + 1j * np.random.randn(*val.shape))
        )
        data_std.append(err)

    m_init = (np.ones((mesh.Ny, mesh.Nz)) * -2.0).flatten()

    # MCMC AYARLARI
    # Step Size çok önemli:
    # Büyük olursa hep reddedilir (Accept Rate %0),
    # Küçük olursa yerinde sayar (Accept Rate %99).
    # Hedef: %20-40 arası Accept Rate.
    STEP_SIZE = 0.05
    TEMP = 1.0
    ITER = 200000  # Normalde 100.000 gerekir!

    print(f"\n [2/3] MCMC Başlıyor ({ITER} adım)...")
    start = time.time()
    m_final, hist = run_metropolis_hastings(
        fwd, Z_obs, data_std, freqs, mesh, m_init, ITER, STEP_SIZE, TEMP
    )
    elapsed = time.time() - start

    print(f"\n Tamamlandı: {elapsed:.2f}s | Final Phi: {hist[-1]:.1f}")

    # Görselleştirme
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.title("MCMC Trace (Random Walk)")
    plt.xlabel("Iterasyon")
    plt.ylabel("Misfit")

    plt.subplot(1, 2, 2)
    MT2DVisualizer.plot_model(
        mesh, m_final.reshape((mesh.Ny, mesh.Nz)), plt.gca(), title="MCMC Sonucu"
    )
    plt.show()


if __name__ == "__main__":
    main()
