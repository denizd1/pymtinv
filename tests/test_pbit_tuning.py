import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Kütüphane yolu
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymtinv.forward import MT2DForward, MT2DMesh
from pymtinv.backward import MT2DGradient


# --- A. YARDIMCI FONKSİYONLAR ---
def create_synthetic_model(mesh):
    sigma = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    sigma[:, 0:2] = 0.1
    sigma[4:8, 8:12] = 0.1
    sigma[12:16, 6:10] = 0.001
    return sigma


def pbit_optimizer(grad_engine, m_init, freqs, mesh, n_steps, lr, t_start, t_end):
    """
    Parametrik P-bit optimizer. Dışarıdan gelen hyperparametreleri kullanır.
    """
    m_current = m_init.copy()
    energy_history = []

    # En iyi sonucu saklamak için
    best_m = m_current.copy()
    best_phi = 1e9

    for step in range(n_steps):
        # Cooling Schedule
        decay_factor = (t_end / t_start) ** (step / n_steps)
        current_temp = t_start * decay_factor

        sigma_curr = 10 ** m_current.reshape((mesh.Ny, mesh.Nz))

        # Gradient
        grad_det, phi = grad_engine.compute_gradient(freqs, sigma_curr, beta=5.0)
        grad_det = grad_det.flatten()

        # Thermal Noise
        noise_scale = np.sqrt(2 * current_temp)
        thermal_noise = np.random.randn(*m_current.shape) * noise_scale

        # Langevin Update
        update = -lr * grad_det + thermal_noise
        m_current += update
        m_current = np.clip(m_current, -4.0, 0.0)

        energy_history.append(phi)

        # En iyi modeli sakla (P-bitler gezinirken en iyi yeri geçebilir, o yüzden kaydetmeliyiz)
        if phi < best_phi:
            best_phi = phi
            best_m = m_current.copy()

    return best_m, energy_history, best_phi


# --- B. GRID SEARCH (IZGARA ARAMASI) ---
def run_tuning():
    # 1. Setup
    mesh = MT2DMesh(dy=np.ones(25) * 500, dz=np.ones(20) * 500)
    frequencies = [100.0, 30.0, 10.0, 1.0, 0.1]

    print(" [Setup] Veri ve Model Hazırlanıyor...")
    sigma_true = create_synthetic_model(mesh)
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(frequencies, sigma_true)

    Z_obs, data_std = [], []
    np.random.seed(42)
    for f in frequencies:
        z_val = Z_true[f]
        noise_std = 0.05 * np.abs(z_val)
        noise = noise_std * (
            np.random.randn(*z_val.shape) + 1j * np.random.randn(*z_val.shape)
        )
        Z_obs.append(z_val + noise)
        data_std.append(noise_std)

    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    sigma_init = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    m_init = np.log10(sigma_init).flatten()

    # --- TUNING PARAMETRELERİ ---
    # Hangi kombinasyonları deneyelim?
    # Learning Rate: Çok yavaştan (0.001) çok hızlıya (0.02)
    learning_rates = [0.001, 0.005, 0.02]

    # Temp Start: Soğuktan (0.01) çok sıcağa (1.0)
    temp_starts = [0.01, 0.1, 1.0]

    # Sabit Parametreler
    N_STEPS = 500  # Tuning için çok uzun tutmayalım
    TEMP_END = 0.0001

    results = []

    print(f"\n {'LR':<10} | {'T_start':<10} | {'Final Phi':<15} | {'Durum'}")
    print("-" * 55)

    # Grid Search Loop
    for lr in learning_rates:
        for t_start in temp_starts:
            start_t = time.time()

            # Run Simulation
            _, hist, final_phi = pbit_optimizer(
                grad_engine,
                m_init,
                frequencies,
                mesh,
                n_steps=N_STEPS,
                lr=lr,
                t_start=t_start,
                t_end=TEMP_END,
            )

            elapsed = time.time() - start_t

            # Sonucu Kaydet
            results.append(
                {"lr": lr, "t_start": t_start, "phi": final_phi, "hist": hist}
            )

            print(f" {lr:<10} | {t_start:<10} | {final_phi:<15.2f} | {elapsed:.2f}s")

    # --- EN İYİ SONUCU BUL ---
    # Phi değerine göre sırala (Küçükten büyüğe)
    results.sort(key=lambda x: x["phi"])
    best = results[0]

    print("\n" + "=" * 40)
    print(f" EN İYİ PARAMETRELER:")
    print(f" Learning Rate : {best['lr']}")
    print(f" Start Temp    : {best['t_start']}")
    print(f" Minimum Phi   : {best['phi']:.2f}")
    print("=" * 40)

    # --- KARŞILAŞTIRMALI GRAFİK ---
    plt.figure(figsize=(10, 6))

    # En iyi 3 sonucu çizdir
    for i in range(min(3, len(results))):
        res = results[i]
        label = f"LR={res['lr']}, T0={res['t_start']} (Phi={res['phi']:.0f})"
        plt.plot(res["hist"], label=label, linewidth=2)

    # En kötü sonucu da çizdir (Farkı görmek için)
    worst = results[-1]
    plt.plot(
        worst["hist"],
        "--",
        color="gray",
        alpha=0.5,
        label=f"Kötü: LR={worst['lr']}, T0={worst['t_start']}",
    )

    plt.title("Hyperparameter Tuning: Enerji Düşüş Karşılaştırması")
    plt.xlabel("Adım")
    plt.ylabel("Enerji (Misfit)")
    plt.legend()
    plt.yscale("log")  # Logaritmik skala farkları daha iyi gösterir
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()


if __name__ == "__main__":
    run_tuning()
