import numpy as np
import matplotlib.pyplot as plt


from pymtinv.forward import MT2DForward
from pymtinv.backward import MT2DGradient
from pymtinv.visualization import MT2DVisualizer
from pymtinv.utils import (
    pbit_optimizer_with_rbm,
    tune_sciml_hyperparameters,
)
from pymtinv.mesh import MT2DMesh as TensorMesh


def create_complex_model_30x25(mesh):
    # RBM 30x25 eğitildiği için boyutlar sabit olmalı
    sigma = np.ones((mesh.Ny, mesh.Nz)) * 0.01
    # İletken Blok
    sigma[10:15, 8:15] = 1.0
    # Yatay Katman
    sigma[:, 18:21] = 0.5
    return sigma


def main():
    print("=========================================")
    print("   SciML INVERSION: P-BITS + RBM (AI)")
    print("=========================================\n")

    # 1. RBM ile Uyumlu Mesh (30x25 Core)
    # Not: Eğitimde padding yoktu, o yüzden inversion'da sadece core bölgesini
    # RBM'e besleyeceğiz. Ancak basitlik için direkt 30x25 mesh kuralım.
    # (Padding RBM'i şaşırtabilir, bu POC için paddingsiz çalışalım veya az padding verelim)

    # Basit Mesh (Paddingsiz, RBM ile birebir uyumlu)
    # Gerçek uygulamada Core bölgesi kesilip verilir.

    dy = np.ones(30) * 500
    dz = np.ones(25) * 250
    mesh = TensorMesh(dy, dz)

    freqs = np.logspace(2, -1, 5)  # 5 Frekans

    # 2. Gerçek Model ve Veri
    sigma_true = create_complex_model_30x25(mesh)
    fwd = MT2DForward(mesh)
    Z_true, _ = fwd.solve_te(freqs, sigma_true)

    Z_obs = []
    data_std = []
    np.random.seed(42)
    for f in freqs:
        val = Z_true[f]
        err = 0.05 * np.abs(val)
        Z_obs.append(
            val + err * (np.random.randn(*val.shape) + 1j * np.random.randn(*val.shape))
        )
        data_std.append(err)

    grad_engine = MT2DGradient(fwd, Z_obs, data_std)
    m_init = np.log10(np.ones((mesh.Ny, mesh.Nz)) * 0.01).flatten()

    # 3. SciML Inversion (RBM Destekli)
    # Alpha: RBM'in sözünün ne kadar geçeceği.
    # Alpha=5.0 -> RBM çok baskın (Jeolojiye çok benzesin)
    # Alpha=0.1 -> Veri baskın
    RBM_FILE = "rbm_geology_prior.pkl"

    print("\n [1/2] Hiperparametreler Otomatik Ayarlanıyor...")
    # Not: Tuning sırasında rbm_path string olarak veriliyor
    BEST_ALPHA, BEST_LR = tune_sciml_hyperparameters(
        grad_engine, freqs, mesh, m_init, RBM_FILE
    )

    # --- ANA OPTİMİZASYON ---
    print(f"\n [2/2] Ana Inversion Başlıyor (Alpha={BEST_ALPHA}, LR={BEST_LR})...")

    m_final, hist, phi = pbit_optimizer_with_rbm(
        grad_engine,
        m_init,
        freqs,
        mesh,
        n_steps=5000,  # Ana işlem uzun sürsün
        lr=BEST_LR,  # Otomatik bulundu
        t_start=0.1,
        t_end=0.0001,
        rbm_path=RBM_FILE,
        alpha=BEST_ALPHA,  # Otomatik bulundu
        average_last_steps=1000,  # Gürültü temizleme
    )

    m_final_2d = m_final.reshape(mesh.Ny, mesh.Nz)

    # 4. Görselleştirme
    plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 3, 1)
    MT2DVisualizer.plot_model(mesh, np.log10(sigma_true), ax=ax1, title="Gerçek (True)")

    ax2 = plt.subplot(1, 3, 2)
    MT2DVisualizer.plot_model(
        mesh, m_final_2d, ax=ax2, title=f"SciML Result (Phi={phi:.1f})"
    )

    ax3 = plt.subplot(1, 3, 3)
    plt.plot(hist)
    plt.title("Convergence (Physics + AI)")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
