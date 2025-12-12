# tests/test_gradient.py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Path ayarı
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from pymtinv import create_padded_mesh, MT2DForward
from pymtinv.backward import MT2DGradient


def run_gradient_check():
    print("--- Gradient Doğrulama Testi (Taylor Test) ---")

    # 1. Setup (Küçük bir model)
    mesh = create_padded_mesh(
        core_width=1000,
        core_depth=1000,
        core_dy=100,
        core_dz=50,
        pad_factor=1.5,
        n_pad_y=5,
        n_pad_z=5,
    )

    # Başlangıç Modeli (Uniform)
    m0 = np.log10(np.ones((mesh.Ny, mesh.Nz)) * 0.01)  # 100 ohm-m
    sigma_0 = 10**m0

    # Solver'lar
    fwd = MT2DForward(mesh)
    freqs = [10.0, 1.0]  # İki frekans

    # 2. Sentetik Veri Üret (Gözlem olarak kullanacağız)
    # Hedef modelde bir anomali olsun
    m_true = m0.copy()
    # Ortaya iletken bir blok koyalım
    c_y, c_z = mesh.Ny // 2, mesh.Nz // 2
    m_true[c_y - 2 : c_y + 2, c_z - 2 : c_z + 2] = np.log10(0.1)  # 10 ohm-m

    Z_obs_dict, _ = fwd.solve_te(freqs, 10**m_true)

    # Veri formatı (List of arrays)
    Z_obs = [Z_obs_dict[f] for f in freqs]
    # Hata ağırlığı (Noise level) %5 varsayalım
    Z_err = [np.abs(z) * 0.05 for z in Z_obs]

    # 3. Gradyan Hesaplayıcıyı Başlat
    grad_engine = MT2DGradient(fwd, Z_obs, Z_err)

    # 4. Referans Noktasında (m0) Gradyan Hesapla
    print("Adjoint Gradient hesaplanıyor...")
    grad_adj, phi0 = grad_engine.compute_gradient(freqs, sigma_0)

    # 5. Pertürbasyon Yönü (Rastgele bir h vektörü)
    h = np.random.randn(*m0.shape)
    # h vektörünü normalize et
    h = h / np.linalg.norm(h)

    # 6. Taylor Testi Döngüsü
    # Epsilon küçüldükçe, Finite Difference ve Adjoint arasındaki hata
    # epsilon^2 ile azalmalıdır (Second order convergence).

    epsilons = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    errors = []

    print(
        f"{'Epsilon':<10} | {'Phi(m)':<12} | {'Phi(m+eps*h)':<12} | {'Finite Diff':<12} | {'Adjoint Dot':<12} | {'Ratio':<10}"
    )
    print("-" * 80)

    # Adjoint tahmini (Gradient dot h) sabittir
    adj_dot = np.sum(grad_adj * h)

    for eps in epsilons:
        # Pertürbe edilmiş model
        m_pert = m0 + eps * h
        sigma_pert = 10**m_pert

        # Yeni Phi hesapla (Sadece forward run)
        # Gradient motorunun compute_gradient fonksiyonu phi'yi de döner,
        # ama biz sadece phi istiyoruz, manuel hesaplayalım veya fonksiyonu kullanalım.
        _, phi_pert = grad_engine.compute_gradient(freqs, sigma_pert)

        # Finite Difference Türevi
        fd_diff = (phi_pert - phi0) / eps

        # Hata Oranı: |FD - Adjoint| / |FD|
        # İdeal durumda eps->0 iken bu oran 0'a gitmeli,
        # veya oran 1.0'a yaklaşmalı (fd / adj).
        ratio = fd_diff / adj_dot

        print(
            f"{eps:<10.1e} | {phi0:<12.5e} | {phi_pert:<12.5e} | {fd_diff:<12.5e} | {adj_dot:<12.5e} | {ratio:<10.5f}"
        )
        errors.append(abs(ratio - 1.0))

    # Görselleştirme
    plt.figure()
    plt.loglog(epsilons, errors, "o-", label="Relative Error")
    plt.loglog(epsilons, epsilons, "k--", label="First Order (Linear)")
    plt.xlabel("Epsilon")
    plt.ylabel("Error |(FD - Adj)/Adj|")
    plt.title("Gradient Verification (Taylor Test)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    run_gradient_check()
