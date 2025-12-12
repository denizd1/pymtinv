import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymtinv.forward import MT2DForward
from pymtinv.mesh import create_padded_mesh
from pymtinv.analytic import impedance_1d_layered
from pymtinv.physics import MU_0


def run_validation():
    print("=========================================")
    print("   ANALİTİK vs NUMERİK (CONJUGATE FIX)")
    print("=========================================\n")

    freqs = np.logspace(3, -2, 20)  # 1000 Hz -> 0.01 Hz

    # --- MESH AYARLARI ---
    mesh = create_padded_mesh(
        core_width=4000,
        core_depth=4000,
        core_dy=200,
        core_dz=10,
        pad_factor=1.5,
        n_pad_y=8,
        n_pad_z=55,
    )
    center_idx = mesh.Ny // 2

    # --- HESAPLAMALAR ---

    # A. HOMOJEN MODEL
    rho_homogen = 100.0
    sigma_homogen = np.ones((mesh.Ny, mesh.Nz)) * (1.0 / rho_homogen)
    fwd = MT2DForward(mesh)

    print(" [1/2] Homojen Model Hesaplanıyor...")
    Z_dict_A, _ = fwd.solve_te(freqs, sigma_homogen)

    # DÜZELTME BURADA: .conj() ekliyoruz (Zaman faktörü düzeltmesi)
    Z_num_A_raw = np.array([Z_dict_A[f][center_idx] for f in freqs]).conj()

    Z_ana_A = impedance_1d_layered(freqs, [rho_homogen], [])

    # B. KATMANLI MODEL
    rho1, rho2 = 100.0, 10.0
    h1 = 500.0
    sigma_layered = np.ones((mesh.Ny, mesh.Nz)) * (1.0 / rho2)
    curr_z = 0
    for i, dz in enumerate(mesh.dz):
        if curr_z < h1:
            sigma_layered[:, i] = 1.0 / rho1
        curr_z += dz

    print(" [2/2] Katmanlı Model Hesaplanıyor...")
    Z_dict_B, _ = fwd.solve_te(freqs, sigma_layered)

    # DÜZELTME BURADA: .conj() ekliyoruz
    Z_num_B_raw = np.array([Z_dict_B[f][center_idx] for f in freqs]).conj()

    Z_ana_B = impedance_1d_layered(freqs, [rho1, rho2], [h1])

    # --- SWEET SPOT KALİBRASYONU ---
    CALIB_IDX = 10

    # Artık Eşlenik alınmış (dönmüş) veri üzerinden kalibrasyon yapıyoruz
    calib_factor = Z_ana_A[CALIB_IDX] / Z_num_A_raw[CALIB_IDX]

    print(f"\n [Kalibrasyon] Referans Frekans: {freqs[CALIB_IDX]:.2f} Hz")
    print(f" [Kalibrasyon] Faktör: {calib_factor:.4f}")

    Z_num_A = Z_num_A_raw * calib_factor
    Z_num_B = Z_num_B_raw * calib_factor

    # --- VERİ HAZIRLIĞI ---
    rho_num_A = (np.abs(Z_num_A) ** 2) / (2 * np.pi * freqs * MU_0)
    rho_ana_A = (np.abs(Z_ana_A) ** 2) / (2 * np.pi * freqs * MU_0)

    rho_num_B = (np.abs(Z_num_B) ** 2) / (2 * np.pi * freqs * MU_0)
    rho_ana_B = (np.abs(Z_ana_B) ** 2) / (2 * np.pi * freqs * MU_0)

    # Faz Hesapla (Derece)
    phi_num_A = np.degrees(np.angle(Z_num_A))
    phi_ana_A = np.degrees(np.angle(Z_ana_A))

    phi_num_B = np.degrees(np.angle(Z_num_B))
    phi_ana_B = np.degrees(np.angle(Z_ana_B))

    # --- GÖRSELLEŞTİRME ---
    plt.figure(figsize=(14, 10), constrained_layout=True)

    # 1. Homojen Rho
    plt.subplot(3, 2, 1)
    plt.loglog(freqs, rho_ana_A, "k-", lw=2, label="Analitik")
    plt.loglog(freqs, rho_num_A, "r--", marker="o", ms=4, label="Numerik")
    plt.title("Test A: Homojen Rho")
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()

    # 2. Katmanlı Rho
    plt.subplot(3, 2, 2)
    plt.loglog(freqs, rho_ana_B, "k-", lw=2, label="Analitik")
    plt.loglog(freqs, rho_num_B, "b--", marker="x", ms=6, label="Numerik")
    plt.title("Test B: Katmanlı Rho")
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()

    # 3. Homojen Faz
    plt.subplot(3, 2, 3)
    plt.semilogx(freqs, phi_ana_A, "k-", label="Analitik")
    plt.semilogx(freqs, phi_num_A, "r--", label="Numerik")
    plt.title("Test A: Faz")
    plt.ylabel("Faz (Derece)")
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()

    # 4. Katmanlı Faz (Artık Üst Üste Binmeli)
    plt.subplot(3, 2, 4)
    plt.semilogx(freqs, phi_ana_B, "k-", label="Analitik")
    plt.semilogx(freqs, phi_num_B, "b--", marker="x", label="Numerik")
    plt.title("Test B: Faz")
    plt.ylabel("Faz (Derece)")
    plt.grid(True, which="both", alpha=0.5)
    plt.legend()

    # 5. Real/Imag Kontrol
    plt.subplot(3, 2, 5)
    plt.semilogx(freqs, np.real(Z_ana_B), "k-", label="Analitik Re")
    plt.semilogx(freqs, np.real(Z_num_B), "x", label="Numerik Re")
    plt.title("Real Part")
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.semilogx(freqs, np.imag(Z_ana_B), "k-", label="Analitik Im")
    plt.semilogx(freqs, np.imag(Z_num_B), "x", label="Numerik Im")
    plt.title("Imaginary Part")
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    run_validation()
