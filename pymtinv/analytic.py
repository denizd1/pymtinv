# pymtinv/analytic.py
import numpy as np
from .physics import MU_0


def impedance_1d_layered(freqs, resistivities, thicknesses):
    """
    1D Katmanlı ortam için Cagniard Empedansını hesaplar (Recursive).
    resistivities: [rho1, rho2, ..., rho_last] (Ohm-m)
    thicknesses: [h1, h2, ..., h_(last-1)] (m)
    """
    omega = 2 * np.pi * freqs
    n_layer = len(resistivities)
    Z = np.zeros(len(freqs), dtype=np.complex128)

    for i, w in enumerate(omega):
        # En alttaki katman (Yarım uzay) empedansı
        k_last = np.sqrt(-1j * w * MU_0 / resistivities[-1])
        Z_val = (1j * w * MU_0) / k_last

        # Yukarı doğru rekürsif yayılım
        for j in range(n_layer - 2, -1, -1):
            rho = resistivities[j]
            h = thicknesses[j]
            k = np.sqrt(-1j * w * MU_0 / rho)
            gamma = k  # Propagation constant

            # Waite (1982) formülasyonu
            tanh_gh = np.tanh(gamma * h)
            intrinsic_Z = (1j * w * MU_0) / k

            # Empedans taşıma
            Z_val = (
                intrinsic_Z
                * (Z_val + intrinsic_Z * tanh_gh)
                / (intrinsic_Z + Z_val * tanh_gh)
            )

        Z[i] = Z_val

    return Z


def halfspace_fields(freq, rho, z_array):
    """
    Homojen yarım uzayda E alanı derinlikle nasıl azalır (Skin Effect).
    Analitik referans için.
    E(z) = E0 * exp(-k * z)
    """
    omega = 2 * np.pi * freq
    k = np.sqrt(-1j * omega * MU_0 / rho)  # Dalga sayısı
    # Yüzeyde E = 1 (Normalize) kabul edelim
    E_z = np.exp(-k * z_array)
    return E_z
