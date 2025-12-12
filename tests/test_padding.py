import numpy as np
import matplotlib.pyplot as plt
from pymtinv.mesh import create_padded_mesh
from pymtinv.forward import MT2DForward
from pymtinv.analytic import halfspace_fields


def run_padding_test():
    # Parametreler
    rho_half = 100.0
    freq = 1.0

    # Mesh Tasarımı (Padding ile)
    # Core: 2km genişlik, 2km derinlik (İlgi alanı)
    # Padding: Dünyayı 50km+ uzağa taşıyacak
    mesh = create_padded_mesh(
        core_width=2000,
        core_depth=2000,
        core_dy=50.0,  # 50m y-çözünürlük
        core_dz=20.0,  # 20m z-çözünürlük (Yüzeye yakın sık)
        pad_factor=1.4,  # Hızlı büyüme
        n_pad_y=12,
        n_pad_z=15,
    )

    # Model (Tüm uzay 100 ohm-m)
    sigma = np.ones((mesh.Ny, mesh.Nz)) * (1.0 / rho_half)

    # Çözüm
    solver = MT2DForward(mesh)
    Z_calc, E_fields = solver.solve_te([freq], sigma)

    # Sonuçlar
    Ex_num = E_fields[freq]

    # --- Görselleştirme ---
    Y, Z = mesh.get_node_coordinates()

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Elektrik Alan Dağılımı (2D Kesit)
    # Log scale görüntülemek için abs alıyoruz
    c = ax[0].pcolormesh(
        Y / 1000, Z / 1000, np.log10(np.abs(Ex_num)), shading="auto", cmap="jet"
    )
    ax[0].set_ylim(10, 0)  # İlk 10km'ye odaklan
    ax[0].set_xlim(-5, 5)  # Yatayda merkeze odaklan
    ax[0].set_title(f"Log10(|Ex|) Field Distribution (Freq={freq} Hz)")
    ax[0].set_xlabel("Distance (km)")
    ax[0].set_ylabel("Depth (km)")
    plt.colorbar(c, ax=ax[0], label="Log10(V/m)")

    # 2. Derinlik Profili Karşılaştırması (Merkezde)
    center_idx = mesh.Ny // 2
    z_nodes = mesh.node_z

    ex_center = np.abs(Ex_num[center_idx, :])
    ex_analytic = np.abs(halfspace_fields(freq, rho_half, z_nodes))

    ax[1].semilogy(z_nodes, ex_analytic, "k-", lw=2, label="Analytic")
    ax[1].semilogy(z_nodes, ex_center, "r--", lw=2, label="Numeric (Padded)")
    ax[1].set_title("Ex Field Decay vs Depth")
    ax[1].set_xlabel("Depth (m)")
    ax[1].set_ylabel("|Ex| (V/m)")
    ax[1].legend()
    ax[1].grid(True, which="both")

    # Hata Hesabı (Sadece ilk 5 skin depth içinde)
    skin_depth = 500 * np.sqrt(rho_half / freq)
    mask = z_nodes < 5 * skin_depth
    err = np.linalg.norm(ex_center[mask] - ex_analytic[mask]) / np.linalg.norm(
        ex_analytic[mask]
    )

    print(f"Skin Depth: {skin_depth:.1f} m")
    print(f"Weighted Error (Top 5 SD): {err * 100:.4f}%")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_padding_test()
