import numpy as np
import matplotlib.pyplot as plt
from pymtinv.mesh import create_padded_mesh
from pymtinv.visualization import MT2DVisualizer


def main():
    # 1. Paddingli Mesh Oluştur
    # Core: 10km genişlik, 5km derinlik. Hücreler: 500m x 250m
    print("Mesh oluşturuluyor...")
    mesh = create_padded_mesh(
        core_width=10000,
        core_depth=5000,
        core_dy=500,
        core_dz=250,
        pad_factor=1.4,
        n_pad_y=5,
        n_pad_z=5,
    )

    # 2. Örnek Bir Model Oluştur
    # Arkaplan: -2 (100 ohm-m)
    sigma_log = np.ones((mesh.Ny, mesh.Nz)) * -2.0

    # Core bölgesine bir blok koyalım (İndeksleri kabaca tahmin ediyoruz)
    # Paddingler yüzünden core indeksleri biraz içeride başlar.
    # Mesh origin'i negatifte başladığı için, 0 civarı core başlangıcıdır.

    # Basit bir "iletken blok" (Kırmızı)
    # Gridin tam ortasına koymaya çalışalım
    center_y, center_z = mesh.Ny // 2, mesh.Nz // 2
    sigma_log[
        center_y - 4 : center_y + 4, center_z - 4 : center_z + 4
    ] = -1.0  # 10 ohm-m

    # 3. GÖRSELLEŞTİRME
    plt.figure(figsize=(14, 6))

    # A) Sadece Mesh Yapısı (Boş)
    ax1 = plt.subplot(1, 2, 1)
    MT2DVisualizer.plot_mesh(mesh, ax=ax1, title="Sadece Mesh (Padding Görünümü)")
    # Paddinglerin nasıl büyüdüğünü görmek için zoom yapabilirsin veya tümünü görebilirsin.

    # B) Mesh + Model
    ax2 = plt.subplot(1, 2, 2)
    cbar_mesh, _ = MT2DVisualizer.plot_model(
        mesh, sigma_log, ax=ax2, title="Mesh + Model", grid_lines=True
    )
    plt.colorbar(cbar_mesh, ax=ax2, label="log10(Conductivity)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
