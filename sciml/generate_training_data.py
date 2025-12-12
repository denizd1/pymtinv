import numpy as np
import matplotlib.pyplot as plt
import os


def generate_synthetic_geology(n_samples, ny, nz):
    """
    RBM Eğitimi için rastgele jeolojik yapılar (Bloklar ve Katmanlar) üretir.
    Çıktı: 0 (Yalıtkan) ve 1 (İletken) arası binary haritalar.
    """
    data = np.zeros((n_samples, ny * nz))

    for i in range(n_samples):
        # Arkaplan (0)
        img = np.zeros((ny, nz))

        # Rastgele 1-3 adet blok veya katman ekle
        n_shapes = np.random.randint(1, 4)
        for _ in range(n_shapes):
            # Rastgele tip: 0=Blok, 1=Yatay Katman
            shape_type = np.random.randint(0, 2)

            if shape_type == 0:  # Kare Blok
                w = np.random.randint(3, 8)
                h = np.random.randint(3, 8)
                y = np.random.randint(0, ny - w)
                z = np.random.randint(0, nz - h)
                img[y : y + w, z : z + h] = 1.0

            elif shape_type == 1:  # Yatay Katman
                z_start = np.random.randint(0, nz - 2)
                thickness = np.random.randint(1, 4)
                img[:, z_start : z_start + thickness] = 1.0

        data[i] = img.flatten()

    return data


if __name__ == "__main__":
    # Mesh boyutlarımız (Complex inversion örneğindeki gibi 30x25 varsayalım)
    NY, NZ = 30, 25
    N_SAMPLES = 50000

    print(f"{N_SAMPLES} adet sentetik jeoloji üretiliyor...")
    dataset = generate_synthetic_geology(N_SAMPLES, NY, NZ)

    # Görsel Kontrol
    plt.figure(figsize=(10, 2))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(dataset[i].reshape(NY, NZ), cmap="gray")
        plt.axis("off")
    plt.suptitle("Eğitim Verisi Örnekleri")
    plt.show()

    # Kaydet
    np.save("geology_dataset.npy", dataset)
    print("Veri 'geology_dataset.npy' olarak kaydedildi.")
