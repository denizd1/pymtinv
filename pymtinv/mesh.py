import numpy as np


class MT2DMesh:
    """
    2D Manyetotellürik Mesh (Grid) Yapısı.
    """

    def __init__(self, dy, dz, origin=(0, 0)):
        self.dy = np.asarray(dy, dtype=float)
        self.dz = np.asarray(dz, dtype=float)
        self.Ny = len(self.dy)
        self.Nz = len(self.dz)
        self.origin = origin

        # Düğüm Koordinatları (Node Coordinates) - Plotting ve Solver için
        # y: 0, dy1, dy1+dy2 ...
        self.y_nodes = np.concatenate(([0], np.cumsum(self.dy))) + origin[0]
        self.z_nodes = np.concatenate(([0], np.cumsum(self.dz))) + origin[1]

        # Hücre Merkezleri (Cell Centers) - Inversion modeli için
        self.cc_y = (self.y_nodes[:-1] + self.y_nodes[1:]) / 2
        self.cc_z = (self.z_nodes[:-1] + self.z_nodes[1:]) / 2

    def get_node_coordinates(self):
        """Meshgrid olarak düğüm koordinatlarını döner"""
        return np.meshgrid(self.y_nodes, self.z_nodes, indexing="ij")


def create_padded_mesh(
    core_width, core_depth, core_dy, core_dz, pad_factor=1.3, n_pad_y=10, n_pad_z=10
):
    """
    Otomatik padding (kenar genişletme) ile mesh oluşturur.
    """
    # 1. Core Mesh
    n_core_y = int(np.ceil(core_width / core_dy))
    n_core_z = int(np.ceil(core_depth / core_dz))

    dy_core = np.ones(n_core_y) * core_dy
    dz_core = np.ones(n_core_z) * core_dz

    # 2. Padding (Y - Sol/Sağ)
    pad_y = core_dy * (pad_factor ** np.arange(1, n_pad_y + 1))
    dy_final = np.concatenate([pad_y[::-1], dy_core, pad_y])

    # 3. Padding (Z - Alt)
    pad_z = core_dz * (pad_factor ** np.arange(1, n_pad_z + 1))
    dz_final = np.concatenate([dz_core, pad_z])

    # Origin ayarı (Core bölgesi y=0'da başlasın diye sola kaydırıyoruz)
    origin_y = -np.sum(pad_y)

    mesh = MT2DMesh(dy_final, dz_final, origin=(origin_y, 0))

    print(f" Mesh Oluşturuldu: {mesh.Ny}x{mesh.Nz} hücre.")
    print(
        f" Toplam Boyut: {np.sum(dy_final) / 1000:.1f} km x {np.sum(dz_final) / 1000:.1f} km"
    )

    return mesh
