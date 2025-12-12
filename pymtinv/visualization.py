import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class MT2DVisualizer:
    """
    MT2DMesh ve Modelleri çizdirmek için yardımcı sınıf.
    SimPEG tarzı görselleştirme sağlar.
    """

    @staticmethod
    def plot_mesh(mesh, ax=None, title="Mesh Structure"):
        """
        Sadece Mesh ızgarasını (Grid) çizer. Model renklendirmesi yapmaz.
        Padding alanları ve hücre sınırları net görünür.
        """
        if ax is None:
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

        # Grid düğüm noktalarını al (Köşeler)
        Y, Z = mesh.get_node_coordinates()

        # Pcolormesh ile boş bir grid çiz (Sadece kenar çizgileri)
        # Dummy data (sıfırlar) veriyoruz, alpha=0 ile içini görünmez yapıyoruz.
        dummy_data = np.zeros((mesh.Ny, mesh.Nz))

        # Mesh çizimi
        # edgecolors='k': Siyah çizgiler
        # facecolor='none': İçi boş
        # linewidth: Çizgi kalınlığı
        ax.pcolormesh(
            Y.T,
            Z.T,
            dummy_data.T,
            edgecolors="k",
            facecolor="none",
            linewidth=0.5,
            shading="flat",
        )

        # Eksen Ayarları
        ax.set_title(title)
        ax.set_xlabel("Y Distance (m)")
        ax.set_ylabel("Z Depth (m)")
        ax.invert_yaxis()  # Derinlik aşağı doğru artsın
        ax.set_aspect("equal")  # Oranları koru (1 metre = 1 metre)

        return ax

    @staticmethod
    def plot_model(
        mesh,
        model,
        ax=None,
        title="Resistivity Model",
        cmap="jet",
        grid_lines=True,
        vmin=None,
        vmax=None,
    ):
        """
        Modeli mesh üzerinde renkli olarak çizer.

        Parametreler:
        - model: (Ny, Nz) boyutunda model matrisi (log10(sigma) veya sigma)
        - grid_lines: True ise hücre sınırlarını çizer (SimPEG tarzı).
        """
        if ax is None:
            fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

        # Grid düğüm noktalarını al (Köşeler)
        # Transpose (.T) alıyoruz çünkü pcolormesh (X, Y) beklerken biz (Y, Z) tutuyoruz.
        Y, Z = mesh.get_node_coordinates()

        # Çizim (Pcolormesh değişken hücre boyutlarını destekler)
        # shading='flat' -> Hücre merkezindeki değeri o hücreye boyar.
        if grid_lines:
            edge_color = "k"
            line_width = 0.3
        else:
            edge_color = "face"  # Çizgi yok
            line_width = 0

        # Modelin boyutunu ve yönünü mesh'e uydur (Transpose gerekebilir)
        plot_data = model.T

        c = ax.pcolormesh(
            Y.T,
            Z.T,
            plot_data,
            cmap=cmap,
            edgecolors=edge_color,
            linewidth=line_width,
            vmin=vmin,
            vmax=vmax,
            shading="flat",
        )

        # Eksen Ayarları
        ax.set_title(title)
        ax.set_xlabel("Y Distance (m)")
        ax.set_ylabel("Z Depth (m)")
        ax.invert_yaxis()  # Z eksenini ters çevir (Derinlik pozitif)
        ax.set_aspect("equal")  # Fiziksel oranları koru

        return c, ax
