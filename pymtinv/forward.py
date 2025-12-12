# pymtinv/forward.py
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu
from .physics import MU_0


class MT2DForward:
    def __init__(self, mesh):
        self.mesh = mesh
        # Sparse matris yapısını her seferinde tekrar hesaplamamak için cache eklenebilir
        # ama şimdilik basitlik için her seferinde kuruyoruz.

    def _solve_1d_field(self, omega, sigma_col, dz_array):
        """
        Kenar sınır koşulları için 1D Helmholtz denklemini çözer.
        z=0'da E=1 kabul eder (Dirichlet).
        """
        Nz = len(dz_array)
        N_nodes = Nz + 1

        rows, cols, data = [], [], []
        rhs = np.zeros(N_nodes, dtype=np.complex128)

        # Üst Sınır (z=0, Node 0) -> E = 1
        rows.append(0)
        cols.append(0)
        data.append(1.0)
        rhs[0] = 1.0

        # Alt Sınır (z=max, Node Nz) -> E = 0 (Deep boundary)
        rows.append(Nz)
        cols.append(Nz)
        data.append(1.0)
        rhs[Nz] = 0.0

        # İç Noktalar
        for i in range(1, Nz):
            dz1 = dz_array[i - 1]
            dz2 = dz_array[i]

            # Sigma averajlama
            s_avg = (sigma_col[i - 1] * dz1 + sigma_col[i] * dz2) / (dz1 + dz2)

            # FDM Katsayıları
            alpha = 2 / (dz1 * (dz1 + dz2))
            gamma = 2 / (dz2 * (dz1 + dz2))
            beta = -2 / (dz1 * dz2)
            k_sq = 1j * omega * MU_0 * s_avg

            rows.append(i)
            cols.append(i - 1)
            data.append(alpha)  # Üst
            rows.append(i)
            cols.append(i + 1)
            data.append(gamma)  # Alt
            rows.append(i)
            cols.append(i)
            data.append(beta - k_sq)  # Merkez

        A_1d = sp.coo_matrix((data, (rows, cols)), shape=(N_nodes, N_nodes)).tocsc()
        e_1d = splu(A_1d).solve(rhs)
        return e_1d

    def get_system_matrix(self, freq, sigma_model):
        """
        Verilen frekans ve model için A matrisini ve RHS vektörünü oluşturur.
        A * x = rhs

        Dönüş:
            A: (N_nodes x N_nodes) scipy.sparse.csc_matrix
            rhs: (N_nodes) numpy array (complex128)
        """
        omega = 2 * np.pi * freq
        Ny, Nz = self.mesh.Ny, self.mesh.Nz
        dy, dz = self.mesh.dy, self.mesh.dz

        # Global düğüm indeksi
        def get_idx(iy, iz):
            return iz * (Ny + 1) + iy

        # --- 1. Sınır Değerlerini Hesapla (1D Boundary) ---
        sig_left = sigma_model[0, :]
        E_left = self._solve_1d_field(omega, sig_left, dz)

        sig_right = sigma_model[-1, :]
        E_right = self._solve_1d_field(omega, sig_right, dz)

        # --- 2. Matris Assembly ---
        rows, cols, data = [], [], []
        rhs = np.zeros((Ny + 1) * (Nz + 1), dtype=np.complex128)

        # A. Sınır Koşulları (Dirichlet)

        # Üst Sınır (z=0) -> E=1
        for iy in range(Ny + 1):
            idx = get_idx(iy, 0)
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rhs[idx] = 1.0

        # Alt Sınır (z=Nz) -> E=0
        for iy in range(Ny + 1):
            idx = get_idx(iy, Nz)
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rhs[idx] = 0.0

        # Sol Sınır (y=0) -> 1D Çözüm
        for iz in range(1, Nz):
            idx = get_idx(0, iz)
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rhs[idx] = E_left[iz]

        # Sağ Sınır (y=Ny) -> 1D Çözüm
        for iz in range(1, Nz):
            idx = get_idx(Ny, iz)
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            rhs[idx] = E_right[iz]

        # B. İç Noktalar (Helmholtz)
        for iz in range(1, Nz):
            for iy in range(1, Ny):
                idx = get_idx(iy, iz)

                dy1, dy2 = dy[iy - 1], dy[iy]
                dz1, dz2 = dz[iz - 1], dz[iz]

                # Volume weighted sigma
                # Node (iy, iz) etrafındaki 4 hücre
                s1 = sigma_model[iy - 1, iz - 1]
                s2 = sigma_model[iy, iz - 1]
                s3 = sigma_model[iy - 1, iz]
                s4 = sigma_model[iy, iz]

                area = (dy1 + dy2) * (dz1 + dz2)
                s_node = (
                    s1 * dy1 * dz1 + s2 * dy2 * dz1 + s3 * dy1 * dz2 + s4 * dy2 * dz2
                ) / area

                k_sq = 1j * omega * MU_0 * s_node

                # Coefficients
                alpha_y = 2 / (dy1 * (dy1 + dy2))
                gamma_y = 2 / (dy2 * (dy1 + dy2))
                beta_y = -2 / (dy1 * dy2)

                alpha_z = 2 / (dz1 * (dz1 + dz2))
                gamma_z = 2 / (dz2 * (dz1 + dz2))
                beta_z = -2 / (dz1 * dz2)

                # Stencil
                rows.append(idx)
                cols.append(get_idx(iy - 1, iz))
                data.append(alpha_y)
                rows.append(idx)
                cols.append(get_idx(iy + 1, iz))
                data.append(gamma_y)
                rows.append(idx)
                cols.append(get_idx(iy, iz - 1))
                data.append(alpha_z)
                rows.append(idx)
                cols.append(get_idx(iy, iz + 1))
                data.append(gamma_z)

                diag = beta_y + beta_z - k_sq
                rows.append(idx)
                cols.append(idx)
                data.append(diag)

        # CSC Formatına Çevir (Solver için en iyisi)
        A = sp.coo_matrix(
            (data, (rows, cols)), shape=((Ny + 1) * (Nz + 1), (Ny + 1) * (Nz + 1))
        ).tocsc()

        return A, rhs

    def solve_te(self, freqs, sigma_model):
        """
        TE Modu (Ex) Çözümü.
        Artık get_system_matrix kullanıyor.
        """
        E_fields = {}
        Z_te = {}
        Ny, Nz = self.mesh.Ny, self.mesh.Nz
        dz = self.mesh.dz

        for freq in freqs:
            # 1. Sistemi Kur
            A, rhs = self.get_system_matrix(freq, sigma_model)

            # 2. Sistemi Çöz (Direct Solver - LU Decomposition)
            # Büyük modellerde burası Iterative Solver (örn. GMRES) ile değiştirilebilir.
            lu = splu(A)
            ex_vec = lu.solve(rhs)

            # 3. Sonuçları Sakla
            Ex_grid = ex_vec.reshape((Ny + 1, Nz + 1), order="F")
            E_fields[freq] = Ex_grid

            # 4. Empedans Hesabı (Yüzeyde)
            # Zxy = Ex / Hy
            omega = 2 * np.pi * freq
            Ex0 = Ex_grid[:, 0]
            Ex1 = Ex_grid[:, 1]
            dEx_dz = (Ex1 - Ex0) / dz[0]

            Hy = (-1 / (1j * omega * MU_0)) * dEx_dz
            Z_te[freq] = Ex0 / Hy

        return Z_te, E_fields
