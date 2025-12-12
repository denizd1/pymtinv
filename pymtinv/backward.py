import numpy as np
from scipy.sparse.linalg import splu
from .physics import MU_0


class MT2DGradient:
    def __init__(self, forward_solver, data_obs, data_std):
        """
        Adjoint State Method ile Gradyan Hesaplayıcı.
        """
        self.fwd = forward_solver
        self.d_obs = data_obs
        data_std = np.asarray(data_std)
        self.Wd = 1.0 / (data_std + 1e-12)

    def _compute_regularization(self, m_model):
        """
        Model pürüzlülüğünü (Roughness) ve gradyanını hesaplar.
        First-order Tikhonov (Gradient of model).
        """
        Ny, Nz = self.fwd.mesh.Ny, self.fwd.mesh.Nz
        m = m_model  # Log10(sigma) matrisi (Ny, Nz)

        # Yönlü türevler (Yatay ve Dikey komşu farkları)
        # dy: m[i+1, j] - m[i, j]
        diff_y = np.diff(m, axis=0)
        # dz: m[i, j+1] - m[i, j]
        diff_z = np.diff(m, axis=1)

        # 1. Pürüzlülük Değeri (Scalar Phi_m)
        # L2 Normun karesi: Toplam(fark^2)
        phi_m = 0.5 * (np.sum(diff_y**2) + np.sum(diff_z**2))

        # 2. Pürüzlülük Gradyanı (dPhi_m / dm)
        # Her hücre için: 2*m_i - m_{i-1} - m_{i+1} (Laplasyen benzeri)
        grad_reg = np.zeros_like(m)

        # Yatay türevlerin gradyana katkısı
        # diff_y[i] = m[i+1] - m[i]
        # m[i] için katkı: -diff_y[i] (kendisinden sonrakine etkisi) + diff_y[i-1] (kendisinden öncekine etkisi)
        grad_reg[:-1, :] -= diff_y
        grad_reg[1:, :] += diff_y

        # Dikey türevlerin gradyana katkısı
        grad_reg[:, :-1] -= diff_z
        grad_reg[:, 1:] += diff_z

        return phi_m, grad_reg

    def compute_gradient(self, freqs, sigma_model, beta=0.0):
        """
        Amaç fonksiyonu gradyanı.

        Parametreler:
        - beta: Regularization katsayısı (Tikhonov).
                0.0 ise sadece veri hatasına bakar.
                Yüksek değerler (örn. 1.0, 10.0) modeli çok pürüzsüz yapar.
        """
        mesh = self.fwd.mesh
        Ny, Nz = mesh.Ny, mesh.Nz
        dy, dz = mesh.dy, mesh.dz

        # Model m = log10(sigma)
        # Gradyan hesabı 'sigma' üzerinden değil 'm' üzerinden dönecek
        m_model = np.log10(sigma_model)

        total_grad_sigma = np.zeros_like(sigma_model, dtype=np.float64)
        total_phi_d = 0.0

        # --- 1. Veri Kısmı (Data Misfit) ---
        Z_calc, E_fields = self.fwd.solve_te(freqs, sigma_model)

        for freq_idx, freq in enumerate(freqs):
            omega = 2 * np.pi * freq

            # Residual Hesapla
            z_pred = Z_calc[freq]
            z_obs = self.d_obs[freq_idx]
            weight = self.Wd[freq_idx]
            residual = weight * (z_pred - z_obs)

            total_phi_d += 0.5 * np.sum(np.abs(residual) ** 2)

            # Adjoint Kaynak (RHS)
            q = np.zeros((Ny + 1) * (Nz + 1), dtype=np.complex128)

            Ex = E_fields[freq][:, 0]
            coeff_H = -1.0 / (1j * omega * MU_0 * dz[0])
            Hy = coeff_H * (E_fields[freq][:, 1] - E_fields[freq][:, 0])

            weighted_res = np.conj(residual) * weight
            term_dZ_dE = 1.0 / Hy
            term_dZ_dH = -(Ex / (Hy**2))

            idx_surf = np.arange(Ny + 1)
            idx_sub = np.arange(Ny + 1) + (Ny + 1)

            src_E = weighted_res * term_dZ_dE
            q[idx_surf] += src_E
            src_H = weighted_res * term_dZ_dH
            q[idx_sub] += src_H * coeff_H
            q[idx_surf] -= src_H * coeff_H

            # Adjoint Solve
            A, _ = self.fwd.get_system_matrix(freq, sigma_model)
            v = splu(A.T.tocsc()).solve(q)

            # Sensitivity
            u = E_fields[freq].flatten("F")
            u_grid = u.reshape((Ny + 1, Nz + 1), order="F")
            v_grid = v.reshape((Ny + 1, Nz + 1), order="F")

            const_term = 1j * omega * MU_0
            uv = u_grid * v_grid
            uv_center = 0.25 * (uv[:-1, :-1] + uv[1:, :-1] + uv[:-1, 1:] + uv[1:, 1:])

            sensitivity = uv_center * const_term
            total_grad_sigma += np.real(sensitivity)

        # Sigma türevinden m (log10) türevine geçiş
        # dm = dsigma * (ln(10) * sigma)
        grad_m_data = total_grad_sigma * sigma_model * np.log(10)

        # --- 2. Regularization Kısmı (Tikhonov) ---
        if beta > 0.0:
            phi_m, grad_m_reg = self._compute_regularization(m_model)
        else:
            phi_m = 0.0
            grad_m_reg = np.zeros_like(m_model)

        # Toplam Amaç Fonksiyonu ve Gradyan
        total_phi = total_phi_d + beta * phi_m
        total_grad_m = grad_m_data + beta * grad_m_reg

        return total_grad_m, total_phi
