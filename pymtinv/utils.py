import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import expit  # Sigmoid fonksiyonu (Logistic)
import joblib


def find_optimal_beta_fast(grad_engine, freqs, mesh, m_init, target_phi):
    """
    Hızlı bir ön tarama (Fast Scan) ile optimum Tikhonov (Beta) katsayısını tahmin eder.
    Tam bir L-Curve analizi yapmak yerine, aday değerlerle kısa süreli (5-10 adım)
    inversion yaparak hedefe en çok yaklaşanı seçer.

    Parametreler:
    - grad_engine: MT2DGradient instance'ı.
    - freqs: Frekans listesi.
    - mesh: MT2DMesh instance'ı.
    - m_init: Başlangıç modeli (vektör).
    - target_phi: Hedeflenen veri uyumsuzluğu (Genellikle N_data / 2).

    Dönüş:
    - best_beta: Seçilen en iyi beta değeri.
    """
    print(f"\n--- Otomatik Beta Taraması (Hedef Phi ≈ {target_phi:.0f}) ---")

    # Adaylar (Büyükten küçüğe)
    candidates = [100.0, 50.0, 10.0, 5.0, 1.0, 0.5, 0.1]

    global_best_beta = candidates[0]  # Varsayılan
    global_min_dist = 1e20

    print(f"{'Beta':<10} | {'Phi_d (10 iter)':<15} | {'Fark':<10} | {'Durum'}")
    print("-" * 60)

    for beta in candidates:

        def objective_func(m_vec):
            sigma_curr = 10 ** m_vec.reshape((mesh.Ny, mesh.Nz))
            grad_m, phi_total = grad_engine.compute_gradient(
                freqs, sigma_curr, beta=beta
            )
            return phi_total, grad_m.flatten()

        # İterasyon sayısını 10 yaptık, algoritma yönünü bulsun diye
        res = minimize(
            objective_func,
            m_init,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-4, 0)] * len(m_init),
            options={"maxiter": 10, "disp": False},
        )

        # Saf Veri Hatasını (Phi_d) çekmek için Beta=0 ile hesapla
        m_temp = res.x.reshape((mesh.Ny, mesh.Nz))
        _, phi_d_only = grad_engine.compute_gradient(freqs, 10**m_temp, beta=0.0)

        # --- ROBUST KARAR MEKANİZMASI ---
        distance = abs(phi_d_only - target_phi)

        # Durum etiketi (Bilgi amaçlı)
        status = ""
        if phi_d_only < target_phi:
            status = "Overfit?"
        elif phi_d_only > target_phi * 5:
            status = "Underfit"
        else:
            status = "Makul"

        # En iyi adayı güncelle (Ne olursa olsun en yakını seç)
        if distance < global_min_dist:
            global_min_dist = distance
            global_best_beta = beta
            status += " (*)"  # Seçileni işaretle

        print(f"{beta:<10} | {phi_d_only:<15.1f} | {distance:<10.1f} | {status}")

    print("-" * 60)
    print(f" >>> SEÇİLEN OPTİMUM BETA: {global_best_beta}")

    return global_best_beta


def pbit_optimizer(
    grad_engine, m_init, freqs, mesh, n_steps, lr, t_start, t_end, beta=5.0
):
    """
    Standart P-bit (Langevin Dynamics) Simülatörü.
    """
    m_current = m_init.copy()
    energy_history = []

    best_m = m_current.copy()
    best_phi = 1e20

    for step in range(n_steps):
        # Cooling
        decay_factor = (t_end / t_start) ** (step / n_steps)
        current_temp = t_start * decay_factor

        # Gradient
        sigma_curr = 10 ** m_current.reshape((mesh.Ny, mesh.Nz))
        grad_det, phi = grad_engine.compute_gradient(freqs, sigma_curr, beta=beta)
        grad_det = grad_det.flatten()

        # Best Model Tracker
        if phi < best_phi:
            best_phi = phi
            best_m = m_current.copy()

        # Langevin Update
        noise_scale = np.sqrt(2 * current_temp)
        thermal_noise = np.random.randn(*m_current.shape) * noise_scale

        update = -lr * grad_det + thermal_noise
        m_current += update
        m_current = np.clip(m_current, -4.0, 0.0)

        energy_history.append(phi)

    return best_m, energy_history, best_phi


def tune_pbit_hyperparameters(grad_engine, freqs, mesh, m_init, beta=5.0):
    """
    P-bit simülasyonu için en iyi 'Learning Rate' ve 'Başlangıç Sıcaklığı'nı
    otomatik olarak bulur (Grid Search).
    """
    print(f"\n--- P-bit Hiperparametre Taraması (Auto-Tuning) ---")

    learning_rates = [0.0005, 0.001, 0.005]
    temp_starts = [0.01, 0.1, 0.5]

    # Tuning için kısa koşu (300 adım)
    TUNING_STEPS = 300
    TEMP_END = 0.0001

    results = []

    print(f"{'LR':<10} | {'T_start':<10} | {'Phi (300 iter)':<15} | {'Durum'}")
    print("-" * 55)

    for lr in learning_rates:
        for t_start in temp_starts:
            # Simülasyonu çalıştır
            _, _, final_phi = pbit_optimizer(
                grad_engine,
                m_init,
                freqs,
                mesh,
                n_steps=TUNING_STEPS,
                lr=lr,
                t_start=t_start,
                t_end=TEMP_END,
                beta=beta,
            )

            status = ""
            if np.isnan(final_phi) or final_phi > 1e6:
                status = "Patladı"
            elif final_phi < 1000:
                status = "İyi Aday"
            else:
                status = "Yavaş"

            print(f"{lr:<10} | {t_start:<10} | {final_phi:<15.2f} | {status}")

            if not np.isnan(final_phi):
                results.append({"lr": lr, "t_start": t_start, "phi": final_phi})

    if not results:
        print("UYARI: Tüm kombinasyonlar patladı! Varsayılan değerler dönülüyor.")
        return 0.001, 0.01

    results.sort(key=lambda x: x["phi"])
    best = results[0]

    print("-" * 55)
    print(f" >>> SEÇİLEN PARAMETRELER: LR={best['lr']}, T_start={best['t_start']}")

    return best["lr"], best["t_start"]


# --- 3. YARDIMCI ARAÇLAR (MONITOR) ---
class InversionMonitor:
    """
    L-BFGS gibi iteratif süreçleri konsolda takip etmek için callback sınıfı.
    """

    def __init__(self):
        self.iter_count = 0
        self.start_time = time.time()
        self.current_phi = 0.0
        self.current_grad_norm = 0.0
        print(f"\n{'Iter':<5} | {'Phi Data':<15} | {'|Grad|':<15} | {'Time (s)':<10}")
        print("-" * 55)

    def store_stats(self, phi, grad):
        """Objective function içinden çağrılır."""
        self.current_phi = phi
        self.current_grad_norm = np.linalg.norm(grad)

    def callback(self, xk):
        """Scipy minimize tarafından her adımda çağrılır."""
        self.iter_count += 1
        elapsed = time.time() - self.start_time
        print(
            f"{self.iter_count:<5} | {self.current_phi:<15.5e} | {self.current_grad_norm:<15.5e} | {elapsed:<10.2f}"
        )


def rbm_gradient(m_vec, rbm_model):
    """
    RBM'in 'Free Energy' gradyanını (veya rekonstrüksiyon hatasını) hesaplar.
    Bu vektör, modeli RBM'in öğrendiği şekillere doğru iten kuvvettir.
    """
    # 1. Normalizasyon: Model (-4, 0) aralığından (0, 1) aralığına
    # RBM binary (0-1) verilerle eğitildiği için bu şart.
    m_min, m_max = -4.0, 0.0
    m_norm = (m_vec - m_min) / (m_max - m_min)

    # 2. RBM Rekonstrüksiyonu (Hayal Etme)
    # P(h=1|v) -> Gizli katmanı aktif et
    h_prob = expit(
        np.dot(m_norm, rbm_model.components_.T) + rbm_model.intercept_hidden_
    )

    # P(v=1|h) -> Geriye, görünür katmana dön (Hayal edilen jeoloji)
    v_recon = expit(
        np.dot(h_prob, rbm_model.components_) + rbm_model.intercept_visible_
    )

    # 3. Kuvvet Yönü: (Mevcut Hal - Hayal Edilen Hal)
    # Bu kuvvet, p-bitleri RBM'in "ideal" gördüğü şekle iter.
    grad_direction = m_norm - v_recon

    return grad_direction


def pbit_optimizer_with_rbm(
    grad_engine,
    m_init,
    freqs,
    mesh,
    n_steps,
    lr,
    t_start,
    t_end,
    rbm_path=None,
    alpha=0.1,
    average_last_steps=1000,
):
    """
    SciML Versiyonu: Hem Fizik (Data) hem Yapay Zeka (RBM) kullanır.
    Ayrıca son adımların ORTALAMASINI (Averaging) alarak gürültüyü temizler.
    """
    m_current = m_init.copy()
    energy_history = []
    best_m = m_current.copy()
    best_phi = 1e20

    # Ortalama almak için değişkenler
    m_sum = np.zeros_like(m_current)
    avg_count = 0

    # RBM Modelini Yükle
    rbm_model = None
    if rbm_path:
        if isinstance(rbm_path, str):
            # Sadece tuning sırasında konsolu kirletmemek için print'i şarta bağlayabilirsin
            if n_steps > 1000:
                print(f" [SciML] RBM Modeli yükleniyor: {rbm_path}")
            rbm_model = joblib.load(rbm_path)
        else:
            # Eğer rbm_path yerine direkt model objesi geldiyse
            rbm_model = rbm_path

    if n_steps > 1000:  # Sadece uzun koşularda yazdır
        print(f" [SciML] Hibrit Optimizasyon Başlıyor (Alpha={alpha})...")
        print(f" [Bilgi] Son {average_last_steps} adımın ortalaması alınacak.")
    start_time = time.time()

    for step in range(n_steps):
        # Soğutma (Annealing)
        decay = (t_end / t_start) ** (step / n_steps)
        temp = t_start * decay

        # 1. Fizik Gradyanı
        sigma_curr = 10 ** m_current.reshape((mesh.Ny, mesh.Nz))
        grad_data, phi = grad_engine.compute_gradient(freqs, sigma_curr, beta=0.0)
        grad_data = grad_data.flatten()

        # 2. Yapay Zeka Gradyanı
        grad_prior = np.zeros_like(grad_data)
        if rbm_model is not None:
            grad_prior = rbm_gradient(m_current, rbm_model)

        # En iyi anlık phi'yi sakla (Bilgi amaçlı)
        if phi < best_phi:
            best_phi = phi
            # best_m'i burada güncellemiyoruz, ortalamayı kullanacağız.

        # 3. Langevin Güncellemesi
        noise_scale = np.sqrt(2 * temp)
        noise = np.random.randn(*m_current.shape) * noise_scale

        total_grad = grad_data + (alpha * grad_prior)
        update = -lr * total_grad + noise
        m_current += update
        m_current = np.clip(m_current, -4.0, 0.0)

        # --- ORTALAMA ALMA (AVERAGING) ---
        # Optimizasyonun sonlarına doğru modelleri biriktir
        if step >= (n_steps - average_last_steps):
            m_sum += m_current
            avg_count += 1

        energy_history.append(phi)

        if step % 200 == 0 or step == n_steps - 1:
            elapsed = time.time() - start_time
            print(
                f" Adım {step}/{n_steps} | Phi: {phi:.1f} | Temp: {temp:.4f} | Süre: {elapsed:.0f}s",
                end="\r",
            )

    print("\n [SciML] Optimizasyon tamamlandı.")

    # Sonuç olarak ORTALAMA modeli döndür
    if avg_count > 0:
        m_final_avg = m_sum / avg_count
        print(f" [Sonuç] {avg_count} adımın ortalaması alındı (Gürültü temizlendi).")
    else:
        # Eğer adım sayısı çok azsa mecburen son hali döndür
        m_final_avg = m_current
        print(" [Uyarı] Ortalama alınamadı (Yetersiz adım sayısı).")

    # Not: best_phi anlık en iyi değerdir, ortalama modelin phi'si değildir.
    return m_final_avg, energy_history, best_phi


def tune_sciml_hyperparameters(grad_engine, freqs, mesh, m_init, rbm_path):
    """
    SciML için en iyi Alpha (RBM Gücü) ve Learning Rate değerlerini bulur.
    Kısa süreli denemeler (Grid Search) yapar.
    """
    print(f"\n=============================================")
    print(f"   SciML AUTO-TUNING (Alpha & LR Search)")
    print(f"=============================================")

    # Denenecek Adaylar (Grid Search)
    # Alpha: Yapay zeka ne kadar baskın olsun?
    candidate_alphas = [0.5, 2.0, 5.0, 10.0]
    # LR: Ne kadar hızlı gitsin?
    candidate_lrs = [0.001, 0.002, 0.005]

    # Sabitler (Tuning için kısa tutuyoruz)
    TUNING_STEPS = 300
    TEMP_START = 0.1
    TEMP_END = 0.01  # Hızlı soğuma

    results = []

    print(f"{'Alpha':<10} | {'LR':<10} | {'Final Phi':<15} | {'Durum'}")
    print("-" * 55)

    best_combo = None
    min_phi = 1e20

    # RBM Modelini bir kere yükle (Hız için)
    rbm_model = joblib.load(rbm_path)

    for alpha in candidate_alphas:
        for lr in candidate_lrs:
            # Kısa bir simülasyon yap
            # Not: Burada pbit_optimizer_with_rbm fonksiyonunu modifiye ederek çağırıyoruz
            # Fonksiyonun içine rbm_path yerine direkt modeli verebilmek için
            # yukarıdaki fonksiyonu hafif güncellemek gerekebilir veya
            # model path'ini tekrar vererek yükletiriz (biraz yavaş olur ama çalışır).
            # Hız kazanmak için, optimizer fonksiyonunu rbm_model nesnesini alacak şekilde
            # güncellemek en iyisidir, ama şimdilik path ile devam edelim.

            _, _, final_phi = pbit_optimizer_with_rbm(
                grad_engine,
                m_init,
                freqs,
                mesh,
                n_steps=TUNING_STEPS,
                lr=lr,
                t_start=TEMP_START,
                t_end=TEMP_END,
                rbm_path=rbm_path,
                alpha=alpha,
                average_last_steps=50,  # Tuning'de son 50 adım yeter
            )

            # Durum Analizi
            status = ""
            if np.isnan(final_phi) or final_phi > 1e6:
                status = "Patladı (Unstable)"
            elif final_phi < 2000:
                status = "İyi Aday"  # Modele göre değişir
            else:
                status = "Yavaş / Kötü"

            print(f"{alpha:<10} | {lr:<10} | {final_phi:<15.1f} | {status}")

            if not np.isnan(final_phi) and final_phi < min_phi:
                min_phi = final_phi
                best_combo = (alpha, lr)

    print("-" * 55)
    print(f" >>> SEÇİLEN: Alpha={best_combo[0]}, LR={best_combo[1]}")
    return best_combo[0], best_combo[1]
