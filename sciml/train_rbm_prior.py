import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import joblib  # Modeli kaydetmek için

# 1. Veriyi Yükle
print("Veri yükleniyor...")
X = np.load("geology_dataset.npy")
NY, NZ = 30, 25  # Mesh boyutunu bilmemiz lazım

# 2. RBM Modeli Kur
# n_components: Gizli nöron sayısı (Ne kadar karmaşık şekilleri öğrenecek?)
rbm = BernoulliRBM(
    n_components=64, learning_rate=0.01, batch_size=10, n_iter=20, verbose=1
)

# 3. Eğit
print("RBM Eğitiliyor (Bu işlem jeolojiyi öğreniyor)...")
rbm.fit(X)

# 4. Ağırlıkları Görselleştir (Ne öğrenmiş?)
plt.figure(figsize=(10, 10))
for i in range(64):
    plt.subplot(8, 8, i + 1)
    # RBM'in öğrendiği her bir "özellik" (feature)
    plt.imshow(rbm.components_[i].reshape(NY, NZ), cmap="viridis")
    plt.axis("off")
plt.suptitle("RBM'in Öğrendiği Jeolojik Filtreler (Weights)")
plt.show()

# 5. Kaydet
joblib.dump(rbm, "rbm_geology_prior.pkl")
print("Model 'rbm_geology_prior.pkl' olarak kaydedildi.")
