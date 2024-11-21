import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Liste des fichiers d'entraînement (sans attaques)
file_paths = [
    "data/data_icmp_1h_no_attack.csv",
    "data/data_icmp_6h_no_attack.csv",
    "data/data_icmp_10m_no_attack.csv",
    "data/data_icmp_12h_no_attack.csv"
]

# Charger et combiner les données depuis tous les fichiers
data_frames = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    print(f"Fichier {file_path} chargé avec {len(df)} lignes.")  # Vérifier le nombre de lignes par fichier
    data_frames.append(df)

# Combiner toutes les données en un seul DataFrame
network_data = pd.concat(data_frames, ignore_index=True)
print(f"Total des données combinées : {len(network_data)} lignes.")  # Vérifier la combinaison

# Supprimer les lignes avec des valeurs nulles ou zéro pour éviter les biais
network_data = network_data[network_data["_value"] > 0]
print(f"Total des données après suppression des valeurs nulles ou zéro : {len(network_data)} lignes.")

# Extraire et normaliser la colonne _value
values = network_data["_value"].values.reshape(-1, 1)
scaler = MinMaxScaler()
normalized_values = scaler.fit_transform(values)

# Diviser les données en train et test
train_size = int(len(normalized_values) * 0.8)
train_data = normalized_values[:train_size]
test_data = normalized_values[train_size:]

print(f"Taille des données d'entraînement : {len(train_data)}")
print(f"Taille des données de test : {len(test_data)}")

# Construire un autoencodeur simple
autoencoder = Sequential([
    Dense(32, activation='relu', input_dim=1),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(16, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # La sortie est normalisée
])

# Compiler le modèle
autoencoder.compile(optimizer='adam', loss='mse')

# Entraîner le modèle
history = autoencoder.fit(
    train_data,
    train_data,
    epochs=50,
    batch_size=32,
    validation_data=(test_data, test_data),
    verbose=1
)

# Évaluer le modèle sur les données de test
reconstructed = autoencoder.predict(test_data)
reconstruction_errors = np.mean((test_data - reconstructed) ** 2, axis=1)

# Fixer le seuil basé sur la distribution des erreurs
mean_error = np.mean(reconstruction_errors)
std_error = np.std(reconstruction_errors)
k = 8  # Multiplicateur pour fixer le seuil
threshold = mean_error + k * std_error

print(f"Seuil calculé pour les anomalies (mean + {k} * std) : {threshold}")

# Sauvegarder le modèle
autoencoder.save("network_latency_autoencoder.h5")
autoencoder.save("network_latency_autoencoder.keras")

print("Modèle entraîné et sauvegardé sous 'network_latency_autoencoder.h5'.")