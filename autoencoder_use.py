import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
autoencoder = load_model("network_latency_autoencoder.keras")

# Charger les nouvelles données (avec attaques DDoS)
ddos_data_path = "data/data_tcp_6h_120s_attack.csv"  # Remplace par le chemin de ton fichier
ddos_data = pd.read_csv(ddos_data_path)

# Convertir la colonne _time en format datetime
ddos_data["_time"] = pd.to_datetime(ddos_data["_time"], format='ISO8601', errors='coerce')

# Vérifier et supprimer les valeurs non convertibles
ddos_data = ddos_data.dropna(subset=["_time"])

# Extraire et normaliser les colonnes _value
ddos_values = ddos_data["_value"].values.reshape(-1, 1)
scaler = MinMaxScaler()
normalized_ddos_values = scaler.fit_transform(ddos_values)

# Recontruire les données avec l'autoencodeur
reconstructed_ddos = autoencoder.predict(normalized_ddos_values)

# Calculer les erreurs de reconstruction
reconstruction_errors = np.mean((normalized_ddos_values - reconstructed_ddos) ** 2, axis=1)

# Fixer un seuil pour détecter les anomalies
threshold = 0.000598948281836561  # Utiliser l'erreur moyenne précédente comme référence
anomalies = reconstruction_errors > threshold

# Ajouter les anomalies détectées aux données
ddos_data["Anomaly"] = anomalies
ddos_data["Reconstruction_Error"] = reconstruction_errors

# Sauvegarder les résultats dans un fichier
ddos_data.to_csv("ddos_detection_results.csv", index=False)

# Résumé des anomalies détectées
num_anomalies = anomalies.sum()
print(f"Nombre d'anomalies détectées : {num_anomalies}")
print("Les résultats détaillés ont été sauvegardés dans 'ddos_detection_results.csv'.")

# Création des subplots
fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Subplot 1 : Latence en fonction du temps
axs[0].plot(ddos_data["_time"], ddos_data["_value"], label="Latency", color="blue")
axs[0].set_title("Latency Over Time")
axs[0].set_ylabel("Latency (ms)")
axs[0].legend()
axs[0].grid(True)

# Subplot 2 : Erreurs de reconstruction et anomalies
axs[1].plot(ddos_data["_time"], reconstruction_errors, label="Reconstruction Error", color="green")
axs[1].axhline(y=threshold, color="red", linestyle="--", label="Threshold")
axs[1].scatter(ddos_data["_time"][anomalies], reconstruction_errors[anomalies], color="orange", label="Anomalies", marker='x')
axs[1].set_title("Reconstruction Errors with Anomalies")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Reconstruction Error")
axs[1].legend()
axs[1].grid(True)
print(reconstruction_errors[anomalies])
print(ddos_data["_time"][anomalies])
# Ajustement de l'affichage
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()