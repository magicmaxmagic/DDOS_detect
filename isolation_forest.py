import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Fonction pour trouver le taux de contamination optimal
def find_optimal_rate(data, lower_percentile=99.5, upper_percentile=99.99):
    """
    Trouve le taux de contamination optimal pour Isolation Forest en fonction des percentiles.

    Args:
        data (pd.Series or np.array): Les données à analyser.
        lower_percentile (float): Le seuil inférieur pour considérer les anomalies (en pourcentage).
        upper_percentile (float): Le seuil supérieur pour considérer les anomalies (en pourcentage).

    Returns:
        float: Le taux de contamination optimal (limité à 0.5).
    """
    # Convertir les données en tableau numpy si nécessaire
    data = np.array(data)

    # Calculer les valeurs de seuils basées sur les percentiles
    lower_threshold = np.percentile(data, lower_percentile)
    upper_threshold = np.percentile(data, upper_percentile)

    # Nombre de points en dehors des seuils
    anomalies = np.sum((data < lower_threshold) | (data > upper_threshold))

    # Calcul du taux de contamination optimal
    contamination_rate = anomalies / len(data)

    # Limiter le taux de contamination à 0.5 (maximum acceptable par Scikit-learn)
    return min(max(contamination_rate, 0.01), 0.5)

# Charger les données
file_path = 'data/data_tcp_24h_120s_attack.csv'  # Remplacez par le chemin de votre fichier
data = pd.read_csv(file_path)

# Conversion du temps en datetime
data['_time'] = pd.to_datetime(data['_time'], format='mixed', errors='coerce')
data = data.dropna(subset=['_time'])  # Retirer les dates invalides

# Préparation des données pour l'analyse
latency_values = data['_value'].astype(float)

# Normalisation des données
scaler = StandardScaler()
latency_scaled = scaler.fit_transform(latency_values.values.reshape(-1, 1))

# Calcul du taux de contamination optimal
optimal_rate = find_optimal_rate(latency_values, lower_percentile=99.5, upper_percentile=99.99)
print(f"Taux de contamination optimal: {optimal_rate:.4f}")

# Isolation Forest avec le taux optimal
isolation_forest = IsolationForest(n_estimators=200, contamination=optimal_rate, random_state=42)
data['anomaly_iforest'] = isolation_forest.fit_predict(latency_scaled)
data['anomaly_iforest'] = data['anomaly_iforest'].apply(lambda x: 'Anomalie' if x == -1 else 'Normal')

# Détection avec un seuil statistique (3 écarts-types)
mean = latency_values.mean()
std_dev = latency_values.std()
threshold = mean + 3 * std_dev  # Seuil basé sur 3 écarts-types
data['anomaly_threshold'] = data['_value'].apply(lambda x: 'Anomalie' if x > threshold else 'Normal')

# Visualisation avancée des anomalies
plt.figure(figsize=(16, 8))

# Graphique de latence
plt.plot(data['_time'], latency_values, label='Latence', color='blue', alpha=0.6)

# Points détectés par Isolation Forest
plt.scatter(
    data['_time'][data['anomaly_iforest'] == 'Anomalie'], 
    latency_values[data['anomaly_iforest'] == 'Anomalie'], 
    color='red', label='Anomalies (Isolation Forest)', s=40
)

# Points détectés par la méthode seuil
plt.scatter(
    data['_time'][data['anomaly_threshold'] == 'Anomalie'], 
    latency_values[data['anomaly_threshold'] == 'Anomalie'], 
    color='orange', label='Anomalies (Seuil 3σ)', s=40
)

# Ajout de la ligne de seuil
plt.axhline(y=threshold, color='green', linestyle='--', label=f'Seuil 3σ ({threshold:.2f})')

# Informations supplémentaires sur le graphique
plt.xlabel('Temps')
plt.ylabel('Latence (ms)')
plt.title('Détection des anomalies dans les latences réseau')
plt.legend()
plt.grid(True, alpha=0.5)

# Affichage du graphique
plt.tight_layout()
plt.show()

# Sauvegarder les résultats dans un fichier CSV
output_file = 'resultats_anomalies_avance.csv'
data.to_csv(output_file, index=False)
print(f"Les résultats ont été sauvegardés dans '{output_file}'")