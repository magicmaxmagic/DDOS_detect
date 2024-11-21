from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd

def optimize_autoencoder(file_paths, layer_configs, epochs_list, batch_sizes, k_values):
    """
    Optimise les paramètres de l'autoencodeur pour les données fournies.

    :param file_paths: Liste des chemins des fichiers CSV contenant les données d'entraînement.
    :param layer_configs: Liste de configurations pour les couches de l'autoencodeur (ex. [[32, 16, 8], [64, 32, 16]]).
    :param epochs_list: Liste du nombre d'époques à tester (ex. [50, 100]).
    :param batch_sizes: Liste des tailles de batch à tester (ex. [32, 64]).
    :param k_values: Liste des valeurs de k pour calculer le seuil (ex. [5, 8, 10]).
    :return: Un dictionnaire contenant les paramètres optimaux et le modèle entraîné.
    """
    # Charger et combiner les données
    data_frames = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        data_frames.append(df)

    # Combiner toutes les données en un seul DataFrame
    network_data = pd.concat(data_frames, ignore_index=True)
    network_data = network_data[network_data["_value"] > 0]  # Supprimer les valeurs nulles ou zéro

    # Normaliser les données
    values = network_data["_value"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(values)

    # Diviser les données en train et test
    train_size = int(len(normalized_values) * 0.8)
    train_data = normalized_values[:train_size]
    test_data = normalized_values[train_size:]

    # Variables pour stocker les meilleurs paramètres
    best_config = None
    best_threshold = None
    best_model = None
    lowest_validation_loss = float('inf')

    # Boucle sur les combinaisons de paramètres
    for layers in layer_configs:
        for epochs in epochs_list:
            for batch_size in batch_sizes:
                # Construire l'autoencodeur
                autoencoder = Sequential()
                for units in layers:
                    autoencoder.add(Dense(units, activation='relu', input_dim=1 if autoencoder.layers == [] else None))
                autoencoder.add(Dense(1, activation='sigmoid'))  # Dernière couche

                # Compiler le modèle
                autoencoder.compile(optimizer='adam', loss='mse')

                # Entraîner le modèle
                history = autoencoder.fit(
                    train_data,
                    train_data,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(test_data, test_data),
                    verbose=0
                )

                # Calculer les erreurs de reconstruction
                reconstructed = autoencoder.predict(test_data)
                reconstruction_errors = np.mean((test_data - reconstructed) ** 2, axis=1)

                # Calculer le seuil optimal pour chaque k
                mean_error = np.mean(reconstruction_errors)
                std_error = np.std(reconstruction_errors)
                for k in k_values:
                    threshold = mean_error + k * std_error

                    # Évaluer la perte de validation pour ce modèle
                    val_loss = history.history['val_loss'][-1]
                    if val_loss < lowest_validation_loss:
                        # Mettre à jour les meilleurs paramètres
                        lowest_validation_loss = val_loss
                        best_config = {
                            "layers": layers,
                            "epochs": epochs,
                            "batch_size": batch_size,
                            "k": k,
                            "threshold": threshold
                        }
                        best_threshold = threshold
                        best_model = autoencoder

    return {
        "best_model": best_model,
        "best_config": best_config,
        "scaler": scaler,
        "threshold": best_threshold
    }

    # Liste des fichiers CSV
file_paths = [
    "data/data_icmp_1h_no_attack.csv",
    "data/data_icmp_6h_no_attack.csv",
    "data/data_icmp_10m_no_attack.csv",
    "data/data_icmp_12h_no_attack.csv"
]

# Paramètres à tester
layer_configs = [[32, 16, 8], [64, 32, 16], [128, 64, 32]]  # Différentes tailles de couches
epochs_list = [50, 100]  # Différents nombres d'époques
batch_sizes = [32, 64]  # Différentes tailles de batch
k_values = [5, 8, 10]  # Différentes valeurs pour k

# Optimiser les paramètres
result = optimize_autoencoder(file_paths, layer_configs, epochs_list, batch_sizes, k_values)

# Résultat
print("Meilleurs paramètres :")
print(result["best_config"])

# Sauvegarder le modèle optimal
result["best_model"].save("optimized_autoencoder.keras")
print("Modèle optimal sauvegardé sous 'optimized_autoencoder.h5'.")