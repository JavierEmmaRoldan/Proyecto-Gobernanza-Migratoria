import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# 1. Configuración del Dispositivo y Semillas (Reproducibilidad)
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 2. Carga de Dataset Real (Gobernanza Migratoria)
ruta_archivo = 'data/Dataset_Migracion.xlsx'
# Carga solo la hoja específica indicando su nombre
df = pd.read_excel(ruta_archivo, sheet_name='Datos_Migratorios') 

# Calculamos la mediana de la columna
umbral = df['Tasa_Migración_Neta'].median()

# Creamos una nueva columna: True (1) si es mayor a la mediana, False (0) si no
df['flujo_migratorio_alto'] = (df['Tasa_Migración_Neta'] > umbral).astype(int)

# Selecciona las columnas exactas que usarás como características (X)
columnas_features = ['Índice_Corrupción_(IPC_0-100)', 'Tasa_Pobreza_(%)', 'PIB_per_Cápita_(USD PPA)', 'Tasa_Desempleo_(%)', 'IDH_País'] 
X_full = df[columnas_features].values

# Selecciona tu variable objetivo (y). 
columna_target = 'flujo_migratorio_alto' 
y_full = df[columna_target].values

# 3. Definición del Dataset para PyTorch
class MigracionDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MigracionDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(MigracionDataset(X_val, y_val), batch_size=32, shuffle=False)

# 4. Arquitectura del Modelo (Multi-Layer Perceptron)
class MLPClasificador(nn.Module):
    def __init__(self, input_size):
        super(MLPClasificador, self).__init__()
        self.red = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Capa final para probabilidad binaria
        )

    def forward(self, x):
        return self.red(x)

modelo = MLPClasificador(input_size=5).to(device)

# 5. Pipeline de Entrenamiento y Validación
criterion = nn.BCELoss()
optimizer = optim.Adam(modelo.parameters(), lr=0.01)
epochs = 50

print("\n--- Iniciando Entrenamiento ---")
for epoch in range(epochs):
    # --- FASE DE ENTRENAMIENTO ---
    modelo.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = modelo(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)
    
    # --- FASE DE VALIDACIÓN ---
    modelo.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = modelo(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            
            # Convertir probabilidades a clases (0 o 1)
            preds = (outputs >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_targets, all_preds)
    
    # Imprimir métricas cada 10 épocas
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")

print("\nPipeline finalizado con éxito.")