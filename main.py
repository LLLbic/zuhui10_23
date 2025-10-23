import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os
from tqdm import tqdm  # 导入 tqdm 用于进度条

# Custom Dataset Class
class IIoTDataset(Dataset):
    def __init__(self, features, labels, seq_len=1):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        feature = self.features[idx].reshape(self.seq_len, -1)
        return feature, self.labels[idx]

# Self-Adaptive Attention Layer
class SelfAdaptiveAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAdaptiveAttention, self).__init__()
        self.Wq = nn.Linear(hidden_size, hidden_size)  # Query
        self.Wk = nn.Linear(hidden_size, hidden_size)  # Key
    
    def forward(self, hidden_states):
        Q = self.Wq(hidden_states)
        K = self.Wk(hidden_states)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (hidden_states.size(-1) ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context_vector = torch.matmul(attention_weights, hidden_states)
        return context_vector.mean(dim=1)

# Enhanced BiLSTM Model
class EnhancedBiLSTM(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size1=100, hidden_size2=50, dense_size=20, dropout=0.2):
        super(EnhancedBiLSTM, self).__init__()
        self.bilstm1 = nn.LSTM(input_size, hidden_size1, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_size1 * 2, hidden_size2, bidirectional=True, batch_first=True)
        self.attention = SelfAdaptiveAttention(hidden_size2 * 2)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size2 * 2, dense_size)
        self.output = nn.Linear(dense_size, num_classes)
    
    def forward(self, x):
        out, _ = self.bilstm1(x)
        out = self.dropout(out)
        out, _ = self.bilstm2(out)
        out = self.dropout(out)
        out = self.attention(out)
        out = torch.relu(self.dense(out))
        out = self.output(out)
        return out

# Training Function with Metrics Tracking and Progress Bar
def train_model(model, train_loader, val_loader, optimizer, criterion, epochs=10, device='cuda'):
    model.to(device)
    model.train()
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_preds, train_labels = [], []
        # 添加 tqdm 进度条，显示训练批次进度
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for features, labels in train_bar:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            # 更新进度条显示当前批次 loss
            train_bar.set_postfix({'batch_loss': loss.item()})
        
        train_loss = total_train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []
        # 添加 tqdm 进度条，显示验证批次进度
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
        with torch.no_grad():
            for features, labels in val_bar:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                # 更新进度条显示当前批次 loss
                val_bar.set_postfix({'batch_loss': loss.item()})
        
        val_loss = total_val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # 打印 epoch 整体结果
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Testing Function
def test_model(model, test_loader, device='cuda'):
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []
    # 添加 tqdm 进度条，显示测试批次进度
    test_bar = tqdm(test_loader, desc="Testing", leave=False)
    with torch.no_grad():
        for features, labels in test_bar:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Test Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    return all_preds, all_labels

# Plotting Function
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, dataset_name):
    epochs_range = range(1, epochs + 1)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy vs Epochs ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'accuracy_{dataset_name}.png')
    plt.close()
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss vs Epochs ({dataset_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_{dataset_name}.png')
    plt.close()

# Main Workflow
def main(dataset_path, batch_size=16, epochs=10, total_rows=200, train_rows=100, seq_len=1):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if file exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

    # Load and Preprocess Dataset
    print(f"Loading {total_rows} rows from {dataset_path}")
    df = pd.read_csv(dataset_path, nrows=total_rows)
    
    # Handle column names (strip spaces)
    df.columns = df.columns.str.strip()
    
    # Replace problematic values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    # Check if 'Label' column exists
    if 'Label' not in df.columns:
        raise ValueError("CSV file must contain a 'Label' column.")
    
    # Select features (all columns except 'Label')
    feature_columns = [col for col in df.columns if col != 'Label']
    features = df[feature_columns].values
    labels = df['Label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_temp = scaler.fit_transform(features[:train_rows])
    X_test = scaler.transform(features[train_rows:])
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    y_temp = labels[:train_rows]
    y_test = labels[train_rows:]
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    
    # Split training data into train and validation (80:20)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Testing samples: {len(X_test)}")
    
    # Create DataLoaders with pin_memory for faster GPU data transfer
    train_dataset = IIoTDataset(X_train, y_train, seq_len=seq_len)
    val_dataset = IIoTDataset(X_val, y_val, seq_len=seq_len)
    test_dataset = IIoTDataset(X_test, y_test, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Model Setup
    input_size = X_train.shape[1]
    model = EnhancedBiLSTM(input_size, num_classes, hidden_size1=100, hidden_size2=50, dense_size=20)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Train and collect metrics
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, criterion, epochs=epochs, device=device
    )
    
    # Plot metrics
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies, epochs, dataset_name)
    print(f"Plots saved as accuracy_{dataset_name}.png and loss_{dataset_name}.png")
    
    # Test
    all_preds, all_labels = test_model(model, test_loader, device=device)
    
    # Save results to CSV
    results = pd.DataFrame({
        'True_Label': le.inverse_transform(all_labels),
        'Predicted_Label': le.inverse_transform(all_preds)
    })
    output_file = 'test_results_' + dataset_name + '.csv'
    results.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# Command-line interface
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM with self-adaptive attention on GPU with progress bar")
    parser.add_argument('--dataset_path', type=str, default='./CIC-IDS-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
                        help='Path to the input CSV file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--total_rows', type=int, default=220000, help='Total number of rows to load')
    parser.add_argument('--train_rows', type=int, default=110000, help='Number of rows for training')
    args = parser.parse_args()
    
    main(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        total_rows=args.total_rows,
        train_rows=args.train_rows
    )