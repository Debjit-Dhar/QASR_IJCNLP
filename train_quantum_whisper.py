"""
Train Quantum Whisper with Frozen Pretrained Layers

This script trains a quantum version of Whisper where only the quantum convolutional layers
are trained while keeping all other pretrained layers frozen. It replaces classical Conv1d
layers with quantum equivalents. Supports both Google Speech Commands and LibriSpeech datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
import numpy as np
from tqdm import tqdm
import jiwer
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import argparse
from datasets import load_dataset, Audio
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
from whisper.model import Whisper, ModelDimensions
from quantum_whisper import create_quantum_whisper_tiny, freeze_non_quantum_layers
import warnings
warnings.filterwarnings('ignore')

class AudioDataset(Dataset):
    """Dataset class for Google Speech Commands using torchaudio"""
    def __init__(self, dataset, label_to_idx, max_length=1500):
        self.dataset = dataset
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        waveform, sample_rate, label, speaker_id, utterance_number = item
        
        # Preprocess audio
        audio_features = self.preprocess_audio(waveform, sample_rate)
        
        # Convert label to tensor and get the index
        label_idx = self.label_to_idx[label]
        
        return audio_features, torch.tensor(label_idx, dtype=torch.long)
    
    def preprocess_audio(self, waveform, sample_rate=16000):
        """Preprocess audio to mel spectrogram"""
        # Convert to numpy array
        audio = waveform.numpy().flatten().astype(np.float32)
        
        # Resample if necessary
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        
        # Use official Whisper preprocessing
        from whisper.audio import pad_or_trim, log_mel_spectrogram
        audio = pad_or_trim(audio)
        mel_spec = log_mel_spectrogram(audio)
        
        # Ensure correct shape for Whisper (80 mel bins, 3000 time steps)
        # Whisper expects mel spectrogram of shape (80, 3000)
        target_length = 3000
        
        if mel_spec.shape[1] > target_length:
            mel_spec = mel_spec[:, :target_length]
        else:
            # Pad with zeros to reach target length
            pad_length = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_length)), mode='constant')
        
        return torch.FloatTensor(mel_spec)

class LibriSpeechDataset(Dataset):
    """Dataset class for LibriSpeech ASR using Hugging Face datasets"""
    def __init__(self, dataset, label_to_idx, max_length=1500):
        self.dataset = dataset
        self.label_to_idx = label_to_idx
        self.max_length = max_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # LibriSpeech has 'audio' and 'text' fields
        audio = item['audio']['array']
        text = item['text'].lower().strip()
        
        # Preprocess audio
        audio_features = self.preprocess_audio(audio)
        
        # Convert text to label index (for ASR, we'll use text as label)
        # In a real ASR scenario, you might want to use character-level or subword tokenization
        label_idx = self.label_to_idx.get(text, 0)  # Default to 0 if text not found
        
        return audio_features, torch.tensor(label_idx, dtype=torch.long)
    
    def preprocess_audio(self, audio, sample_rate=16000):
        """Preprocess audio to mel spectrogram"""
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Resample if necessary (LibriSpeech is usually 16kHz, but let's be safe)
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Use official Whisper preprocessing
        from whisper.audio import pad_or_trim, log_mel_spectrogram
        audio = pad_or_trim(audio)
        mel_spec = log_mel_spectrogram(audio)
        
        # Ensure correct shape for Whisper (80 mel bins, 3000 time steps)
        target_length = 3000
        
        if mel_spec.shape[1] > target_length:
            mel_spec = mel_spec[:, :target_length]
        else:
            # Pad with zeros to reach target length
            pad_length = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_length)), mode='constant')
        
        return torch.FloatTensor(mel_spec)

def calculate_wer(predictions, targets):
    """Calculate Word Error Rate using jiwer"""
    from jiwer import wer
    # For classification tasks, we need to convert numeric predictions to strings
    # Convert predictions and targets to string representations
    pred_str = [str(p) for p in predictions]
    target_str = [str(t) for t in targets]
    
    # Calculate WER for each prediction-target pair and average
    total_wer = 0.0
    for pred, target in zip(pred_str, target_str):
        total_wer += wer(target, pred)
    
    return total_wer / len(predictions) if predictions else 0.0

class QuantumWhisperClassifier(nn.Module):
    """Quantum Whisper-based classifier for speech commands"""
    def __init__(self, quantum_whisper_model, num_classes=35):
        super().__init__()
        self.quantum_whisper = quantum_whisper_model
        self.classifier = nn.Linear(quantum_whisper_model.dims.n_audio_state, num_classes)
        
    def to(self, device):
        super().to(device)
        # Ensure the quantum whisper model is also moved to device
        if hasattr(self.quantum_whisper, 'to'):
            self.quantum_whisper = self.quantum_whisper.to(device)
        return self
        
    def forward(self, mel_spec):
        # Extract audio features using quantum Whisper encoder
        audio_features = self.quantum_whisper.embed_audio(mel_spec)
        
        # Global average pooling
        pooled_features = torch.mean(audio_features, dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        return logits

def train_quantum_model(model, train_loader, val_loader, epochs=100, device='cpu', lr=1e-3):
    """Train the quantum Whisper model with three different model saving strategies"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Track best models for different metrics
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_val_wer = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', unit="batch")
        for batch_idx, (audio, labels) in enumerate(train_pbar):
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Debug: Check tensor devices (only on first batch)
            if batch_idx == 0:
                if str(audio.device) != str(device):
                    print(f"Warning: Audio tensor on {audio.device}, expected {device}")
                if str(labels.device) != str(device):
                    print(f"Warning: Labels tensor on {labels.device}, expected {device}")
            
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        train_loss = total_train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', unit="batch")
            for audio, labels in val_pbar:
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = model(audio)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = total_val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Calculate WER
        wer = calculate_wer(predictions, targets)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, WER: {wer:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best models based on different metrics
        # 1. Best validation accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(model.state_dict(), 'best_quantum_whisper_val_acc.pth')
            print(f'New best quantum model saved (Val Acc): {val_accuracy:.2f}%')
        
        # 2. Best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_quantum_whisper_val_loss.pth')
            print(f'New best quantum model saved (Val Loss): {val_loss:.4f}')
        
        # 3. Best validation WER
        if wer < best_val_wer:
            best_val_wer = wer
            torch.save(model.state_dict(), 'best_quantum_whisper_val_wer.pth')
            print(f'New best quantum model saved (Val WER): {wer:.4f}')
        
        print('-' * 60)
    
    return model, train_losses, val_losses, val_accuracies

def evaluate_quantum_model(model, test_loader, device='cpu'):
    """Evaluate the trained quantum model"""
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc="Evaluating Quantum Model"):
            audio = audio.to(device)
            labels = labels.to(device)
            
            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    wer = calculate_wer(predictions, targets)
    
    return wer, accuracy

def evaluate_all_quantum_models(model_class, quantum_model, test_loader, device='cpu'):
    """Evaluate all three saved quantum models on test set"""
    results = {}
    
    # Load and evaluate best validation accuracy model
    try:
        model = model_class(quantum_model, num_classes=35)
        model.load_state_dict(torch.load('best_quantum_whisper_val_acc.pth', map_location=device))
        wer, acc = evaluate_quantum_model(model, test_loader, device)
        results['best_val_acc'] = {'wer': wer, 'accuracy': acc}
        print(f"Best Val Acc Quantum Model - Test WER: {wer:.4f}, Test Acc: {acc:.2f}%")
    except FileNotFoundError:
        print("Best Val Acc quantum model not found")
    
    # Load and evaluate best validation loss model
    try:
        model = model_class(quantum_model, num_classes=35)
        model.load_state_dict(torch.load('best_quantum_whisper_val_loss.pth', map_location=device))
        wer, acc = evaluate_quantum_model(model, test_loader, device)
        results['best_val_loss'] = {'wer': wer, 'accuracy': acc}
        print(f"Best Val Loss Quantum Model - Test WER: {wer:.4f}, Test Acc: {acc:.2f}%")
    except FileNotFoundError:
        print("Best Val Loss quantum model not found")
    
    # Load and evaluate best validation WER model
    try:
        model = model_class(quantum_model, num_classes=35)
        model.load_state_dict(torch.load('best_quantum_whisper_val_wer.pth', map_location=device))
        wer, acc = evaluate_quantum_model(model, test_loader, device)
        results['best_val_wer'] = {'wer': wer, 'accuracy': acc}
        print(f"Best Val WER Quantum Model - Test WER: {wer:.4f}, Test Acc: {acc:.2f}%")
    except FileNotFoundError:
        print("Best Val WER quantum model not found")
    
    return results

def plot_quantum_training_results(train_losses, val_losses, val_accuracies):
    """Plot quantum training results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Quantum Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(val_accuracies, label='Validation Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Quantum Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('quantum_whisper_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def load_pretrained_weights(quantum_model, pretrained_path):
    """Load pretrained weights into quantum model"""
    if os.path.exists(pretrained_path):
        print(f"Loading pretrained weights from {pretrained_path}")
        # Load pretrained weights and map them to quantum model
        pretrained_state = torch.load(pretrained_path, map_location='cpu')
        
        # Create a mapping for compatible layers
        quantum_state = quantum_model.state_dict()
        loaded_layers = 0
        
        for name, param in pretrained_state.items():
            if name in quantum_state and quantum_state[name].shape == param.shape:
                quantum_state[name] = param
                loaded_layers += 1
        
        quantum_model.load_state_dict(quantum_state)
        print(f"Loaded {loaded_layers} layers from pretrained model")
    else:
        print(f"Pretrained model not found at {pretrained_path}, starting with random weights")
    
    return quantum_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Quantum Whisper on Google Speech Commands')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda, default: auto)')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits for quantum layers (default: 4)')
    parser.add_argument('--pretrained_path', type=str, default='best_whisper_val_acc.pth', 
                       help='Path to pretrained model (default: best_whisper_val_acc.pth)')
    parser.add_argument('--dataset', type=str, default='google', help='Dataset to use (google/librispeech, default: google)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING QUANTUM WHISPER WITH FROZEN PRETRAINED LAYERS")
    print("="*60)
    print(f"Training for {args.epochs} epochs")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"Pretrained model path: {args.pretrained_path}")
    print(f"Dataset: {args.dataset}")
    
    # Model Architecture Information
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INFORMATION")
    print("="*60)
    print("Base Model: Whisper Tiny (Official OpenAI Implementation)")
    print("Model Dimensions:")
    print("  - Audio Encoder: 4 transformer layers, 384 hidden size, 6 attention heads")
    print("  - Text Decoder: 4 transformer layers, 384 hidden size, 6 attention heads")
    print("  - Vocabulary: 51,865 tokens")
    print("  - Parameters: ~39M")
    print("  - Mel Spectrogram: 80 mel bins × 3000 time steps")
    print("\nQuantum Enhancements:")
    print(f"  - Quantum Conv1d layers replacing classical Conv1d in audio encoder")
    print(f"  - Number of qubits: {args.n_qubits}")
    print(f"  - Quantum circuit: Amplitude embedding + parameterized rotations + CNOT gates")
    print("  - PennyLane backend: default.qubit simulator")
    print("\nTraining Strategy:")
    print("  - Only quantum layers and classifier are trained")
    print("  - All pretrained Whisper layers are frozen")
    print("  - Layer freezing: freeze_non_quantum_layers() function")
    print("="*60)
    
    # Load dataset based on user choice
    if args.dataset.lower() == 'librispeech':
        print("\nLoading LibriSpeech ASR dataset from Hugging Face...")
        
        # Load LibriSpeech dataset from Hugging Face
        try:
            # Load train split - use full dataset
            train_dataset = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="train.100",  # Use train.100 for full training
                streaming=False
            )
            
            # Load validation split - use full dataset
            val_dataset = load_dataset(
                "openslr/librispeech_asr",
                "clean", 
                split="validation",
                streaming=False
            )
            
            # Load test split - use full dataset
            test_dataset = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="test",
                streaming=False
            )
            
            print("LibriSpeech dataset loaded successfully!")
            print(f"Train samples: {len(train_dataset)}")
            print(f"Validation samples: {len(val_dataset)}")
            print(f"Test samples: {len(test_dataset)}")
            
            # For LibriSpeech, build vocabulary from ALL training data
            print("\nBuilding vocabulary from ALL LibriSpeech training data...")
            all_texts = set()
            
            # Process ALL training samples to build complete vocabulary
            print("Processing all training samples for vocabulary building...")
            for i, item in enumerate(train_dataset):
                text = item['text'].lower().strip()
                if len(text) > 0:
                    all_texts.add(text)
                
                # Progress indicator for large datasets
                if (i + 1) % 1000 == 0:
                    print(f"Processed {i + 1}/{len(train_dataset)} samples...")
            
            # Create label mapping from complete vocabulary
            label_to_idx = {text: idx for idx, text in enumerate(sorted(all_texts))}
            num_classes = len(label_to_idx)
            
            print(f"Complete vocabulary size: {num_classes}")
            print(f"Sample texts: {list(all_texts)[:10]}")
            
            # Create datasets with full data
            train_dataset = LibriSpeechDataset(train_dataset, label_to_idx)
            val_dataset = LibriSpeechDataset(val_dataset, label_to_idx)
            test_dataset = LibriSpeechDataset(test_dataset, label_to_idx)
            
        except Exception as e:
            print(f"Error loading LibriSpeech dataset: {e}")
            print("Falling back to Google Speech Commands dataset...")
            args.dataset = 'google'
    
    if args.dataset.lower() == 'google':
        print("\nLoading Google Speech Commands dataset...")
        
        # Create datasets for different splits
        train_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data", 
            url="speech_commands_v0.02",
            subset="training",
            download=True
        )
        
        val_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data", 
            url="speech_commands_v0.02",
            subset="validation",
            download=True
        )
        
        test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data", 
            url="speech_commands_v0.02",
            subset="testing",
            download=True
        )
        
        print("Dataset loaded successfully!")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
        
        # Get unique labels for classification
        all_labels = set()
        for dataset in [train_dataset, val_dataset, test_dataset]:
            for i in range(len(dataset)):
                _, _, label, _, _ = dataset[i]
                all_labels.add(label)
        
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        num_classes = len(label_to_idx)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {sorted(all_labels)}")
        
        # Create datasets
        train_dataset = AudioDataset(train_dataset, label_to_idx)
        val_dataset = AudioDataset(val_dataset, label_to_idx)
        test_dataset = AudioDataset(test_dataset, label_to_idx)
    
    # Create data loaders (smaller batch size for quantum)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Initialize quantum Whisper model
    print("\nInitializing Quantum Whisper model...")
    n_qubits = args.n_qubits  # Number of qubits for quantum layers
    quantum_whisper_model = create_quantum_whisper_tiny(n_qubits=n_qubits)
    
    # Load pretrained weights if available
    pretrained_path = args.pretrained_path  # Path to pretrained model
    quantum_whisper_model = load_pretrained_weights(quantum_whisper_model, pretrained_path)
    
    # Freeze non-quantum layers
    quantum_whisper_model = freeze_non_quantum_layers(quantum_whisper_model)
    
    # Create classifier
    model = QuantumWhisperClassifier(quantum_whisper_model, num_classes=num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Debug: Check device placement
    print(f"Model device: {next(model.parameters()).device}")
    
    print(f"Quantum model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train quantum model
    print("\n" + "="*50)
    print("Training Quantum Whisper...")
    print("="*50)
    
    trained_model, train_losses, val_losses, val_accuracies = train_quantum_model(
        model, train_loader, val_loader, epochs=args.epochs, device=device, lr=args.lr
    )
    
    # Evaluate all saved quantum models on test set
    print("\n" + "="*50)
    print("Evaluating all saved quantum models on test set...")
    print("="*50)
    
    test_results = evaluate_all_quantum_models(QuantumWhisperClassifier, quantum_whisper_model, test_loader, device)
    
    # Print final results
    print("\n" + "="*50)
    print("QUANTUM WHISPER FINAL RESULTS")
    print("="*50)
    for model_name, metrics in test_results.items():
        print(f"{model_name}: Test WER: {metrics['wer']:.4f}, Test Acc: {metrics['accuracy']:.2f}%")
    
    print(f"Best Validation Accuracy: {max(val_accuracies):.2f}%")
    print(f"Number of qubits used: {n_qubits}")
    
    # Plot training results
    print("\nGenerating quantum training plots...")
    plot_quantum_training_results(train_losses, val_losses, val_accuracies)
    
    # Save final model
    torch.save(trained_model.state_dict(), 'quantum_whisper_final.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'test_results': test_results,
        'best_val_accuracy': max(val_accuracies),
        'num_classes': num_classes,
        'label_to_idx': label_to_idx,
        'n_qubits': n_qubits,
        'training_params': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'device': str(device),
            'n_qubits': args.n_qubits,
            'pretrained_path': args.pretrained_path,
            'dataset': args.dataset
        }
    }
    
    import json
    with open('quantum_whisper_training_history.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_serializable = {}
        for key, value in training_history.items():
            if isinstance(value, np.ndarray):
                history_serializable[key] = value.tolist()
            elif isinstance(value, dict) and key == 'label_to_idx':
                history_serializable[key] = value
            else:
                history_serializable[key] = value
        json.dump(history_serializable, f, indent=2)
    
    print("\nQuantum training completed! Models saved:")
    print("- best_quantum_whisper_val_acc.pth (best validation accuracy)")
    print("- best_quantum_whisper_val_loss.pth (best validation loss)")
    print("- best_quantum_whisper_val_wer.pth (best validation WER)")
    print("- quantum_whisper_final.pth (final model)")

if __name__ == "__main__":
    main()
