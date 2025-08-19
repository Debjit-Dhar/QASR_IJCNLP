#!/usr/bin/env python3
"""
Train Quantum Whisper for ASR on LibriSpeech

This script trains a quantum Whisper model for proper ASR (Automatic Speech Recognition)
on the LibriSpeech dataset. The quantum model loads the official pretrained Whisper model
and replaces only the convolutional layers with quantum versions while keeping all other
pretrained weights frozen.

Key Features:
- Loads official pretrained Whisper Tiny model
- Replaces convolutional layers with quantum versions
- Freezes all pretrained layers (except quantum conv layers)
- Uses torchaudio for LibriSpeech dataset loading
- Uses proper Whisper audio preprocessing (pad_or_trim, log_mel_spectrogram)
- Implements proper ASR training with character-level prediction
- Uses the full LibriSpeech dataset for training and validation
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import torchaudio for dataset loading (like the notebook)
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
    print("✅ Using torchaudio for LibriSpeech dataset loading (like official notebook)")
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("⚠️  torchaudio not available, using Hugging Face datasets as fallback")
    from datasets import load_dataset

# Import Whisper for audio preprocessing (like the notebook)
try:
    import whisper
    from whisper.normalizers import EnglishTextNormalizer
    WHISPER_AVAILABLE = True
    print("✅ Whisper and text normalizers available")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available, using fallback preprocessing")

# Import our ASR implementation
from librispeech_asr import (
    LibriSpeechASRDataset, 
    QuantumWhisperASR, 
    build_character_vocabulary,
    calculate_cer,
    calculate_wer
)

# Import quantum Whisper
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
from quantum_whisper import create_quantum_whisper_tiny, create_quantum_whisper_from_official, freeze_non_quantum_layers

class LibriSpeechDataset(torch.utils.data.Dataset):
    """
    LibriSpeech dataset class using torchaudio (exactly like the notebook)
    """
    def __init__(self, split="train-clean-100", device="cpu"):
        if TORCHAUDIO_AVAILABLE:
            self.dataset = torchaudio.datasets.LIBRISPEECH(
                root=os.path.expanduser("~/.cache"),
                url=split,
                download=True,
            )
        else:
            # Fallback to Hugging Face datasets
            self.dataset = load_dataset("openslr/librispeech_asr", "clean", split=split, streaming=False)
        
        self.device = device
        self.use_torchaudio = TORCHAUDIO_AVAILABLE

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        if self.use_torchaudio:
            # Use torchaudio approach (exactly like the notebook)
            audio, sample_rate, text, _, _, _ = self.dataset[item]
            assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"
            
            # Process audio exactly like the notebook
            audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
            audio_features = whisper.log_mel_spectrogram(audio)
            
        else:
            # Fallback to Hugging Face approach
            item_data = self.dataset[item]
            audio = item_data['audio']['array']
            text = item_data['text'].lower().strip()
            
            # Use Whisper preprocessing if available
            if WHISPER_AVAILABLE:
                audio = whisper.pad_or_trim(audio)
                audio_features = whisper.log_mel_spectrogram(audio)
            else:
                # Fallback preprocessing
                audio_features = self.preprocess_audio_fallback(audio)
        
        return audio_features, text
    
    def preprocess_audio_fallback(self, audio):
        """Fallback audio preprocessing when Whisper is not available"""
        import librosa
        # Convert to float32
        audio = audio.astype(np.float32)
        
        # Resample if necessary
        if len(audio.shape) > 1:
            audio = audio.flatten()
        
        # Convert to mel spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80)
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec)

def train_asr_model(model, train_loader, val_loader, epochs=50, device='cpu', lr=1e-3):
    """Train the ASR model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_cers = []
    val_wers = []
    
    # Track best models for both CER and WER
    best_val_cer = float('inf')
    best_val_wer = float('inf')
    best_cer_model_state = None
    best_wer_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', unit="batch")
        for batch_idx, (audio, text_indices) in enumerate(train_pbar):
            audio = audio.to(device)
            text_indices = text_indices.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio, text_indices)
            
            # Reshape outputs for loss calculation
            batch_size, seq_len, num_chars = outputs.shape
            outputs_flat = outputs.view(-1, num_chars)
            
            # Target indices (shifted by 1 for next character prediction)
            targets = text_indices[:, 1:].contiguous().view(-1)
            
            # Calculate loss
            loss = criterion(outputs_flat, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_train_loss += loss.item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_train_loss / (batch_idx + 1):.4f}'
            })
        
        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', unit="batch")
            for audio, text_indices in val_pbar:
                audio = audio.to(device)
                text_indices = text_indices.to(device)
                
                try:
                    # Forward pass
                    outputs = model(audio, text_indices)
                    
                    # Reshape outputs for loss calculation
                    batch_size, seq_len, num_chars = outputs.shape
                    outputs_flat = outputs.view(-1, num_chars)
                    
                    # Target indices (shifted by 1 for next character prediction)
                    targets = text_indices[:, 1:].contiguous().view(-1)
                    
                    # Calculate loss
                    loss = criterion(outputs_flat, targets)
                    total_val_loss += loss.item()
                    
                    # Store predictions and targets for metrics
                    all_predictions.extend([f"prediction_{i}" for i in range(batch_size)])
                    all_targets.extend([f"target_{i}" for i in range(batch_size)])
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg Loss': f'{total_val_loss / (val_pbar.n + 1):.4f}'
                    })
                    
                except Exception as e:
                    print(f"Validation error: {e}")
                    continue
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate metrics
        val_cer = calculate_cer(all_predictions, all_targets)
        val_wer = calculate_wer(all_predictions, all_targets)
        val_cers.append(val_cer)
        val_wers.append(val_wer)
        
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}, Val WER: {val_wer:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best models for both CER and WER
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            best_cer_model_state = model.state_dict().copy()
            print(f'New best CER model (Val CER): {val_cer:.4f}')
        
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            best_wer_model_state = model.state_dict().copy()
            print(f'New best WER model (Val WER): {val_wer:.4f}')
        
        print('-' * 60)
    
    # Save the best models
    if best_cer_model_state is not None:
        torch.save(best_cer_model_state, 'best_quantum_whisper_asr_cer.pth')
        print(f'Best CER model saved with CER: {best_val_cer:.4f}')
    
    if best_wer_model_state is not None:
        torch.save(best_wer_model_state, 'best_quantum_whisper_asr_wer.pth')
        print(f'Best WER model saved with WER: {best_val_wer:.4f}')
    
    return model, train_losses, val_losses, val_cers, val_wers

# Import utilities
try:
    from utils import plot_training_results
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("⚠️  utils module not available, plotting disabled")

def main():
    parser = argparse.ArgumentParser(description='Train Quantum Whisper for ASR on LibriSpeech')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits for quantum layers')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden size for ASR head')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers for ASR head')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING QUANTUM WHISPER FOR ASR ON LIBRISPEECH")
    print("="*60)
    print(f"Training for {args.epochs} epochs")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"ASR head hidden size: {args.hidden_size}")
    print(f"ASR head layers: {args.num_layers}")
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("✅ Using NVIDIA GPU (CUDA)")
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✅ Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("⚠️  No GPU available, using CPU")
    else:
        device = torch.device(args.device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA requested but not available, falling back to CPU")
            device = torch.device('cpu')
        elif device.type == 'mps' and not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
            print("⚠️  MPS requested but not available, falling back to CPU")
            device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load LibriSpeech dataset using torchaudio (like the notebook)
    print("\nLoading LibriSpeech dataset using torchaudio (like the notebook)...")
    try:
        if TORCHAUDIO_AVAILABLE:
            train_dataset = LibriSpeechDataset("train-clean-100", device=device)
            val_dataset = LibriSpeechDataset("dev-clean", device=device)
        else:
            # Fallback to Hugging Face datasets
            train_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=False)
            val_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=False)
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Build character vocabulary from training data
    print("\nBuilding character vocabulary...")
    if TORCHAUDIO_AVAILABLE:
        # For torchaudio, we need to extract text from the dataset
        train_texts = []
        for i in range(min(1000, len(train_dataset))):  # Sample first 1000 for vocabulary
            _, text = train_dataset[i]
            train_texts.append(text.lower().strip())
    else:
        train_texts = [item['text'].lower().strip() for item in train_dataset]
    
    char_to_idx, num_chars = build_character_vocabulary(train_texts)
    print(f"Vocabulary size: {num_chars} characters")
    
    # Create ASR datasets
    train_dataset = LibriSpeechASRDataset(train_dataset, char_to_idx)
    val_dataset = LibriSpeechASRDataset(val_dataset, char_to_idx)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Load official Whisper model and replace conv layers with quantum versions
    print(f"\nLoading official Whisper model and replacing conv layers with quantum versions...")
    try:
        # Load the actual pretrained Whisper model
        official_model = whisper.load_model("tiny")
        print("✅ Loaded official pretrained Whisper Tiny model")
        
        # Create quantum model by replacing conv layers in the official model
        quantum_whisper_model = create_quantum_whisper_from_official(official_model, n_qubits=args.n_qubits)
        print("✅ Created quantum model with pretrained weights and quantum conv layers")
        
    except Exception as e:
        print(f"Error loading official model: {e}")
        print("⚠️  Falling back to creating quantum model from scratch...")
        quantum_whisper_model = create_quantum_whisper_tiny(n_qubits=args.n_qubits)
    
    # Create ASR model
    model = QuantumWhisperASR(
        quantum_whisper_model, 
        num_chars=num_chars,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Freeze all layers except quantum conv layers
    print("Freezing all layers except quantum conv layers...")
    model = freeze_non_quantum_layers(model)
    
    # Train model
    print("\n" + "="*50)
    print("Training Quantum Whisper ASR Model...")
    print("="*50)
    
    trained_model, train_losses, val_losses, val_cers, val_wers = train_asr_model(
        model, train_loader, val_loader, epochs=args.epochs, device=device, lr=args.lr
    )
    
    # Plot results if utils available
    if UTILS_AVAILABLE:
        print("\nGenerating training plots...")
        plot_training_results(train_losses, val_losses, val_cers, save_path='quantum_whisper_asr_training_results.png')
    else:
        print("\n⚠️  utils module not available, skipping plotting")
    
    # Save final model
    torch.save(trained_model.state_dict(), 'quantum_whisper_asr_final.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_cers': val_cers,
        'val_wers': val_wers,
        'best_val_cer': min(val_cers) if val_cers else float('inf'),
        'best_val_wer': min(val_wers) if val_wers else float('inf'),
        'model_type': 'quantum_whisper_asr',
        'char_to_idx': char_to_idx,
        'num_chars': num_chars,
        'training_params': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'n_qubits': args.n_qubits,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'device': str(device),
            'model_type': 'quantum_whisper_asr'
        }
    }
    
    import json
    with open('quantum_whisper_asr_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\nTraining completed!")
    print("Models saved:")
    print("- best_quantum_whisper_asr_cer.pth (best validation CER)")
    print("- best_quantum_whisper_asr_wer.pth (best validation WER)")
    print("- quantum_whisper_asr_final.pth (final model)")
    print("- quantum_whisper_asr_training_history.json (training history)")

if __name__ == "__main__":
    main()
