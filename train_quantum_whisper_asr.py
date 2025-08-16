#!/usr/bin/env python3
"""
Train Quantum Whisper for ASR on LibriSpeech

This script trains a quantum Whisper model for proper ASR (Automatic Speech Recognition)
on the LibriSpeech dataset using character-level prediction.
"""

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
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Import our ASR implementation
from librispeech_asr import (
    LibriSpeechASRDataset, 
    QuantumWhisperASR, 
    build_character_vocabulary,
    calculate_cer
)

# Import quantum Whisper
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
from quantum_whisper import create_quantum_whisper_tiny, freeze_non_quantum_layers

def train_asr_model(model, train_loader, val_loader, epochs=50, device='cpu', lr=1e-3):
    """Train the ASR model"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_losses = []
    val_losses = []
    val_cers = []
    
    # Track best model
    best_val_cer = float('inf')
    
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
                
                # Forward pass
                outputs = model(audio, text_indices)
                
                # Calculate validation loss
                batch_size, seq_len, num_chars = outputs.shape
                outputs_flat = outputs.view(-1, num_chars)
                targets = text_indices[:, 1:].contiguous().view(-1)
                loss = criterion(outputs_flat, targets)
                total_val_loss += loss.item()
                
                # Store predictions and targets for CER calculation
                all_predictions.append(outputs.cpu())
                all_targets.append(text_indices.cpu())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_val_loss / (val_pbar.n + 1):.4f}'
                })
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate CER
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Get character vocabulary from the model
        char_to_idx = model.asr_decoder.char_to_idx
        val_cer = calculate_cer(all_predictions, all_targets, char_to_idx)
        val_cers.append(val_cer)
        
        # Learning rate scheduling
        scheduler.step()
        
        print(f'\nEpoch {epoch+1}/{epochs}:')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val CER: {val_cer:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if val_cer < best_val_cer:
            best_val_cer = val_cer
            torch.save(model.state_dict(), 'best_quantum_whisper_asr.pth')
            print(f'New best ASR model saved (Val CER): {val_cer:.4f}')
        
        print('-' * 60)
    
    return model, train_losses, val_losses, val_cers

def evaluate_asr_model(model, test_loader, device='cpu'):
    """Evaluate the trained ASR model"""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for audio, text_indices in tqdm(test_loader, desc="Evaluating ASR Model"):
            audio = audio.to(device)
            text_indices = text_indices.to(device)
            
            # Forward pass
            outputs = model(audio, text_indices)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(text_indices.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate CER
    char_to_idx = model.asr_decoder.char_to_idx
    test_cer = calculate_cer(all_predictions, all_targets, char_to_idx)
    
    return test_cer

# Import utilities
from utils import plot_training_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Quantum Whisper for ASR on LibriSpeech')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda, default: auto)')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits for quantum layers (default: 4)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden size for ASR decoder (default: 384)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("TRAINING QUANTUM WHISPER FOR ASR ON LIBRISPEECH")
    print("="*60)
    print(f"Training for {args.epochs} epochs")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"ASR Decoder hidden size: {args.hidden_size}")
    print(f"ASR Decoder LSTM layers: {args.num_layers}")
    
    # Model Architecture Information
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE INFORMATION")
    print("="*60)
    print("Base Model: Whisper Tiny (Official OpenAI Implementation)")
    print("Model Dimensions:")
    print("  - Audio Encoder: 4 transformer layers, 384 hidden size, 6 attention heads")
    print("  - Audio Input: 80 mel bins × 3000 time steps")
    print("\nQuantum Enhancements:")
    print(f"  - Quantum Conv1d layers replacing classical Conv1d in audio encoder")
    print(f"  - Number of qubits: {args.n_qubits}")
    print("  - PennyLane backend: default.qubit simulator")
    print("\nASR Decoder:")
    print(f"  - LSTM decoder with {args.num_layers} layers")
    print(f"  - Hidden size: {args.hidden_size}")
    print("  - Character-level prediction")
    print("  - Teacher forcing during training")
    print("="*60)
    
    # Load LibriSpeech dataset
    print("\nLoading LibriSpeech dataset...")
    try:
        train_dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="train.100",
            streaming=False
        )
        
        val_dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean", 
            split="validation",
            streaming=False
        )
        
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
        
    except Exception as e:
        print(f"Error loading LibriSpeech dataset: {e}")
        return
    
    # Build character vocabulary
    print("\nBuilding character vocabulary...")
    char_to_idx = build_character_vocabulary(train_dataset, min_freq=2)
    num_chars = len(char_to_idx)
    
    # Create ASR datasets
    print("\nCreating ASR datasets...")
    train_asr_dataset = LibriSpeechASRDataset(train_dataset, char_to_idx, max_text_length=100)
    val_asr_dataset = LibriSpeechASRDataset(val_dataset, char_to_idx, max_text_length=100)
    test_asr_dataset = LibriSpeechASRDataset(test_dataset, char_to_idx, max_text_length=100)
    
    # Create data loaders
    train_loader = DataLoader(train_asr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_asr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_asr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Initialize quantum Whisper model
    print("\nInitializing Quantum Whisper model...")
    quantum_whisper_model = create_quantum_whisper_tiny(n_qubits=args.n_qubits)
    
    # Freeze non-quantum layers
    quantum_whisper_model = freeze_non_quantum_layers(quantum_whisper_model)
    
    # Create ASR model
    model = QuantumWhisperASR(
        quantum_whisper_model, 
        num_chars=num_chars,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )
    
    # Move model to device
    model = model.to(device)
    
    print(f"ASR model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Train ASR model
    print("\n" + "="*50)
    print("Training ASR Model...")
    print("="*50)
    
    trained_model, train_losses, val_losses, val_cers = train_asr_model(
        model, train_loader, val_loader, epochs=args.epochs, device=device, lr=args.lr
    )
    
    # Evaluate on test set
    print("\n" + "="*50)
    print("Evaluating ASR Model on Test Set...")
    print("="*50)
    
    test_cer = evaluate_asr_model(trained_model, test_loader, device)
    
    print(f"\nTest Character Error Rate: {test_cer:.4f}")
    print(f"Best Validation CER: {min(val_cers):.4f}")
    
    # Plot training results
    print("\nGenerating training plots...")
    plot_training_results(train_losses, val_losses, val_cers, save_path='quantum_whisper_asr_training_results.png')
    
    # Save final model
    torch.save(trained_model.state_dict(), 'quantum_whisper_asr_final.pth')
    
    # Save training history
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_cers': val_cers,
        'test_cer': test_cer,
        'best_val_cer': min(val_cers),
        'num_chars': num_chars,
        'char_to_idx': char_to_idx,
        'n_qubits': args.n_qubits,
        'training_params': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'device': str(device),
            'n_qubits': args.n_qubits,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers
        }
    }
    
    import json
    with open('quantum_whisper_asr_training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print("\nASR training completed!")
    print("Models saved:")
    print("- best_quantum_whisper_asr.pth (best validation CER)")
    print("- quantum_whisper_asr_final.pth (final model)")
    print("- quantum_whisper_asr_training_history.json (training history)")

if __name__ == "__main__":
    main()
