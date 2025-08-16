#!/usr/bin/env python3
"""
Evaluate Quantum Whisper ASR Model on LibriSpeech

This script evaluates a trained quantum Whisper ASR model on the LibriSpeech test set
using proper ASR metrics like Character Error Rate (CER).
"""

import torch
import torch.nn as nn
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
    calculate_cer,
    predictions_to_text,
    targets_to_text
)

# Import quantum Whisper
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
from quantum_whisper import create_quantum_whisper_tiny, freeze_non_quantum_layers

def load_trained_asr_model(model_path, n_qubits=4, hidden_size=384, num_layers=2):
    """Load a trained ASR model"""
    # Initialize quantum Whisper model
    quantum_whisper_model = create_quantum_whisper_tiny(n_qubits=n_qubits)
    quantum_whisper_model = freeze_non_quantum_layers(quantum_whisper_model)
    
    # Load character vocabulary from training history
    import json
    history_path = model_path.replace('.pth', '_training_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        char_to_idx = history['char_to_idx']
        num_chars = history['num_chars']
        print(f"Loaded character vocabulary with {num_chars} characters")
    else:
        print("Warning: Training history not found. Using default vocabulary size.")
        # Default vocabulary size (will be overridden when loading model weights)
        num_chars = 100
        char_to_idx = {}
    
    # Create ASR model
    model = QuantumWhisperASR(
        quantum_whisper_model, 
        num_chars=num_chars,
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    return model, char_to_idx

def evaluate_asr_model(model, test_loader, device='cpu', char_to_idx=None):
    """Evaluate the trained ASR model"""
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    sample_predictions = []
    
    with torch.no_grad():
        for i, (audio, text_indices) in enumerate(tqdm(test_loader, desc="Evaluating ASR Model")):
            audio = audio.to(device)
            text_indices = text_indices.to(device)
            
            # Forward pass
            outputs = model(audio, text_indices)
            
            all_predictions.append(outputs.cpu())
            all_targets.append(text_indices.cpu())
            
            # Store some sample predictions for analysis
            if i < 5:  # Store first 5 samples
                for j in range(min(3, audio.size(0))):  # Store first 3 from each batch
                    if len(sample_predictions) < 10:  # Limit to 10 total samples
                        pred_text = predictions_to_text(outputs[j], char_to_idx)
                        target_text = targets_to_text(text_indices[j], char_to_idx)
                        sample_predictions.append({
                            'predicted': pred_text,
                            'target': target_text,
                            'sample_id': f"Sample_{i}_{j}"
                        })
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate CER
    test_cer = calculate_cer(all_predictions, all_targets, char_to_idx)
    
    return test_cer, sample_predictions

def analyze_predictions(sample_predictions):
    """Analyze sample predictions"""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS ANALYSIS")
    print("="*60)
    
    for i, sample in enumerate(sample_predictions):
        print(f"\nSample {i+1}:")
        print(f"Target:     '{sample['target']}'")
        print(f"Predicted:  '{sample['predicted']}'")
        
        # Calculate sample-specific CER
        if len(sample['target']) > 0:
            sample_cer = levenshtein_distance(sample['predicted'], sample['target']) / len(sample['target'])
            print(f"Sample CER: {sample_cer:.4f}")
        else:
            print("Sample CER: N/A (empty target)")
        
        print("-" * 40)

def levenshtein_distance(s1, s2):
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def plot_cer_distribution(sample_predictions):
    """Plot CER distribution for sample predictions"""
    if not sample_predictions:
        return
    
    cers = []
    for sample in sample_predictions:
        if len(sample['target']) > 0:
            cer = levenshtein_distance(sample['predicted'], sample['target']) / len(sample['target'])
            cers.append(cer)
    
    if not cers:
        return
    
    plt.figure(figsize=(10, 6))
    plt.hist(cers, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Character Error Rate (CER)')
    plt.ylabel('Frequency')
    plt.title('Sample CER Distribution')
    plt.grid(True, alpha=0.3)
    plt.axvline(np.mean(cers), color='red', linestyle='--', label=f'Mean CER: {np.mean(cers):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('asr_sample_cer_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Quantum Whisper ASR Model on LibriSpeech')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained ASR model (.pth file)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation (default: 16)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda, default: auto)')
    parser.add_argument('--n_qubits', type=int, default=4, help='Number of qubits for quantum layers (default: 4)')
    parser.add_argument('--hidden_size', type=int, default=384, help='Hidden size for ASR decoder (default: 384)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers (default: 2)')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of test samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATING QUANTUM WHISPER ASR MODEL ON LIBRISPEECH")
    print("="*60)
    print(f"Model path: {args.model_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of qubits: {args.n_qubits}")
    print(f"ASR Decoder hidden size: {args.hidden_size}")
    print(f"ASR Decoder LSTM layers: {args.num_layers}")
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Load LibriSpeech test dataset
    print("\nLoading LibriSpeech test dataset...")
    try:
        test_dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
            streaming=False
        )
        
        # Limit samples if specified
        if args.max_samples:
            test_dataset = test_dataset.select(range(min(args.max_samples, len(test_dataset))))
        
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error loading LibriSpeech dataset: {e}")
        return
    
    # Load trained model
    print("\nLoading trained ASR model...")
    try:
        model, char_to_idx = load_trained_asr_model(
            args.model_path, 
            n_qubits=args.n_qubits,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create test dataset
    print("\nCreating test dataset...")
    test_asr_dataset = LibriSpeechASRDataset(test_dataset, char_to_idx, max_text_length=100)
    test_loader = DataLoader(test_asr_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Evaluate model
    print("\n" + "="*50)
    print("Evaluating ASR Model...")
    print("="*50)
    
    test_cer, sample_predictions = evaluate_asr_model(model, test_loader, device, char_to_idx)
    
    # Print results
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Character Error Rate (CER): {test_cer:.4f}")
    print(f"Test Character Error Rate (CER): {test_cer * 100:.2f}%")
    
    # Analyze sample predictions
    if sample_predictions:
        analyze_predictions(sample_predictions)
        
        # Plot CER distribution
        print("\nGenerating CER distribution plot...")
        plot_cer_distribution(sample_predictions)
    
    # Save results
    results = {
        'test_cer': test_cer,
        'test_cer_percentage': test_cer * 100,
        'num_test_samples': len(test_dataset),
        'model_path': args.model_path,
        'sample_predictions': sample_predictions,
        'evaluation_params': {
            'batch_size': args.batch_size,
            'device': str(device),
            'n_qubits': args.n_qubits,
            'hidden_size': args.hidden_size,
            'num_layers': args.num_layers,
            'max_samples': args.max_samples
        }
    }
    
    import json
    results_file = 'quantum_whisper_asr_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {results_file}")
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
