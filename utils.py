"""
Utility functions for Quantum Whisper project

This module contains common functions used across multiple training and evaluation scripts
to reduce code duplication and improve maintainability.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
import jiwer
import editdistance

def calculate_cer(predictions, targets):
    """Calculate Character Error Rate using editdistance library"""
    if not predictions or not targets:
        return 1.0
    
    total_cer = 0
    total_chars = 0
    
    for pred, target in zip(predictions, targets):
        if len(target) > 0:
            # Use editdistance library for efficient Levenshtein distance
            distance = editdistance.eval(pred.lower(), target.lower())
            cer = distance / len(target)
            total_cer += cer
            total_chars += len(target)
    
    return total_cer / len(predictions) if total_chars > 0 else 1.0

def calculate_wer(predictions, targets):
    """Calculate Word Error Rate using jiwer library"""
    if not predictions or not targets:
        return 1.0
    
    try:
        # Use jiwer library for robust WER calculation
        print(f"Using jiwer library for WER calculation")
        wer = jiwer.wer(targets, predictions)
        return wer
    except Exception as e:
        print(f"Warning: jiwer calculation failed, falling back to custom implementation: {e}")
        # Fallback to custom implementation if jiwer fails
        return _calculate_wer_fallback(predictions, targets)

def _calculate_wer_fallback(predictions, targets):
    """Fallback WER calculation if jiwer fails"""
    print(f"Fallback WER calculation if jiwer fails")
    total_wer = 0
    total_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.lower().split()
        target_words = target.lower().split()
        
        if len(target_words) > 0:
            distance = editdistance.eval(pred_words, target_words)
            wer = distance / len(target_words)
            total_wer += wer
            total_words += len(target_words)
    
    return total_wer / len(predictions) if total_words > 0 else 1.0

# Keep the old functions for backward compatibility but mark as deprecated
def levenshtein_distance_chars(s1, s2):
    """Calculate Levenshtein distance between character strings (DEPRECATED - use editdistance)"""
    import warnings
    warnings.warn("levenshtein_distance_chars is deprecated. Use editdistance.eval() instead.", DeprecationWarning)
    return editdistance.eval(s1, s2)

def levenshtein_distance_words(s1, s2):
    """Calculate Levenshtein distance between word lists (DEPRECATED - use editdistance)"""
    import warnings
    warnings.warn("levenshtein_distance_words is deprecated. Use editdistance.eval() instead.", DeprecationWarning)
    return editdistance.eval(s1, s2)

def preprocess_audio_for_whisper(audio, sample_rate=16000):
    """Preprocess audio for Whisper model"""
    # Convert to numpy array if it's a list
    if isinstance(audio, list):
        audio = np.array(audio)
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Resample if necessary
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    # Use official Whisper preprocessing
    from whisper.audio import pad_or_trim, log_mel_spectrogram
    audio = pad_or_trim(audio)
    mel_spec = log_mel_spectrogram(audio)
    
    return mel_spec

def analyze_predictions(sample_predictions, include_metrics=True):
    """Analyze sample predictions with optional metrics calculation"""
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS ANALYSIS")
    print("="*60)
    
    for i, sample in enumerate(sample_predictions):
        print(f"\nSample {i+1}:")
        print(f"Target:     '{sample['target']}'")
        print(f"Predicted:  '{sample['predicted']}'")
        
        # Show normalized text if available
        if 'normalized_target' in sample and 'normalized_predicted' in sample:
            print(f"Normalized Target:     '{sample['normalized_target']}'")
            print(f"Normalized Predicted:  '{sample['normalized_predicted']}'")
        

        
        if include_metrics:
            # Use normalized text for metrics if available, otherwise fall back to original
            if 'normalized_target' in sample and 'normalized_predicted' in sample:
                # Calculate metrics using normalized text
                if len(sample['normalized_target']) > 0:
                    sample_cer = levenshtein_distance_chars(sample['normalized_predicted'], sample['normalized_target']) / len(sample['normalized_target'])
                    print(f"Sample CER (normalized): {sample_cer:.4f}")
                
                target_words = sample['normalized_target'].split()
                pred_words = sample['normalized_predicted'].split()
                if len(target_words) > 0:
                    sample_wer = levenshtein_distance_words(pred_words, target_words) / len(target_words)
                    print(f"Sample WER (normalized): {sample_wer:.4f}")
            else:
                # Fallback to original text metrics
                if len(sample['target']) > 0:
                    sample_cer = levenshtein_distance_chars(sample['predicted'].upper(), sample['target'].upper()) / len(sample['target'])
                    print(f"Sample CER: {sample_cer:.4f}")
                
                target_words = sample['target'].upper().split()
                pred_words = sample['predicted'].upper().split()
                if len(target_words) > 0:
                    sample_wer = levenshtein_distance_words(pred_words, target_words) / len(target_words)
                    print(f"Sample WER: {sample_wer:.4f}")
        
        print("-" * 40)

def plot_cer_distribution(sample_predictions, save_path='cer_distribution.png'):
    """Plot CER distribution for sample predictions"""
    if not sample_predictions:
        return
    
    cers = []
    for sample in sample_predictions:
        # Use normalized text if available, otherwise fall back to original
        if 'normalized_target' in sample and 'normalized_predicted' in sample:
            if len(sample['normalized_target']) > 0:
                cer = levenshtein_distance_chars(sample['normalized_predicted'], sample['normalized_target']) / len(sample['normalized_target'])
                cers.append(cer)
        else:
            if len(sample['target']) > 0:
                cer = levenshtein_distance_chars(sample['predicted'].upper(), sample['target'].upper()) / len(sample['target'])
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_metrics_distribution(sample_predictions, save_path='metrics_distribution.png'):
    """Plot CER and WER distribution for sample predictions"""
    if not sample_predictions:
        return
    
    cers = []
    wers = []
    
    for sample in sample_predictions:
        # Use normalized text if available, otherwise fall back to original
        if 'normalized_target' in sample and 'normalized_predicted' in sample:
            if len(sample['normalized_target']) > 0:
                # Calculate CER using normalized text
                cer = levenshtein_distance_chars(sample['normalized_predicted'], sample['normalized_target']) / len(sample['normalized_target'])
                cers.append(cer)
                
                # Calculate WER using normalized text
                target_words = sample['normalized_target'].split()
                pred_words = sample['normalized_predicted'].split()
                if len(target_words) > 0:
                    wer = levenshtein_distance_words(pred_words, target_words) / len(target_words)
                    wers.append(wer)
        else:
            # Fallback to original text
            if len(sample['target']) > 0:
                # Calculate CER
                cer = levenshtein_distance_chars(sample['predicted'].upper(), sample['target'].upper()) / len(sample['target'])
                cers.append(cer)
                
                # Calculate WER
                target_words = sample['target'].upper().split()
                pred_words = sample['predicted'].upper().split()
                if len(target_words) > 0:
                    wer = levenshtein_distance_words(pred_words, target_words) / len(target_words)
                    wers.append(wer)
    
    if not cers or not wers:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot CER distribution
    ax1.hist(cers, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Character Error Rate (CER)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Sample CER Distribution')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(cers), color='red', linestyle='--', label=f'Mean CER: {np.mean(cers):.4f}')
    ax1.legend()
    
    # Plot WER distribution
    ax2.hist(wers, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.set_xlabel('Word Error Rate (WER)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sample WER Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(wers), color='red', linestyle='--', label=f'Mean WER: {np.mean(wers):.4f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_results(train_losses, val_losses, val_cers=None, val_wers=None, save_path='training_results.png'):
    """Plot training results with flexible metrics"""
    if val_cers is not None and val_wers is not None:
        # Plot with both CER and WER
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot CER
        ax2.plot(val_cers, label='Validation CER', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Character Error Rate')
        ax2.set_title('Validation Character Error Rate')
        ax2.legend()
        ax2.grid(True)
        
        # Plot WER
        ax3.plot(val_wers, label='Validation WER', color='orange')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Word Error Rate')
        ax3.set_title('Validation Word Error Rate')
        ax3.legend()
        ax3.grid(True)
        
        # Combined metrics
        ax4.plot(val_cers, label='CER', color='green')
        ax4.plot(val_wers, label='WER', color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Validation Error Rates Comparison')
        ax4.legend()
        ax4.grid(True)
        
    elif val_cers is not None:
        # Plot with CER only
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot CER
        ax2.plot(val_cers, label='Validation CER', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Character Error Rate')
        ax2.set_title('Validation Character Error Rate')
        ax2.legend()
        ax2.grid(True)
        
    else:
        # Plot losses only
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        ax1.plot(train_losses, label='Train Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_training_history(training_history, filename):
    """Save training history to JSON file"""
    with open(filename, 'w') as f:
        json.dump(training_history, f, indent=2)

def save_evaluation_results(results, filename):
    """Save evaluation results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def get_device(device_arg='auto'):
    """Get device (CPU/CUDA) based on argument"""
    if device_arg == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_arg)

def print_model_info(model, model_type="Model"):
    """Print model information and parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{model_type} parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

def print_training_header(title, **kwargs):
    """Print formatted training header with parameters"""
    print("="*60)
    print(title)
    print("="*60)
    for key, value in kwargs.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print()

def print_epoch_results(epoch, total_epochs, train_loss, val_loss, val_cer=None, val_wer=None, lr=None):
    """Print formatted epoch results"""
    print(f'\nEpoch {epoch}/{total_epochs}:')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Val Loss: {val_loss:.4f}')
    
    if val_cer is not None:
        print(f'Val CER: {val_cer:.4f}')
    if val_wer is not None:
        print(f'Val WER: {val_wer:.4f}')
    if lr is not None:
        print(f'Learning Rate: {lr:.6f}')
    
    print('-' * 60)
