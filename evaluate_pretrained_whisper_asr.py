#!/usr/bin/env python3
"""
Evaluate Pretrained Whisper ASR Model on LibriSpeech

This script evaluates a pretrained Whisper model (classical, not quantum) on the LibriSpeech
dataset using both Character Error Rate (CER) and Word Error Rate (WER) metrics.
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

# Import Whisper
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))
import whisper

def load_pretrained_whisper_model(model_size="tiny"):
    """Load a pretrained Whisper model"""
    print(f"Loading pretrained Whisper {model_size} model...")
    model = whisper.load_model(model_size)
    print(f"Model loaded successfully!")
    return model

def preprocess_audio_for_whisper_transcribe(audio, target_sample_rate=16000):
    """Preprocess audio for Whisper transcribe method - expects raw audio samples"""
    # Convert to numpy array if it's a list
    if isinstance(audio, list):
        audio = np.array(audio)
    
    # Ensure audio is float32
    audio = audio.astype(np.float32)
    
    # Flatten if multi-dimensional
    if len(audio.shape) > 1:
        audio = audio.flatten()
    
    # Normalize audio to [-1, 1] range if needed
    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))
    
    # Whisper transcribe expects raw audio samples at 16kHz
    # If the audio is already at the right sample rate, return as is
    return audio

def normalize_text_for_evaluation(text):
    """Normalize text for evaluation by converting to UPPERCASE and removing punctuation"""
    import re
    # Convert to UPPERCASE and remove punctuation
    text = re.sub(r'[^\w\s]', '', text.upper())
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def evaluate_whisper_asr(model, test_dataset, device='cpu', max_samples=None):
    """Evaluate the pretrained Whisper model on LibriSpeech"""
    model = model.to(device)
    
    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    all_predictions = []
    all_targets = []
    all_normalized_predictions = []
    all_normalized_targets = []
    sample_predictions = []
    
    print(f"Evaluating on {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for i, item in enumerate(tqdm(test_dataset, desc="Evaluating Whisper ASR")):
            # Get audio and text
            audio = item['audio']['array']
            target_text = item['text'].strip()
            
            # Preprocess audio for Whisper transcribe (raw audio, not mel spectrogram)
            processed_audio = preprocess_audio_for_whisper_transcribe(audio)
            
            # Use proper Whisper transcription
            try:
                # Use the transcribe method with the preprocessed audio array
                result = model.transcribe(processed_audio, verbose=False)
                predicted_text = result["text"].strip()
            except Exception as e:
                print(f"Warning: Transcription failed for sample {i}: {e}")
                predicted_text = "[TRANSCRIPTION_ERROR]"
            
            # Ensure we have a valid string
            if not predicted_text:
                predicted_text = "[NO_PREDICTION]"
            
            # Store original texts
            all_predictions.append(predicted_text)
            all_targets.append(target_text)
            
            # Store normalized texts for evaluation
            normalized_pred = normalize_text_for_evaluation(predicted_text)
            normalized_target = normalize_text_for_evaluation(target_text)
            all_normalized_predictions.append(normalized_pred)
            all_normalized_targets.append(normalized_target)
            
            # Store sample predictions for analysis
            if i < 10:  # Store first 10 samples
                sample_predictions.append({
                    'predicted': predicted_text,
                    'target': target_text,
                    'normalized_predicted': normalized_pred,
                    'normalized_target': normalized_target,
                    'sample_id': f"Sample_{i+1}"
                })
    
    # Calculate metrics using normalized text
    test_cer = calculate_cer(all_normalized_predictions, all_normalized_targets)
    test_wer = calculate_wer(all_normalized_predictions, all_normalized_targets)
    
    return test_cer, test_wer, sample_predictions

# Import utilities
from utils import calculate_wer, calculate_cer, levenshtein_distance_words, levenshtein_distance_chars
from utils import analyze_predictions, plot_metrics_distribution

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Pretrained Whisper ASR Model on LibriSpeech')
    parser.add_argument('--model_size', type=str, default='tiny', 
                       choices=['tiny', 'base', 'small', 'medium', 'large'], 
                       help='Whisper model size (default: tiny)')
    parser.add_argument('--device', type=str, default='auto', 
                       help='Device to use (auto/cpu/cuda, default: auto)')
    parser.add_argument('--max_samples', type=int, default=None, 
                       help='Maximum number of test samples to evaluate (default: all)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATING PRETRAINED WHISPER ASR MODEL ON LIBRISPEECH")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    
    # Load LibriSpeech test dataset
    print("\nLoading LibriSpeech test dataset...")
    try:
        test_dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
            streaming=False
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
    except Exception as e:
        print(f"Error loading LibriSpeech dataset: {e}")
        return
    
    # Load pretrained Whisper model
    try:
        model = load_pretrained_whisper_model(args.model_size)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    
    # Evaluate model
    print("\n" + "="*50)
    print("Evaluating Pretrained Whisper ASR...")
    print("="*50)
    
    test_cer, test_wer, sample_predictions = evaluate_whisper_asr(
        model, test_dataset, device, args.max_samples
    )
    
    # Print results
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test Character Error Rate (CER): {test_cer:.4f}")
    print(f"Test Character Error Rate (CER): {test_cer * 100:.2f}%")
    print(f"Test Word Error Rate (WER): {test_wer:.4f}")
    print(f"Test Word Error Rate (WER): {test_wer * 100:.2f}%")
    
    # Analyze sample predictions
    if sample_predictions:
        analyze_predictions(sample_predictions)
        
        # Plot metrics distribution
        print("\nGenerating metrics distribution plots...")
        plot_metrics_distribution(sample_predictions)
    
    # Save results
    results = {
        'model_size': args.model_size,
        'test_cer': test_cer,
        'test_cer_percentage': test_cer * 100,
        'test_wer': test_wer,
        'test_wer_percentage': test_wer * 100,
        'num_test_samples': len(test_dataset) if not args.max_samples else min(args.max_samples, len(test_dataset)),
        'sample_predictions': sample_predictions,
        'evaluation_params': {
            'model_size': args.model_size,
            'device': str(device),
            'max_samples': args.max_samples
        }
    }
    
    import json
    results_file = f'pretrained_whisper_{args.model_size}_asr_evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {results_file}")
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
