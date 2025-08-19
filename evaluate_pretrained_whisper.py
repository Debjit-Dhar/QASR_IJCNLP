"""
Evaluate Official Whisper Model on LibriSpeech Dataset
=====================================================

This script evaluates the official OpenAI Whisper model on the LibriSpeech ASR dataset
using the EXACT SAME approach as the official notebook. It implements proper ASR evaluation
with real transcription and accurate WER calculation.

Key Features (exactly like the notebook):
- Uses torchaudio for LibriSpeech dataset loading
- Uses model.decode() for actual transcription
- Applies text normalization using EnglishTextNormalizer
- Calculates real WER using jiwer library
- Proper audio preprocessing with Whisper functions
- Expected WER: ~4.26% (like the notebook reports)

Author: QASR Research Team
Date: 2024
"""

import os
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
from tqdm import tqdm
import jiwer
from whisper.normalizers import EnglishTextNormalizer
import argparse
import json
import warnings
import sys
warnings.filterwarnings('ignore')

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils import calculate_cer_pure

# Set device like the notebook
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    EXACTLY like the official notebook.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)

def evaluate_whisper_model(model, loader, options):
    """Evaluate the Whisper model exactly like the notebook"""
    hypotheses = []
    references = []

    print("🎯 Running Whisper inference exactly like the notebook...")
    for mels, texts in tqdm(loader, desc="Transcribing Audio"):
        results = model.decode(mels, options)
        hypotheses.extend([result.text for result in results])
        references.extend(texts)
    
    return hypotheses, references

def main():
    parser = argparse.ArgumentParser(description='Evaluate Official Whisper Model on LibriSpeech')
    parser.add_argument('--model_size', type=str, default='base.en', 
                       choices=['tiny', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large'],
                       help='Whisper model size to evaluate')
    parser.add_argument('--split', type=str, default='test-clean', 
                       choices=['test-clean', 'test-other', 'dev-clean', 'dev-other'],
                       help='LibriSpeech split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of test samples to evaluate')
    
    args = parser.parse_args()
    
    print("="*60)
    print(f"EVALUATING OFFICIAL WHISPER {args.model_size.upper()} MODEL ON LIBRISPEECH")
    print("="*60)
    print(f"Model size: {args.model_size}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print(f"Device: {DEVICE}")
    
    # Load LibriSpeech dataset exactly like the notebook
    print(f"\nLoading LibriSpeech {args.split} dataset using torchaudio (like the notebook)...")
    try:
        dataset = LibriSpeech(args.split, device=DEVICE)
        
        # Limit samples if specified
        if args.max_samples:
            dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
        
        print(f"Dataset samples: {len(dataset)}")
        
    except Exception as e:
        print(f"Error loading LibriSpeech dataset: {e}")
        return
    
    # Create data loader exactly like the notebook
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    
    # Load official Whisper model exactly like the notebook
    print(f"\nLoading official Whisper {args.model_size} model...")
    try:
        model = whisper.load_model(args.model_size)
        print(f"✅ Successfully loaded Whisper {args.model_size} model")
        print(
            f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
        )
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        return
    
    # Set up decoding options exactly like the notebook
    print("\nSetting up decoding options (like the notebook)...")
    options = whisper.DecodingOptions(language="en", without_timestamps=True)
    print("✅ Using Whisper decoding options: English, no timestamps")
    
    # Evaluate model exactly like the notebook
    print("\n" + "="*50)
    print("Evaluating Whisper Model (exactly like the notebook)...")
    print("="*50)
    
    hypotheses, references = evaluate_whisper_model(model, loader, options)
    
    # Create DataFrame exactly like the notebook
    print("\nCreating results DataFrame (like the notebook)...")
    data = pd.DataFrame(dict(hypothesis=hypotheses, reference=references))
    
    # Apply text normalization exactly like the notebook
    print("\nApplying text normalization (like the notebook)...")
    normalizer = EnglishTextNormalizer()
    data["hypothesis_clean"] = [normalizer(text) for text in data["hypothesis"]]
    data["reference_clean"] = [normalizer(text) for text in data["reference"]]
    
    # Calculate WER exactly like the notebook
    print("\nCalculating Word Error Rate (like the notebook)...")
    wer = jiwer.wer(list(data["reference_clean"]), list(data["hypothesis_clean"]))
    
    # Calculate CER (Character Error Rate)
    print("\nCalculating Character Error Rate (CER)...")
    
    cer = calculate_cer_pure(list(data["hypothesis_clean"]), list(data["reference_clean"]))
    
    print(f"📊 Evaluation Results:")
    print(f"   Word Error Rate (WER): {wer * 100:.2f}%")
    print(f"   Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"   Total samples: {len(hypotheses)}")
    
    # Show some sample predictions
    print(f"\n📝 Sample Predictions (first 5):")
    print("-" * 60)
    for i in range(min(5, len(data))):
        print(f"Sample {i+1}:")
        print(f"  Reference: '{data.iloc[i]['reference']}'")
        print(f"  Hypothesis: '{data.iloc[i]['hypothesis']}'")
        print(f"  Reference (clean): '{data.iloc[i]['reference_clean']}'")
        print(f"  Hypothesis (clean): '{data.iloc[i]['hypothesis_clean']}'")
        print()
    
    # Save evaluation results
    evaluation_results = {
        'test_wer': wer,
        'test_cer': cer,
        'model_type': f'official_whisper_{args.model_size}',
        'evaluation_params': {
            'model_size': args.model_size,
            'split': args.split,
            'batch_size': args.batch_size,
            'max_samples': args.max_samples,
            'device': DEVICE
        },
        'sample_predictions': data.head(10).to_dict('records'),
        'dataset_info': {
            'split': args.split,
            'total_samples': len(dataset)
        },
        'note': 'Evaluation using EXACT SAME approach as official LibriSpeech notebook'
    }
    
    results_path = f'whisper_{args.model_size}_evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation results saved to: {results_path}")
    print("\n✅ Evaluation completed successfully!")
    print(f"Expected WER for {args.model_size}: ~4.26% (like the notebook)")
    print(f"Actual WER: {wer * 100:.2f}%")
    print(f"Actual CER: {cer * 100:.2f}%")

if __name__ == "__main__":
    main()
