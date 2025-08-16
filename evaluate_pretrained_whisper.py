"""
Evaluate Pretrained Whisper Tiny on Google Speech Commands or LibriSpeech Dataset

This script evaluates the pretrained Whisper Tiny model on either the Google Speech Commands 
dataset or the LibriSpeech ASR dataset.
"""

import torch
import torch.nn as nn
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
        
        # Convert text to label index
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

class WhisperClassifier(nn.Module):
    """Whisper-based classifier for speech commands"""
    def __init__(self, whisper_model, num_classes=35):
        super().__init__()
        self.whisper = whisper_model
        self.classifier = nn.Linear(whisper_model.dims.n_audio_state, num_classes)
        
    def to(self, device):
        super().to(device)
        if hasattr(self.whisper, 'to'):
            self.whisper = self.whisper.to(device)
        return self
        
    def forward(self, mel_spec):
        # Extract audio features using Whisper encoder
        audio_features = self.whisper.embed_audio(mel_spec)
        
        # Global average pooling
        pooled_features = torch.mean(audio_features, dim=1)
        
        # Classification
        logits = self.classifier(pooled_features)
        return logits

def evaluate_pretrained_model(model, test_loader, device='cpu'):
    """Evaluate the pretrained model"""
    model = model.to(device)
    model.eval()
    
    predictions = []
    targets = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for audio, labels in tqdm(test_loader, desc="Evaluating Pretrained Model"):
            audio = audio.to(device)
            labels = labels.to(device)
            
            # Debug: Check tensor devices
            if str(audio.device) != str(device):
                print(f"Warning: Audio tensor on {audio.device}, expected {device}")
            if str(labels.device) != str(device):
                print(f"Warning: Labels tensor on {labels.device}, expected {device}")
            
            outputs = model(audio)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            predictions.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    wer = calculate_wer(predictions, targets)
    
    return wer, accuracy

def plot_evaluation_results(wer, accuracy, class_accuracies):
    """Plot evaluation results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Overall metrics
    metrics = ['WER', 'Accuracy']
    values = [wer, accuracy]
    colors = ['red', 'green']
    
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_title('Overall Performance Metrics')
    ax1.set_ylabel('Value')
    ax1.set_ylim(0, max(values) * 1.1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    # Class-wise accuracy
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    ax2.bar(range(len(classes)), accuracies, color='skyblue')
    ax2.set_title('Class-wise Accuracy')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('pretrained_whisper_evaluation_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def get_whisper_tiny_dims():
    """Get Whisper Tiny model dimensions"""
    return ModelDimensions(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=384,
        n_audio_head=6,
        n_audio_layer=4,
        n_vocab=51865,
        n_text_ctx=448,
        n_text_state=384,
        n_text_head=6,
        n_text_layer=4,
    )

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate Pretrained Whisper on Speech Dataset')
    parser.add_argument('--dataset', type=str, default='google', help='Dataset to use (google/librispeech, default: google)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (default: 32)')
    args = parser.parse_args()
    
    print("="*60)
    print("EVALUATING PRETRAINED WHISPER TINY ON SPEECH DATASET")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    
    # Model Information
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print("Model: Whisper Tiny (Official OpenAI Implementation)")
    print("Source: Hugging Face Transformers Library")
    print("Model ID: openai/whisper-tiny")
    print("Architecture:")
    print("  - Audio Encoder: 4 transformer layers, 384 hidden size, 6 attention heads")
    print("  - Text Decoder: 4 transformer layers, 384 hidden size, 6 attention heads")
    print("  - Vocabulary: 51,865 tokens")
    print("  - Parameters: ~39M")
    print("  - Mel Spectrogram: 80 mel bins × 3000 time steps")
    print("  - Input Format: Audio features (batch_size, 80, 3000)")
    print("="*60)
    
    # Load dataset based on user choice
    if args.dataset.lower() == 'librispeech':
        print("\nLoading LibriSpeech ASR dataset from Hugging Face...")
        
        try:
            # Load test split - use full dataset
            test_dataset = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="test",
                streaming=False
            )
            
            print("LibriSpeech dataset loaded successfully!")
            print(f"Test samples: {len(test_dataset)}")
            
            # Build vocabulary from ALL training data
            print("\nBuilding vocabulary from ALL LibriSpeech training data...")
            train_dataset = load_dataset(
                "openslr/librispeech_asr",
                "clean",
                split="train.100",
                streaming=False
            )
            
            all_texts = set()
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
            
            # Create test dataset with full data
            test_dataset = LibriSpeechDataset(test_dataset, label_to_idx)
            
        except Exception as e:
            print(f"Error loading LibriSpeech dataset: {e}")
            print("Falling back to Google Speech Commands dataset...")
            args.dataset = 'google'
    
    if args.dataset.lower() == 'google':
        print("\nLoading Google Speech Commands dataset...")
        
        # Create test dataset
        test_dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root="./data", 
            url="speech_commands_v0.02",
            subset="testing",
            download=True
        )
        
        print("Dataset loaded successfully!")
        print(f"Test samples: {len(test_dataset)}")
        
        # Get unique labels for classification
        all_labels = set()
        for i in range(len(test_dataset)):
            _, _, label, _, _ = test_dataset[i]
            all_labels.add(label)
        
        label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
        num_classes = len(label_to_idx)
        
        print(f"Number of classes: {num_classes}")
        print(f"Classes: {sorted(all_labels)}")
        
        # Create test dataset
        test_dataset = AudioDataset(test_dataset, label_to_idx)
    
    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load pretrained Whisper model
    print("\nLoading pretrained Whisper Tiny model...")
    try:
        from transformers import WhisperForConditionalGeneration
        pretrained_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        print("Loaded pretrained model from Hugging Face")
        
        # Create a wrapper to make it compatible with our WhisperClassifier
        class HuggingFaceWhisperWrapper:
            def __init__(self, model):
                self.model = model
                # Get dimensions from the model config
                self.dims = type('Dims', (), {
                    'n_audio_state': model.config.d_model,
                    'n_mels': model.config.num_mel_bins,
                    'n_audio_ctx': model.config.max_source_positions,
                    'n_audio_head': model.config.encoder_attention_heads,
                    'n_audio_layer': model.config.encoder_layers
                })()
            
            def to(self, device):
                self.model = self.model.to(device)
                return self
            
            def embed_audio(self, mel_spec):
                # Use the encoder part of the model
                # mel_spec should already be in the correct format (batch_size, 80, 3000)
                # HuggingFace Whisper expects input_features of shape (batch_size, feature_size, sequence_length)
                encoder_outputs = self.model.model.encoder(mel_spec)
                return encoder_outputs.last_hidden_state
        
        pretrained_model = HuggingFaceWhisperWrapper(pretrained_model)
        
    except Exception as e:
        print(f"Could not load from Hugging Face: {e}")
        print("Using random initialization")
        dims = get_whisper_tiny_dims()
        pretrained_model = Whisper(dims)
    
    # Create classifier with pretrained Whisper
    model = WhisperClassifier(pretrained_model, num_classes=num_classes)
    
    # Move model to device
    model = model.to(device)
    
    # Debug: Check device placement
    print(f"Model device: {next(model.parameters()).device}")
    if hasattr(model.whisper, 'model'):
        print(f"Whisper model device: {next(model.whisper.model.parameters()).device}")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate pretrained model
    print("\n" + "="*50)
    print("Evaluating pretrained model...")
    print("="*50)
    
    wer, accuracy = evaluate_pretrained_model(model, test_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Test WER: {wer:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate class-wise accuracy
    model.eval()
    
    if args.dataset.lower() == 'librispeech':
        # For LibriSpeech, we'll show accuracy for a sample of classes
        class_correct = {}
        class_total = {}
        
        with torch.no_grad():
            for audio, labels in test_loader:
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                
                for i, label in enumerate(labels):
                    # Get text from label index
                    label_text = list(label_to_idx.keys())[label.item()]
                    if label_text not in class_total:
                        class_total[label_text] = 0
                        class_correct[label_text] = 0
                    
                    class_total[label_text] += 1
                    if predicted[i] == label:
                        class_correct[label_text] += 1
        
        # Show top 20 classes by frequency
        sorted_classes = sorted(class_total.items(), key=lambda x: x[1], reverse=True)[:20]
        
        print("\nTop 20 Classes by Frequency:")
        for label, total in sorted_classes:
            if total > 0:
                acc = 100 * class_correct[label] / total
                print(f"{label}: {acc:.2f}% ({total} samples)")
    
    else:
        # For Google Speech Commands, show all classes
        class_correct = {label: 0 for label in all_labels}
        class_total = {label: 0 for label in all_labels}
        
        with torch.no_grad():
            for audio, labels in test_loader:
                audio = audio.to(device)
                labels = labels.to(device)
                
                outputs = model(audio)
                _, predicted = torch.max(outputs.data, 1)
                
                for i, label in enumerate(labels):
                    label_name = list(label_to_idx.keys())[list(label_to_idx.values()).index(label.item())]
                    class_total[label_name] += 1
                    if predicted[i] == label:
                        class_correct[label_name] += 1
        
        class_accuracies = {}
        for label in all_labels:
            if class_total[label] > 0:
                class_accuracies[label] = 100 * class_correct[label] / class_total[label]
            else:
                class_accuracies[label] = 0.0
        
        print("\nClass-wise Accuracy:")
        for label, acc in sorted(class_accuracies.items()):
            print(f"{label}: {acc:.2f}%")
    
    # Plot results
    print("\nGenerating evaluation plots...")
    if args.dataset.lower() == 'librispeech':
        # For LibriSpeech, create a simplified plot
        plt.figure(figsize=(10, 6))
        plt.bar(['WER', 'Accuracy'], [wer, accuracy], color=['red', 'green'])
        plt.title(f'LibriSpeech Evaluation Results')
        plt.ylabel('Value')
        plt.ylim(0, max(wer, accuracy) * 1.1)
        plt.savefig('librispeech_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    else:
        # For Google Speech Commands, use the existing plotting function
        plot_evaluation_results(wer, accuracy, class_accuracies)
    
    # Save results
    results = {
        'dataset': args.dataset,
        'wer': wer,
        'accuracy': accuracy,
        'num_classes': num_classes,
        'label_to_idx': label_to_idx
    }
    
    if args.dataset.lower() == 'google':
        results['class_accuracies'] = class_accuracies
    
    import json
    output_file = f'{args.dataset}_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation completed!")
    print(f"Results saved to '{output_file}'")
    if args.dataset.lower() == 'google':
        print("Plots saved to 'pretrained_whisper_evaluation_results.png'")
    else:
        print("Plots saved to 'librispeech_evaluation_results.png'")

if __name__ == "__main__":
    main()
