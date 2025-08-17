"""
Evaluate Pretrained Whisper Tiny on Google Speech Commands

This script evaluates the pretrained Whisper Tiny model on the Google Speech Commands dataset.
Each audio sample is fed to the model 10 times; the most frequent predicted word (majority vote)
is taken as the final prediction for that sample.
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
import sys
import os
import argparse
from collections import Counter
from datasets import load_dataset, Audio  # kept in case user reuses parts later

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
        target_length = 3000

        if mel_spec.shape[1] > target_length:
            mel_spec = mel_spec[:, :target_length]
        else:
            # Pad with zeros to reach target length
            pad_length = target_length - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_length)), mode='constant')

        return torch.FloatTensor(mel_spec)


def calculate_wer(predictions, targets):
    """Calculate Word Error Rate using jiwer for string lists"""
    from jiwer import wer
    if not predictions:
        return 0.0
    total_wer = 0.0
    for pred, target in zip(predictions, targets):
        # both pred and target are strings (single words in this task)
        total_wer += wer(target, pred)
    return total_wer / len(predictions)


class WhisperClassifier(nn.Module):
    """Whisper-based classifier for speech commands"""
    def __init__(self, whisper_model, num_classes=35):
        super().__init__()
        self.whisper = whisper_model
        # note: whisper_model.dims.n_audio_state expected
        self.classifier = nn.Linear(whisper_model.dims.n_audio_state, num_classes)

    def to(self, device):
        super().to(device)
        if hasattr(self.whisper, 'to'):
            self.whisper = self.whisper.to(device)
        return self

    def forward(self, mel_spec):
        # mel_spec shape expected: (batch_size, 80, 3000) coming from AudioDataset/DataLoader
        audio_features = self.whisper.embed_audio(mel_spec)
        # audio_features shape assumed: (batch, seq_len, dim) -> mean over seq_len
        if audio_features.dim() == 3:
            pooled_features = torch.mean(audio_features, dim=1)
        elif audio_features.dim() == 2:
            pooled_features = audio_features
        else:
            pooled_features = audio_features.view(audio_features.size(0), -1)

        logits = self.classifier(pooled_features)
        return logits


def evaluate_pretrained_model(model, test_loader, idx_to_label, device='cpu', n_votes=10):
    """Evaluate the pretrained model with majority voting (n_votes per sample)."""
    model = model.to(device)
    model.eval()

    predictions = []  # strings
    targets = []      # strings
    correct = 0
    total = 0

    with torch.no_grad():
        for audio_batch, labels_batch in tqdm(test_loader, desc="Evaluating Pretrained Model"):
            # audio_batch: (batch, 80, 3000)
            audio_batch = audio_batch.to(device)
            labels_batch = labels_batch.to(device)

            batch_size = audio_batch.size(0)

            # process each sample in the batch individually (we need to vote per sample)
            for i in range(batch_size):
                sample = audio_batch[i:i+1]  # keep batch dim
                true_idx = labels_batch[i].item()
                true_label = idx_to_label[true_idx]

                # run n_votes inference passes
                votes = []
                for _ in range(n_votes):
                    outputs = model(sample)  # shape (1, num_classes)
                    _, pred = torch.max(outputs.data, 1)
                    pred_idx = pred.item()
                    pred_label = idx_to_label.get(pred_idx, str(pred_idx))
                    votes.append(pred_label)

                # majority vote (most common label)
                most_common_label = Counter(votes).most_common(1)[0][0]

                predictions.append(most_common_label)
                targets.append(true_label)

                total += 1
                if most_common_label == true_label:
                    correct += 1

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    wer = calculate_wer(predictions, targets)

    return wer, accuracy, predictions, targets


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
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{value:.4f}', ha='center', va='bottom')

    # Class-wise accuracy
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())

    ax2.bar(range(len(classes)), accuracies)
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
    parser = argparse.ArgumentParser(description='Evaluate Pretrained Whisper on Speech Dataset (Google Speech Commands only)')
    parser.add_argument('--dataset', type=str, default='google', help='Dataset to use (only google supported)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation (default: 32)')
    parser.add_argument('--votes', type=int, default=10, help='Number of inference votes per audio (default: 10)')
    args = parser.parse_args()

    # Force google mode
    if args.dataset.lower() != 'google':
        print("Only Google Speech Commands is supported in this modified script. Forcing dataset to 'google'.")
        args.dataset = 'google'

    print("=" * 60)
    print("EVALUATING PRETRAINED WHISPER TINY ON GOOGLE SPEECH COMMANDS (MAJORITY VOTE)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Votes per sample: {args.votes}")

    # Model Information
    print("\n" + "=" * 60)
    print("MODEL INFORMATION")
    print("=" * 60)
    print("Model: Whisper Tiny (Official OpenAI Implementation)")
    print("Source: Hugging Face Transformers Library (attempted)")
    print("Model ID: openai/whisper-tiny")
    print("Architecture (expected): audio encoder 4 layers, 384 hidden size, 6 heads")
    print("=" * 60)

    # Load Google Speech Commands dataset
    print("\nLoading Google Speech Commands dataset...")
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
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)

    print(f"Number of classes: {num_classes}")
    print(f"Classes: {sorted(all_labels)}")

    # Create test dataset wrapper
    test_dataset = AudioDataset(test_dataset, label_to_idx)

    # Create test data loader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load pretrained Whisper model
    print("\nLoading pretrained Whisper Tiny model...")
    try:
        from transformers import WhisperForConditionalGeneration
        hf_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        print("Loaded pretrained model from Hugging Face")

        # Create a wrapper to make it compatible with our WhisperClassifier
        class HuggingFaceWhisperWrapper:
            def __init__(self, model):
                self.model = model
                cfg = model.config
                # read expected dims from config
                self.expected_seq_len = getattr(cfg, 'max_source_positions', 3000)
                self.expected_n_mels = getattr(cfg, 'num_mel_bins', 80)
                self.dims = type('Dims', (), {
                    'n_audio_state': getattr(cfg, 'd_model', 384),
                    'n_mels': self.expected_n_mels,
                    'n_audio_ctx': self.expected_seq_len,
                    'n_audio_head': getattr(cfg, 'encoder_attention_heads', 6),
                    'n_audio_layer': getattr(cfg, 'encoder_layers', 4)
                })()

            def to(self, device):
                self.model = self.model.to(device)
                return self

            def embed_audio(self, mel_spec):
                """
                Ensure mel_spec ends up as (batch, seq_len=self.expected_seq_len, feat=self.expected_n_mels)
                HuggingFace Whisper encoder expects shape (batch, seq_len, feature).
                Accept input in (batch, 80, 3000) or (batch, 3000, 80) and pad/trim time dim to expected_seq_len.
                """
                x = mel_spec

                # convert to torch tensor if not already
                if not isinstance(x, torch.Tensor):
                    x = torch.tensor(x)

                # ensure float dtype
                x = x.float()

                b = x.shape[0]

                # handle the common shapes robustly
                if x.dim() == 3:
                    d1, d2 = x.shape[1], x.shape[2]

                    # Case A: (batch, 80, time)
                    if d1 == self.expected_n_mels:
                        time = d2
                        if time < self.expected_seq_len:
                            pad = torch.zeros((b, d1, self.expected_seq_len - time), dtype=x.dtype, device=x.device)
                            x = torch.cat([x, pad], dim=2)
                        elif time > self.expected_seq_len:
                            x = x[:, :, :self.expected_seq_len]
                        # transpose to (batch, seq_len, feat)
                        x = x.transpose(1, 2).contiguous()

                    # Case B: (batch, time, 80)
                    elif d2 == self.expected_n_mels:
                        time = d1
                        if time < self.expected_seq_len:
                            pad = torch.zeros((b, self.expected_seq_len - time, d2), dtype=x.dtype, device=x.device)
                            x = torch.cat([x, pad], dim=1)
                        elif time > self.expected_seq_len:
                            x = x[:, :self.expected_seq_len, :]

                    # Unexpected shape: attempt to coerce by treating second as time if small
                    else:
                        # try to reshape by making last dim expected_n_mels if possible
                        if d1 == self.expected_seq_len:
                            # already (batch, seq_len, feat_unknown) -> hope feat_unknown==expected_n_mels
                            pass
                        else:
                            # fallback: pad/trunc along last dimension to match expected_seq_len*expected_n_mels then reshape
                            flat = x.view(b, -1)
                            needed = self.expected_seq_len * self.expected_n_mels
                            if flat.shape[1] < needed:
                                pad = torch.zeros((b, needed - flat.shape[1]), dtype=x.dtype, device=x.device)
                                flat = torch.cat([flat, pad], dim=1)
                            elif flat.shape[1] > needed:
                                flat = flat[:, :needed]
                            x = flat.view(b, self.expected_seq_len, self.expected_n_mels)

                else:
                    raise ValueError(f"Unexpected mel_spec dim: {x.dim()}")

                # Move to model device and dtype
                model_device = next(self.model.parameters()).device
                x = x.to(device=model_device, dtype=next(self.model.parameters()).dtype)

                # encoder expects (batch, seq_len, feature)
                encoder_outputs = self.model.model.encoder(x, return_dict=True)
                return encoder_outputs.last_hidden_state

        pretrained_model = HuggingFaceWhisperWrapper(hf_model)

    except Exception as e:
        print(f"Could not load from Hugging Face: {e}")
        print("Falling back to random-initialized local Whisper model")
        dims = get_whisper_tiny_dims()
        pretrained_model = Whisper(dims)

    # Create classifier with pretrained Whisper
    model = WhisperClassifier(pretrained_model, num_classes=num_classes)

    # Move model to device
    model = model.to(device)

    # Debug: Check device placement
    try:
        print(f"Model device: {next(model.parameters()).device}")
    except StopIteration:
        pass

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Evaluate pretrained model with majority voting
    print("\n" + "=" * 50)
    print("Evaluating pretrained model with majority voting...")
    print("=" * 50)

    wer, accuracy, predictions, targets = evaluate_pretrained_model(model, test_loader, idx_to_label, device=device, n_votes=args.votes)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test WER (avg over samples): {wer:.4f}")
    print(f"Test Accuracy (majority-vote): {accuracy:.2f}%")

    # Calculate class-wise accuracy using predictions and targets
    class_correct = {label: 0 for label in all_labels}
    class_total = {label: 0 for label in all_labels}

    for pred, tgt in zip(predictions, targets):
        class_total[tgt] += 1
        if pred == tgt:
            class_correct[tgt] += 1

    class_accuracies = {}
    print("\nClass-wise Accuracy:")
    for label in sorted(all_labels):
        total = class_total.get(label, 0)
        correct = class_correct.get(label, 0)
        acc = (100.0 * correct / total) if total > 0 else 0.0
        class_accuracies[label] = acc
        print(f"{label}: {acc:.2f}% ({total} samples)")

    # Plot results
    print("\nGenerating evaluation plots...")
    plot_evaluation_results(wer, accuracy, class_accuracies)

    # Save results
    results = {
        'dataset': args.dataset,
        'wer': wer,
        'accuracy': accuracy,
        'num_classes': num_classes,
        'label_to_idx': label_to_idx,
        'class_accuracies': class_accuracies
    }

    import json
    output_file = f'{args.dataset}_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation completed!")
    print(f"Results saved to '{output_file}'")
    print("Plots saved to 'pretrained_whisper_evaluation_results.png'")


if __name__ == "__main__":
    main()
