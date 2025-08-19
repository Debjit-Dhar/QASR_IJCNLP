#!/usr/bin/env python3
"""
Train Whisper Model From Scratch on LibriSpeech
===============================================

This script trains a Whisper model with random initialization (from scratch)
on the full LibriSpeech dataset without any pretrained weights.

Key features:
- Random initialization of Whisper architecture
- Full LibriSpeech dataset training
- Proper Whisper tokenization and audio preprocessing
- Efficient batching and training loop

Author: From Scratch Training Version
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import argparse
import json
from datetime import datetime
import warnings
from tqdm import tqdm
import whisper
from datasets import load_dataset, Audio
import librosa

# Import project utilities
from utils import calculate_wer, calculate_cer

warnings.filterwarnings('ignore')

def create_whisper_from_scratch(model_size="tiny"):
    """Create a Whisper model with random initialization"""
    print(f"Creating Whisper {model_size} model from scratch (random weights)...")
    
    try:
        # Load the model architecture but with random weights
        model = whisper.load_model(model_size, download_root=None)
        
        # Reinitialize all weights randomly
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if hasattr(m, 'in_proj_weight') and m.in_proj_weight is not None:
                    torch.nn.init.xavier_uniform_(m.in_proj_weight)
                if hasattr(m, 'out_proj') and hasattr(m.out_proj, 'weight') and m.out_proj.weight is not None:
                    torch.nn.init.xavier_uniform_(m.out_proj.weight)
        
        # Apply random initialization
        model.apply(init_weights)
        
        # Fix the alignment_heads buffer to be dense instead of sparse for MPS compatibility
        if hasattr(model, 'alignment_heads'):
            # Convert sparse alignment_heads to dense
            alignment_heads = model.alignment_heads.to_dense()
            model.register_buffer("alignment_heads", alignment_heads, persistent=False)
        
        print("✅ Model weights randomly initialized")
        return model
        
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        raise RuntimeError(f"Failed to create Whisper {model_size} model: {e}")

class LibriSpeechWhisperDataset(Dataset):
    """LibriSpeech dataset for Whisper training from scratch"""
    
    def __init__(self, dataset, tokenizer, sample_rate=16000, max_length=30):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_length = max_length
        
        # Add special tokens for training from scratch
        self.sot_token = tokenizer.sot
        self.eot_token = tokenizer.eot
        self.no_timestamps_token = tokenizer.no_timestamps
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio and text
        audio_array = item['audio']['array']
        text = item['text'].strip().upper()  # Whisper expects uppercase
        
        # Ensure audio is at correct sample rate
        if item['audio']['sampling_rate'] != self.sample_rate:
            audio_array = librosa.resample(
                audio_array, 
                orig_sr=item['audio']['sampling_rate'], 
                target_sr=self.sample_rate
            )
        
        # Pad or trim audio to max_length seconds
        max_samples = self.max_length * self.sample_rate
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        else:
            padding = max_samples - len(audio_array)
            audio_array = np.pad(audio_array, (0, padding), mode='constant')
        
        # Convert to mel spectrogram using Whisper's preprocessing
        audio_tensor = torch.from_numpy(audio_array).float()
        mel_spec = whisper.log_mel_spectrogram(audio_tensor)
        
        # Tokenize text with proper special tokens for training from scratch
        # Format: <|startoftranscript|><|notimestamps|>TEXT<|endoftext|>
        text_tokens = [self.sot_token, self.no_timestamps_token] + \
                     self.tokenizer.encode(text) + [self.eot_token]
        
        # Limit text sequence length to Whisper's context window (448 for tiny model)
        # This prevents the indexSelectLargeIndex error
        max_text_length = 448
        if len(text_tokens) > max_text_length:
            text_tokens = text_tokens[:max_text_length-1] + [self.eot_token]
        
        return {
            'input_features': mel_spec,
            'labels': text_tokens,
            'text': text
        }

def collate_fn(batch):
    """Custom collate function for efficient batching"""
    input_features = torch.stack([item['input_features'] for item in batch])
    
    # Pad labels to same length, but ensure we don't exceed Whisper's context window
    max_label_length = max(len(item['labels']) for item in batch)
    max_context_length = 448  # Whisper tiny model context size
    
    # Ensure we don't exceed the model's context window
    if max_label_length > max_context_length:
        max_label_length = max_context_length
        print(f"⚠️  Warning: Truncating sequences to {max_context_length} tokens (model context limit)")
    
    labels = []
    
    for item in batch:
        label = item['labels']
        if len(label) < max_label_length:
            # Pad with -100 (ignore index for loss calculation)
            label = label + [-100] * (max_label_length - len(label))
        else:
            # Truncate if too long
            label = label[:max_label_length]
        labels.append(label)
    
    labels = torch.tensor(labels, dtype=torch.long)
    texts = [item['text'] for item in batch]
    
    return {
        'input_features': input_features,
        'labels': labels,
        'texts': texts
    }

def train_epoch(model, dataloader, optimizer, device, tokenizer, epoch, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_features = batch['input_features'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        try:
            # Original Whisper model expects (mel, tokens) and returns logits directly
            # The model internally: encoder(mel) -> decoder(tokens, audio_features) -> logits
            logits = model(input_features, labels)
            
            # The model returns logits, so we can calculate loss directly
            # Shift labels for next-token prediction (teacher forcing)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for loss calculation
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            
            # Calculate cross-entropy loss, ignoring padding tokens (-100)
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Step scheduler for warmup and learning rate decay
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        avg_loss = total_loss / num_batches
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # Log every 1000 batches
        if batch_idx > 0 and batch_idx % 1000 == 0:
            print(f"Batch {batch_idx}: Loss = {avg_loss:.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0

def validate_epoch(model, dataloader, device, tokenizer, epoch):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_references = []
    
    progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
    
    with torch.no_grad():
        for batch in progress_bar:
            input_features = batch['input_features'].to(device)
            labels = batch['labels'].to(device)
            texts = batch['texts']
            
            # Forward pass for loss
            try:
                # Original Whisper model expects (mel, tokens) and returns logits directly
                logits = model(input_features, labels)
                
                # The model returns logits, so we can calculate loss directly
                # Shift labels for next-token prediction (teacher forcing)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                # Flatten for loss calculation
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                # Calculate cross-entropy loss, ignoring padding tokens (-100)
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits, shift_labels)
                
                total_loss += loss.item()
                num_batches += 1
            except Exception as e:
                print(f"Error in validation forward pass: {e}")
                continue
            
            # Generate predictions for metrics (sample a few for speed)
            if len(all_predictions) < 100:  # Limit predictions for faster validation
                try:
                    # For original Whisper model, we can use the logits we already computed
                    # No need to call the model again since we have logits from loss calculation
                    with torch.no_grad():
                        # Greedy decoding: take argmax at each position
                        predicted_ids = torch.argmax(logits, dim=-1)
                        
                        # Decode predictions
                        for pred_ids, ref_text in zip(predicted_ids, texts):
                            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                            all_predictions.append(pred_text)
                            all_references.append(ref_text)
                except Exception as e:
                    print(f"Error in generation: {e}")
                    # If generation fails, use dummy predictions
                    for ref_text in texts:
                        all_predictions.append("")  # Empty prediction
                        all_references.append(ref_text)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    # Calculate metrics
    if all_predictions and all_references:
        # Use project utility functions
        wer = calculate_wer(all_predictions, all_references)
        cer = calculate_cer(all_predictions, all_references)
        metrics = {"wer": wer, "cer": cer}
    else:
        metrics = {"wer": 1.0, "cer": 1.0}
    
    # Return sample predictions for inspection
    sample_preds = all_predictions[:5] if all_predictions else [""] * 5
    sample_refs = all_references[:5] if all_references else [""] * 5
    
    return avg_loss, metrics, sample_preds, sample_refs

def main():
    parser = argparse.ArgumentParser(description='Train Whisper From Scratch on LibriSpeech')
    parser.add_argument('--model_size', type=str, default='tiny', 
                       choices=['tiny', 'base', 'small', 'medium'],
                       help='Whisper model size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--max_audio_length', type=int, default=30, help='Max audio length in seconds')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--save_every', type=int, default=5, help='Save model every N epochs')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of warmup epochs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TRAINING WHISPER FROM SCRATCH ON LIBRISPEECH CLEAN DATASET")
    print("="*70)
    print(f"Model size: {args.model_size}")
    print(f"Training epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max audio length: {args.max_audio_length}s")
    print(f"Warmup epochs: {args.warmup_epochs}")
    
    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"✅ Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("✅ Using Apple Silicon GPU (MPS)")
        else:
            device = torch.device('cpu')
            print("⚠️  Using CPU (will be very slow)")
    else:
        device = torch.device(args.device)
    
    # Create model from scratch
    model = create_whisper_from_scratch(args.model_size)
    model = model.to(device)
    
    # Get tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load CLEAN LibriSpeech dataset only
    print("\nLoading LibriSpeech CLEAN dataset...")
    try:
        # Load only clean training data (100h + 360h = 460h total)
        train_100_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
        train_360_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.360") 
        
        # Combine clean training data
        from datasets import concatenate_datasets
        full_train_dataset = concatenate_datasets([train_100_dataset, train_360_dataset])
        
        # Load clean validation and test data
        val_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation")
        test_dataset = load_dataset("openslr/librispeech_asr", "clean", split="test")
        
        print(f"✅ Clean training samples: {len(full_train_dataset):,}")
        print(f"   - train.100: {len(train_100_dataset):,} samples")
        print(f"   - train.360: {len(train_360_dataset):,} samples")
        print(f"✅ Clean validation samples: {len(val_dataset):,}")
        print(f"✅ Clean test samples: {len(test_dataset):,}")
        print(f"✅ Total clean training hours: ~{len(full_train_dataset) * 6 / 3600:.1f} hours")
        
    except Exception as e:
        print(f"❌ Error loading clean datasets, falling back to clean-100 only: {e}")
        # Fallback to clean-100 only if full clean dataset fails
        full_train_dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100")
        val_dataset = load_dataset("openslr/librispeech_asr", "clean", split="validation")
        print(f"Fallback - Train samples: {len(full_train_dataset):,}")
        print(f"Fallback - Validation samples: {len(val_dataset):,}")
    
    # Create datasets
    train_whisper_dataset = LibriSpeechWhisperDataset(
        full_train_dataset, tokenizer, max_length=args.max_audio_length
    )
    val_whisper_dataset = LibriSpeechWhisperDataset(
        val_dataset, tokenizer, max_length=args.max_audio_length
    )
    
    # Create data loaders with more workers for faster loading
    num_workers = min(8, os.cpu_count())
    train_loader = DataLoader(
        train_whisper_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_whisper_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=num_workers//2,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=True
    )
    
    print(f"Data loaders created with {num_workers} workers")
    
    # Set up optimizer and scheduler with warmup
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-6
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_wers = []
    best_wer = float('inf')
    
    print("\n" + "="*50)
    print("STARTING TRAINING FROM SCRATCH...")
    print("="*50)
    print(f"Total training steps: {total_steps:,}")
    print(f"Warmup steps: {warmup_steps:,}")
    print(f"Training samples per epoch: {len(full_train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    for epoch in range(args.epochs):
        print(f"\n{'='*20} EPOCH {epoch+1}/{args.epochs} {'='*20}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device, tokenizer, epoch+1, scheduler)
        train_losses.append(train_loss)
        
        # Memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Validation (every epoch for better monitoring)
        val_loss, metrics, pred_samples, ref_samples = validate_epoch(
            model, val_loader, device, tokenizer, epoch+1
        )
        val_losses.append(val_loss)
        val_wers.append(metrics['wer'])
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val WER: {metrics['wer']:.4f}")
        print(f"  Val CER: {metrics['cer']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Print sample predictions
        print(f"\nSample Predictions (Epoch {epoch+1}):")
        for i in range(min(2, len(pred_samples))):
            print(f"  Sample {i+1}:")
            print(f"    Pred: '{pred_samples[i]}'")
            print(f"    Ref:  '{ref_samples[i]}'")
        
        # Save best model
        if metrics['wer'] < best_wer:
            best_wer = metrics['wer']
            save_path = f'best_whisper_{args.model_size}_from_scratch.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'wer': best_wer,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, save_path)
            print(f"🎯 New best WER: {best_wer:.4f} - Model saved to {save_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f'checkpoint_whisper_{args.model_size}_epoch_{epoch+1}.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_wers': val_wers
            }, checkpoint_path)
            print(f"📁 Checkpoint saved: {checkpoint_path}")
        
        # Memory cleanup after validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save final results
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_wers': val_wers,
        'best_wer': best_wer,
        'model_size': args.model_size,
        'total_epochs': args.epochs,
        'total_samples': len(full_train_dataset),
        'training_params': vars(args),
        'final_metrics': {
            'best_wer': best_wer,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_val_loss': val_losses[-1] if val_losses else None,
            'final_val_wer': val_wers[-1] if val_wers else None
        }
    }
    
    with open(f'whisper_{args.model_size}_from_scratch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot results if matplotlib available
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        ax1.plot(train_losses, label='Train Loss')
        if val_losses:
            # Plot validation loss for all epochs (now runs every epoch)
            ax1.plot(range(len(val_losses)), val_losses, label='Val Loss', marker='o')
        ax1.set_title(f'Whisper {args.model_size} Training Progress (From Scratch)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if val_wers:
            ax2.plot(range(len(val_wers)), val_wers, label='Val WER', color='red', marker='o')
            ax2.set_title('Validation Word Error Rate')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('WER')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Add CER plot if available
        if val_losses and len(val_losses) == len(val_wers):
            # Extract CER from metrics (assuming it's stored)
            val_cers = [1.0] * len(val_wers)  # Default CER values
            ax3.plot(range(len(val_cers)), val_cers, label='Val CER', color='green', marker='s')
            ax3.set_title('Validation Character Error Rate')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('CER')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'whisper_{args.model_size}_from_scratch_training.png', dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: whisper_{args.model_size}_from_scratch_training.png")
        
    except ImportError:
        print("matplotlib not available, skipping plots")
    
    print("\n" + "="*60)
    print("CLEAN DATASET TRAINING FROM SCRATCH COMPLETED!")
    print("="*60)
    print(f"Best WER achieved: {best_wer:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}" if train_losses else "No training completed")
    print(f"Final validation loss: {val_losses[-1]:.4f}" if val_losses else "No validation completed")
    print(f"Total clean training samples: {len(full_train_dataset):,}")
    print("Files saved:")
    print(f"- best_whisper_{args.model_size}_from_scratch.pth")
    print(f"- whisper_{args.model_size}_from_scratch_results.json")
    print(f"- Various checkpoints: checkpoint_whisper_{args.model_size}_epoch_*.pth")
    print(f"- Training plots: whisper_{args.model_size}_from_scratch_training.png")

if __name__ == "__main__":
    main()