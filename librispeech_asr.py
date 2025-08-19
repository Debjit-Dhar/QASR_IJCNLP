"""
LibriSpeech ASR Implementation for Quantum Whisper

This module implements proper ASR (Automatic Speech Recognition) for LibriSpeech
using the EXACT SAME approach as the official notebook. It uses character-level prediction
and proper Whisper audio preprocessing.

Key Features (exactly like the notebook):
- Uses proper Whisper audio preprocessing (pad_or_trim, log_mel_spectrogram)
- Implements character-level prediction for ASR
- Uses the full LibriSpeech dataset for training and validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import from utils
from utils import calculate_cer, calculate_wer

# Add the whisper directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))

# Import Whisper for audio preprocessing (like the notebook)
try:
    import whisper
    from whisper.audio import pad_or_trim, log_mel_spectrogram
    WHISPER_AVAILABLE = True
    print("✅ Whisper audio preprocessing available")
except ImportError:
    WHISPER_AVAILABLE = False
    print("⚠️  Whisper not available, using fallback preprocessing")

class LibriSpeechASRDataset(Dataset):
    """Proper ASR dataset for LibriSpeech with character-level prediction (like the notebook)"""
    
    def __init__(self, dataset, char_to_idx, max_text_length=100):
        self.dataset = dataset
        self.char_to_idx = char_to_idx
        self.max_text_length = max_text_length
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Get audio and text
        if hasattr(item, '__getitem__') and len(item) >= 3:
            # torchaudio format (like the notebook)
            audio, sample_rate, text = item[0], item[1], item[2]
            assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"
            
            # Process audio exactly like the notebook
            if WHISPER_AVAILABLE:
                audio = whisper.pad_or_trim(audio.flatten())
                audio_features = whisper.log_mel_spectrogram(audio)
            else:
                audio_features = self.preprocess_audio_fallback(audio)
                
        else:
            # Hugging Face format
            audio = item['audio']['array']
            text = item['text'].lower().strip()
            
            # Use Whisper preprocessing if available
            if WHISPER_AVAILABLE:
                audio = whisper.pad_or_trim(audio)
                audio_features = whisper.log_mel_spectrogram(audio)
            else:
                audio_features = self.preprocess_audio_fallback(audio)
        
        # Convert text to character indices
        text_indices = self.text_to_indices(text)
        
        return audio_features, text_indices
    
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
    
    def text_to_indices(self, text):
        """Convert text to character indices with padding"""
        # Add special tokens
        text = '<START>' + text + '<END>'
        
        # Convert to indices
        indices = [self.char_to_idx.get(char, self.char_to_idx['<UNK>']) for char in text]
        
        # Pad or truncate to max length
        if len(indices) > self.max_text_length:
            indices = indices[:self.max_text_length]
        else:
            # Pad with <PAD> token
            indices.extend([self.char_to_idx['<PAD>']] * (self.max_text_length - len(indices)))
        
        return torch.LongTensor(indices)
    
    def indices_to_text(self, indices):
        """Convert character indices back to text"""
        chars = []
        for idx in indices:
            if idx == self.char_to_idx['<PAD>']:
                continue
            if idx == self.char_to_idx['<END>']:
                break
            if idx != self.char_to_idx['<START>']:
                chars.append(self.idx_to_char.get(idx, '<UNK>'))
        
        return ''.join(chars)

class QuantumWhisperASR(nn.Module):
    """Quantum Whisper ASR model that extends the base Whisper model"""
    
    def __init__(self, quantum_whisper_model, num_chars, hidden_size=384, num_layers=2):
        super(QuantumWhisperASR, self).__init__()
        
        self.quantum_whisper = quantum_whisper_model
        
        # ASR head for character prediction
        self.asr_head = nn.Sequential(
            nn.Linear(quantum_whisper_model.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            *[nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_size, num_chars)
        )
        
        # Character embedding layer
        self.char_embedding = nn.Embedding(num_chars, hidden_size)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_size))
        
    def forward(self, audio_features, text_indices=None):
        # Get audio embeddings from quantum Whisper
        audio_embeddings = self.quantum_whisper.encoder(audio_features)
        
        # If text indices provided, use them for training
        if text_indices is not None:
            # Character embeddings
            char_embeddings = self.char_embedding(text_indices)
            
            # Add positional encoding
            seq_len = char_embeddings.size(1)
            pos_enc = self.pos_encoding[:, :seq_len, :]
            char_embeddings = char_embeddings + pos_enc
            
            # Combine audio and text embeddings
            combined_embeddings = torch.cat([audio_embeddings, char_embeddings], dim=1)
            
            # Pass through ASR head
            outputs = self.asr_head(combined_embeddings)
            
            return outputs
        else:
            # Inference mode - generate text from audio
            # This would implement the full text generation pipeline
            # For now, return audio embeddings
            return audio_embeddings

def build_character_vocabulary(texts):
    """Build character vocabulary from text data"""
    # Collect all unique characters
    all_chars = set()
    for text in texts:
        all_chars.update(text.lower())
    
    # Add special tokens
    special_tokens = ['<PAD>', 'UNK', '<START>', '<END>']
    all_chars.update(special_tokens)
    
    # Create character to index mapping
    char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
    
    # Ensure special tokens have specific indices
    char_to_idx['<PAD>'] = 0
    char_to_idx['<UNK>'] = 1
    char_to_idx['<START>'] = 2
    char_to_idx['<END>'] = 3
    
    # Reorder other characters
    other_chars = [char for char in sorted(all_chars) if char not in special_tokens]
    for idx, char in enumerate(other_chars):
        char_to_idx[char] = idx + 4
    
    num_chars = len(char_to_idx)
    
    print(f"Built character vocabulary with {num_chars} characters")
    print(f"Special tokens: {special_tokens}")
    print(f"Sample characters: {list(other_chars[:10])}")
    
    return char_to_idx, num_chars

# CER calculation function is now imported from utils.py

# WER calculation function is now imported from utils.py

# Levenshtein distance functions are now available in utils.py

def create_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
    """Create a DataLoader for the dataset"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )

def validate_dataset(dataset, num_samples=5):
    """Validate the dataset by showing sample data"""
    print(f"Dataset validation - showing {num_samples} samples:")
    print("-" * 60)
    
    for i in range(min(num_samples, len(dataset))):
        audio_features, text_indices = dataset[i]
        print(f"Sample {i+1}:")
        print(f"  Audio features shape: {audio_features.shape}")
        print(f"  Text indices shape: {text_indices.shape}")
        print(f"  Text indices: {text_indices[:10].tolist()}...")
        
        # Convert indices back to text
        if hasattr(dataset, 'indices_to_text'):
            text = dataset.indices_to_text(text_indices)
            print(f"  Decoded text: '{text}'")
        print()
    
    print(f"Total samples: {len(dataset)}")
    print("-" * 60)
