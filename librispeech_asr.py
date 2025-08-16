"""
LibriSpeech ASR Implementation for Quantum Whisper

This module implements proper ASR (Automatic Speech Recognition) for LibriSpeech
using character-level prediction instead of classification.
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

# Add the whisper directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))

class LibriSpeechASRDataset(Dataset):
    """Proper ASR dataset for LibriSpeech with character-level prediction"""
    
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
        audio = item['audio']['array']
        text = item['text'].lower().strip()
        
        # Preprocess audio to mel spectrogram
        audio_features = self.preprocess_audio(audio)
        
        # Convert text to character indices
        text_indices = self.text_to_indices(text)
        
        return audio_features, text_indices
    
    def preprocess_audio(self, audio, sample_rate=16000):
        """Preprocess audio to mel spectrogram"""
        # Ensure audio is float32
        audio = audio.astype(np.float32)
        
        # Resample if necessary
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
                chars.append(self.idx_to_char[idx.item()])
        return ''.join(chars)

class ASRDecoder(nn.Module):
    """ASR Decoder for character-level prediction"""
    
    def __init__(self, input_size, hidden_size, num_chars, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_chars = num_chars
        
        # Audio feature projection
        self.audio_projection = nn.Linear(input_size, hidden_size)
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Character prediction head
        self.char_predictor = nn.Linear(hidden_size, num_chars)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, audio_features, text_indices=None, max_length=100):
        batch_size = audio_features.size(0)
        
        # Project audio features
        audio_projected = self.audio_projection(audio_features)  # [batch, 80, hidden]
        
        # Global average pooling over time dimension
        audio_encoded = torch.mean(audio_projected, dim=1)  # [batch, hidden]
        
        if self.training and text_indices is not None:
            # Teacher forcing during training
            return self._forward_teacher_forcing(audio_encoded, text_indices)
        else:
            # Inference mode
            return self._forward_inference(audio_encoded, max_length)
    
    def _forward_teacher_forcing(self, audio_encoded, text_indices):
        """Forward pass with teacher forcing during training"""
        batch_size = audio_encoded.size(0)
        seq_length = text_indices.size(1)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(audio_encoded.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(audio_encoded.device)
        
        # Use audio encoding as initial input
        current_input = audio_encoded.unsqueeze(1)  # [batch, 1, hidden]
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(seq_length - 1):  # -1 because we predict next character
            # LSTM forward pass
            lstm_out, hidden = self.lstm(current_input, hidden)
            
            # Predict next character
            char_logits = self.char_predictor(lstm_out.squeeze(1))
            outputs.append(char_logits)
            
            # Use ground truth for next input (teacher forcing)
            if t < seq_length - 2:
                next_char_embedding = self._get_char_embedding(text_indices[:, t + 1])
                current_input = next_char_embedding.unsqueeze(1)
        
        return torch.stack(outputs, dim=1)  # [batch, seq_len-1, num_chars]
    
    def _forward_inference(self, audio_encoded, max_length):
        """Forward pass during inference (no teacher forcing)"""
        batch_size = audio_encoded.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(audio_encoded.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(audio_encoded.device)
        
        # Use audio encoding as initial input
        current_input = audio_encoded.unsqueeze(1)  # [batch, 1, hidden]
        
        outputs = []
        hidden = (h0, c0)
        
        for t in range(max_length):
            # LSTM forward pass
            lstm_out, hidden = self.lstm(current_input, hidden)
            
            # Predict next character
            char_logits = self.char_predictor(lstm_out.squeeze(1))
            outputs.append(char_logits)
            
            # Get predicted character
            predicted_char = torch.argmax(char_logits, dim=1)
            
            # Check for end token
            if predicted_char.item() == self.char_to_idx.get('<END>', -1):
                break
            
            # Use predicted character for next input
            next_char_embedding = self._get_char_embedding(predicted_char)
            current_input = next_char_embedding.unsqueeze(1)
        
        return torch.stack(outputs, dim=1)
    
    def _get_char_embedding(self, char_indices):
        """Get character embeddings (simplified - could use learned embeddings)"""
        # For now, use a simple projection
        # In practice, you might want to use learned character embeddings
        return F.one_hot(char_indices, num_classes=self.num_chars).float()

class QuantumWhisperASR(nn.Module):
    """Quantum Whisper model adapted for ASR"""
    
    def __init__(self, quantum_whisper_model, num_chars, hidden_size=384, num_layers=2):
        super().__init__()
        self.quantum_whisper = quantum_whisper_model
        
        # ASR decoder
        self.asr_decoder = ASRDecoder(
            input_size=quantum_whisper_model.dims.n_audio_state,
            hidden_size=hidden_size,
            num_chars=num_chars,
            num_layers=num_layers
        )
        
    def forward(self, mel_spec, text_indices=None, max_length=100):
        # Extract audio features using quantum Whisper encoder
        audio_features = self.quantum_whisper.embed_audio(mel_spec)
        
        # Pass to ASR decoder
        return self.asr_decoder(audio_features, text_indices, max_length)

def build_character_vocabulary(dataset, min_freq=2):
    """Build character vocabulary from LibriSpeech dataset"""
    char_counts = {}
    
    print("Building character vocabulary...")
    for i, item in enumerate(dataset):
        text = item['text'].lower().strip()
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} samples...")
    
    # Filter by minimum frequency
    char_counts = {char: count for char, count in char_counts.items() if count >= min_freq}
    
    # Add special tokens
    special_tokens = ['<PAD>', '<UNK>', '<START>', '<END>']
    char_to_idx = {token: idx for idx, token in enumerate(special_tokens)}
    
    # Add regular characters
    for char in sorted(char_counts.keys()):
        if char not in char_to_idx:
            char_to_idx[char] = len(char_to_idx)
    
    print(f"Character vocabulary size: {len(char_to_idx)}")
    print(f"Special tokens: {special_tokens}")
    print(f"Regular characters: {len(char_to_idx) - len(special_tokens)}")
    
    return char_to_idx

# Import utilities
from utils import calculate_cer

def predictions_to_text(predictions, char_to_idx):
    """Convert model predictions to text"""
    # Get predicted character indices
    char_indices = torch.argmax(predictions, dim=-1)
    
    # Convert to text
    chars = []
    for idx in char_indices:
        if idx == char_to_idx['<PAD>']:
            continue
        if idx == char_to_idx['<END>']:
            break
        if idx != char_to_idx['<START>']:
            # Get character from index
            for char, char_idx in char_to_idx.items():
                if char_idx == idx.item():
                    chars.append(char)
                    break
    
    return ''.join(chars)

def targets_to_text(target_indices, char_to_idx):
    """Convert target indices to text"""
    chars = []
    for idx in target_indices:
        if idx == char_to_idx['<PAD>']:
            continue
        if idx == char_to_idx['<END>']:
            break
        if idx != char_to_idx['<START>']:
            # Get character from index
            for char, char_idx in char_to_idx.items():
                if char_idx == idx.item():
                    chars.append(char)
                    break
    
    return ''.join(chars)

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
