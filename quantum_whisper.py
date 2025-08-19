"""
Quantum Whisper Implementation

This module extends the official OpenAI Whisper Tiny implementation by replacing
classical Conv1d layers with quantum convolutional layers using PennyLane.

MODEL ARCHITECTURE:
- Base Model: Whisper Tiny (Official OpenAI Implementation)
- Model Dimensions: 4 transformer layers, 384 hidden size, 6 attention heads
- Total Parameters: ~39M
- Audio Input: 80 mel bins × 3000 time steps
- Vocabulary: 51,865 tokens

QUANTUM ENHANCEMENTS:
- Replaces Conv1d layers in AudioEncoder with QuantumConv1d
- Quantum circuit: Amplitude embedding + parameterized rotations + CNOT gates
- PennyLane backend: default.qubit simulator
- Configurable number of qubits (default: 4)
- Only quantum layers are trained during fine-tuning

USAGE:
- create_quantum_whisper_tiny(n_qubits=4): Creates quantum Whisper Tiny model
- freeze_non_quantum_layers(): Freezes all non-quantum layers
- load_pretrained_whisper(): Loads pretrained Whisper model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from typing import Optional, Tuple, Dict
import sys
import os

# Add the whisper directory to the path so we can import from it
sys.path.append(os.path.join(os.path.dirname(__file__), 'whisper'))

# Import from the official Whisper implementation
from whisper.model import (
    Whisper, ModelDimensions, AudioEncoder, Conv1d, 
    LayerNorm, Linear, MultiHeadAttention, ResidualAttentionBlock
)

class QuantumConv1d(nn.Module):
    """Quantum 1D Convolution layer to replace classical Conv1d in Whisper"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, n_qubits: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.n_qubits = min(n_qubits, in_channels * kernel_size)
        
        # Classical preprocessing and postprocessing layers
        self.pre_conv = nn.Linear(in_channels * kernel_size, self.n_qubits)
        self.post_conv = nn.Linear(self.n_qubits, out_channels)
        
        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Ensure inputs have the correct length for amplitude embedding
            if len(inputs) < 2**self.n_qubits:
                # Pad with zeros to reach required length
                padded_inputs = torch.cat([inputs, torch.zeros(2**self.n_qubits - len(inputs))])
            else:
                padded_inputs = inputs[:2**self.n_qubits]
            
            # Encode classical data into quantum state
            qml.AmplitudeEmbedding(padded_inputs, wires=range(self.n_qubits), normalize=True)
            
            # Apply parameterized rotations
            for i in range(self.n_qubits):
                qml.Rot(weights[i, 0], weights[i, 1], weights[i, 2], wires=i)
            
            # Apply entangling gates
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Measure in computational basis
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        self.quantum_weights = nn.Parameter(torch.randn(self.n_qubits, 3))
        
    def to(self, device):
        super().to(device)
        # Note: Quantum device doesn't need to be moved, but classical layers do
        return self
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, length = x.shape
        
        # Apply padding if needed
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))
        
        # Calculate output length
        output_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = torch.zeros(batch_size, self.out_channels, output_length, device=x.device)
        
        # Process each kernel window
        for i in range(output_length):
            start_idx = i * self.stride
            end_idx = start_idx + self.kernel_size
            kernel_window = x[:, :, start_idx:end_idx]
            kernel_flat = kernel_window.reshape(batch_size, -1)
            
            # Classical preprocessing
            pre_processed = self.pre_conv(kernel_flat)
            
            # Quantum processing
            quantum_outputs = []
            for j in range(batch_size):
                q_out = self.quantum_circuit(pre_processed[j], self.quantum_weights)
                quantum_outputs.append(torch.stack(q_out))
            
            quantum_outputs = torch.stack(quantum_outputs).float()  # Convert to float32
            
            # Classical postprocessing
            conv_output = self.post_conv(quantum_outputs)
            output[:, :, i] = conv_output
        
        return output

class QuantumAudioEncoder(AudioEncoder):
    """Quantum Audio Encoder that replaces Conv1d layers with QuantumConv1d"""
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_qubits: int = 4):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        
        # Replace the Conv1d layers with QuantumConv1d
        self.conv1 = QuantumConv1d(n_mels, n_state, kernel_size=3, padding=1, n_qubits=n_qubits)
        self.conv2 = QuantumConv1d(n_state, n_state, kernel_size=3, stride=2, padding=1, n_qubits=n_qubits)
        
    def to(self, device):
        super().to(device)
        # Ensure quantum layers are moved to device
        self.conv1 = self.conv1.to(device)
        self.conv2 = self.conv2.to(device)
        return self

class QuantumWhisper(Whisper):
    """Quantum Whisper model that uses quantum convolutional layers"""
    def __init__(self, dims: ModelDimensions, n_qubits: int = 4):
        super().__init__(dims)
        
        # Replace the encoder with quantum version
        self.encoder = QuantumAudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            n_qubits
        )
        
    def to(self, device):
        super().to(device)
        # Ensure quantum encoder is moved to device
        self.encoder = self.encoder.to(device)
        return self

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

def load_official_whisper_tiny():
    """Load the official OpenAI Whisper Tiny model"""
    try:
        # Try to load from local whisper directory first
        import whisper
        model = whisper.load_model("tiny")
        print("✅ Loaded official Whisper Tiny model from local whisper directory")
        return model
    except Exception as e:
        print(f"Warning: Could not load from local directory: {e}")
        try:
            # Fallback to loading from Hugging Face
            from transformers import WhisperForConditionalGeneration
            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
            print("✅ Loaded official Whisper Tiny model from Hugging Face")
            return model
        except Exception as e2:
            print(f"Error loading model: {e2}")
            # Create model with official dimensions as last resort
            print("⚠️  Creating model with official Whisper Tiny dimensions")
            dims = get_whisper_tiny_dims()
            model = Whisper(dims)
            return model

def create_whisper_model_from_scratch(vocab_size=51865, n_mels=80, n_audio_ctx=1500, 
                                     use_quantum=False, n_qubits=4, hidden_size=None, 
                                     num_layers=None, num_heads=None):
    """
    Create a Whisper model from scratch (either classical or quantum).
    
    Args:
        vocab_size: Size of the vocabulary
        n_mels: Number of mel spectrogram features
        n_audio_ctx: Audio context length
        use_quantum: Whether to use quantum layers
        n_qubits: Number of qubits for quantum layers (if use_quantum=True)
        hidden_size: Custom hidden size (overrides default 384)
        num_layers: Custom number of layers (overrides default 4)
        num_heads: Custom number of attention heads (overrides default 6)
        
    Returns:
        Whisper model (classical or quantum)
    """
    # Use custom dimensions if provided, otherwise use standard Whisper Tiny dimensions
    audio_state = hidden_size if hidden_size is not None else 384
    audio_layer = num_layers if num_layers is not None else 4
    audio_head = num_heads if num_heads is not None else 6
    text_state = hidden_size if hidden_size is not None else 384
    text_layer = num_layers if num_layers is not None else 4
    text_head = num_heads if num_heads is not None else 6
    
    # The n_audio_ctx should be the output length after convolutions
    # Since conv2 has stride=2, output length = input_length // 2
    # So if we want input length of 1500, we need n_audio_ctx = 750
    audio_ctx_output = n_audio_ctx // 2
    
    dims = ModelDimensions(
        n_mels=n_mels,
        n_audio_ctx=audio_ctx_output,  # This is the output length after conv layers
        n_audio_state=audio_state,
        n_audio_head=audio_head,
        n_audio_layer=audio_layer,
        n_vocab=vocab_size,
        n_text_ctx=448,
        n_text_state=text_state,
        n_text_head=text_head,
        n_text_layer=text_layer,
    )
    
    if use_quantum:
        print("🔬 Creating quantum Whisper model from scratch...")
        model = QuantumWhisper(dims, n_qubits)
    else:
        print("🔨 Creating classical Whisper model from scratch...")
        model = Whisper(dims)
    
    return model

def create_quantum_whisper_from_official(official_model, n_qubits: int = 4):
    """Create quantum Whisper by replacing conv layers in official pretrained model"""
    # Get the official dimensions
    if hasattr(official_model, 'dims'):
        dims = official_model.dims
    else:
        # If using Hugging Face model, use standard dimensions
        dims = get_whisper_tiny_dims()
    
    # Create quantum model with same dimensions
    quantum_model = QuantumWhisper(dims, n_qubits)
    
    # Copy ALL pretrained weights from official model to quantum model
    if hasattr(official_model, 'state_dict'):
        official_state = official_model.state_dict()
        quantum_state = quantum_model.state_dict()
        
        # Copy weights for all layers (including the new quantum conv layers will be random)
        for key in official_state:
            if key in quantum_state:
                quantum_state[key] = official_state[key]
        
        # Load the copied weights
        quantum_model.load_state_dict(quantum_state, strict=False)
        print("✅ Copied ALL pretrained weights from official Whisper Tiny model")
        print("✅ Only quantum conv layers remain randomly initialized")
    
    return quantum_model

def create_quantum_whisper_tiny(n_qubits: int = 4):
    """Create a quantum version of Whisper Tiny with quantum convolutional layers"""
    # First load the official model
    official_model = load_official_whisper_tiny()
    
    # Get the official dimensions
    if hasattr(official_model, 'dims'):
        dims = official_model.dims
    else:
        # If using Hugging Face model, use standard dimensions
        dims = get_whisper_tiny_dims()
    
    # Create quantum model with same dimensions
    quantum_model = QuantumWhisper(dims, n_qubits)
    
    # Copy weights from official model to quantum model (except quantum layers)
    if hasattr(official_model, 'state_dict'):
        official_state = official_model.state_dict()
        quantum_state = quantum_model.state_dict()
        
        # Copy weights for non-quantum layers
        for key in official_state:
            if key in quantum_state and 'conv1' not in key and 'conv2' not in key:
                quantum_state[key] = official_state[key]
        
        # Load the copied weights
        quantum_model.load_state_dict(quantum_state, strict=False)
        print("✅ Copied weights from official Whisper Tiny model to quantum model")
    
    return quantum_model

def freeze_non_quantum_layers(model):
    """Freeze all layers except quantum convolutional layers"""
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # Only train quantum conv layers (conv1, conv2) and ASR head
        if ('conv1' in name or 'conv2' in name or 'asr_head' in name):
            param.requires_grad = True
            trainable_count += 1
            print(f"🔄 Training layer: {name}")
        else:
            param.requires_grad = False
            frozen_count += 1
            print(f"❄️  Frozen layer: {name}")
    
    print(f"\n📊 Layer freezing summary:")
    print(f"   Frozen layers: {frozen_count}")
    print(f"   Trainable layers: {trainable_count}")
    print(f"   Only quantum conv layers and ASR head will be trained")
    
    return model

def load_pretrained_whisper(model_path: str = None):
    """Load pretrained Whisper model"""
    if model_path and os.path.exists(model_path):
        # Load from local path
        model = Whisper(get_whisper_tiny_dims())
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded pretrained model from {model_path}")
    else:
        # Load official model
        model = load_official_whisper_tiny()
    
    return model
