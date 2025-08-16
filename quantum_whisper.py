"""
Quantum Whisper Implementation

This module extends the official OpenAI Whisper implementation by replacing
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
        
    def to(self, device):
        super().to(device)
        # Note: Quantum device doesn't need to be moved, but classical layers do
        return self
        
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

def create_quantum_whisper_tiny(n_qubits: int = 4):
    """Create a quantum version of Whisper Tiny with quantum convolutional layers"""
    dims = get_whisper_tiny_dims()
    return QuantumWhisper(dims, n_qubits)

def freeze_non_quantum_layers(model):
    """Freeze all layers except quantum layers"""
    for name, param in model.named_parameters():
        # Only train quantum layers and classifier
        if ('quantum' in name.lower() or 
            'pre_conv' in name or 
            'post_conv' in name or
            'classifier' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def load_pretrained_whisper(model_path: str = None):
    """Load pretrained Whisper model"""
    if model_path and os.path.exists(model_path):
        # Load from local path
        model = Whisper(get_whisper_tiny_dims())
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded pretrained model from {model_path}")
    else:
        # Load from Hugging Face
        from transformers import WhisperForConditionalGeneration
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        print("Loaded pretrained model from Hugging Face")
    
    return model
