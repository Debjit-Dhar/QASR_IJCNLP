# Quantum Whisper Architecture Explanation

## Overview

The quantum Whisper implementation replaces specific classical layers with quantum equivalents while keeping the overall Whisper architecture intact.

## What Parts of Whisper Are Replaced with Quantum?

### 1. **Conv1d Layers in Audio Encoder**

**Location**: `whisper/model.py` → `AudioEncoder` class
**Replaced**: The first two convolutional layers in the audio encoder

```python
# Original Whisper AudioEncoder
class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        # ... rest of the encoder
```

**Quantum Replacement**:
```python
# Quantum AudioEncoder
class QuantumAudioEncoder(AudioEncoder):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, n_qubits: int = 4):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)
        
        # Replace the Conv1d layers with QuantumConv1d
        self.conv1 = QuantumConv1d(n_mels, n_state, kernel_size=3, padding=1, n_qubits=n_qubits)
        self.conv2 = QuantumConv1d(n_state, n_state, kernel_size=3, stride=2, padding=1, n_qubits=n_qubits)
```

### 2. **QuantumConv1d Implementation**

The `QuantumConv1d` layer replaces classical 1D convolution with a hybrid quantum-classical approach:

```python
class QuantumConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, n_qubits: int = 4):
        super().__init__()
        # Classical preprocessing and postprocessing layers
        self.pre_conv = nn.Linear(in_channels * kernel_size, self.n_qubits)
        self.post_conv = nn.Linear(self.n_qubits, out_channels)
        
        # Quantum device and circuit
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.quantum_circuit = quantum_circuit
        self.quantum_weights = nn.Parameter(torch.randn(self.n_qubits, 3))
```

## Architecture Details

### **What Stays Classical (Frozen)**:
1. **Attention Layers**: All MultiHeadAttention layers remain classical and frozen
2. **Layer Normalization**: All LayerNorm layers remain classical and frozen  
3. **Linear Layers**: Most linear layers remain classical and frozen
4. **Text Decoder**: The entire text decoder remains classical and frozen

### **What Gets Quantum Enhancement**:
1. **Audio Encoder Conv1d Layers**: Only the first two conv layers are quantum
2. **Quantum Circuit**: Uses PennyLane with amplitude embedding and parameterized rotations

### **What Gets Trained**:
1. **Quantum Layers**: `QuantumConv1d` layers (pre_conv, quantum_weights, post_conv)
2. **Classifier**: The final classification head
3. **Everything Else**: Frozen (attention, normalization, etc.)

## Training Strategy

### **Frozen Layers**:
```python
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
```

### **Trainable Parameters**:
- Quantum circuit weights (`quantum_weights`)
- Pre-processing linear layer (`pre_conv`)
- Post-processing linear layer (`post_conv`) 
- Classification head (`classifier`)

## Quantum Circuit Design

The quantum circuit uses:
1. **Amplitude Embedding**: Encodes classical data into quantum state
2. **Parameterized Rotations**: Learnable quantum parameters
3. **Entangling Gates**: CNOT gates for quantum correlations
4. **Measurement**: Pauli-Z measurements for output

```python
@qml.qnode(self.dev, interface="torch")
def quantum_circuit(inputs, weights):
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
```

## Usage Examples

### **Training with Custom Parameters**:
```bash
# Train quantum whisper with custom parameters
python train_quantum_whisper.py --epochs 50 --lr 1e-3 --n_qubits 6 --batch_size 8

# Train normal whisper first
python train_whisper_from_scratch.py --epochs 100 --lr 1e-4 --batch_size 32

# Then train quantum whisper using the best normal model
python train_quantum_whisper.py --epochs 50 --pretrained_path best_whisper_val_acc.pth
```

### **Parameter Options**:
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-3 for quantum, 1e-4 for normal)
- `--batch_size`: Batch size (default: 16 for quantum, 32 for normal)
- `--n_qubits`: Number of qubits for quantum layers (default: 4)
- `--pretrained_path`: Path to pretrained normal whisper model
- `--device`: Device to use (auto/cpu/cuda)

## Key Differences from Normal Whisper

| Aspect | Normal Whisper | Quantum Whisper |
|--------|----------------|-----------------|
| **Conv1d Layers** | Classical | Quantum (first 2 layers only) |
| **Trainable Parameters** | All layers | Only quantum layers + classifier |
| **Pretrained Layers** | None (trained from scratch) | Most layers frozen, quantum layers trained |
| **Computational Cost** | Lower | Higher (quantum simulation) |
| **Batch Size** | 32 | 16 (due to quantum overhead) |
| **Learning Rate** | 1e-4 | 1e-3 |

## Summary

The quantum Whisper implementation is a **hybrid quantum-classical model** where:
- **Most of Whisper remains classical and frozen** (attention, normalization, etc.)
- **Only the first two conv layers are quantum-enhanced**
- **Only quantum layers and classifier are trained**
- **The rest of the architecture is identical to normal Whisper**

This approach allows us to explore quantum advantages while maintaining the proven Whisper architecture and leveraging pretrained knowledge.
