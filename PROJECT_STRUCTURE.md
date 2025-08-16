# Project Structure

## 🎯 **Core Implementation Files**

### **Main Scripts**
- **`train_quantum_whisper.py`** - Main training script for quantum Whisper models
  - Supports both Google Speech Commands and LibriSpeech datasets
  - Uses full dataset loading (no sampling limitations)
  - Configurable quantum parameters (qubits, learning rate, epochs)
  - Model: Whisper Tiny + Quantum Conv1d layers

- **`evaluate_pretrained_whisper.py`** - Evaluation script for pretrained models
  - Evaluates on both datasets
  - Calculates WER, accuracy, and class-wise performance
  - Model: Whisper Tiny (Hugging Face)

- **`quantum_whisper.py`** - Quantum Whisper implementation
  - Extends official Whisper with quantum convolutional layers
  - PennyLane backend for quantum computing
  - Model: Whisper Tiny + Quantum enhancements

### **Supporting Files**
- **`train_whisper_from_scratch.py`** - Classical Whisper training from scratch
  - For comparison with quantum approach
  - Model: Whisper Tiny (classical implementation)

- **`requirements.txt`** - Python dependencies
- **`README.md`** - Project documentation and usage guide

## 📚 **Documentation Files**

- **`QUANTUM_ARCHITECTURE_EXPLANATION.md`** - Detailed quantum implementation explanation
- **`IMPLEMENTATION_SUMMARY.md`** - Implementation overview and methodology
- **`PROJECT_SUMMARY.md`** - Project goals and research context
- **`PROJECT_STRUCTURE.md`** - This file

## 🗂️ **Directories**

- **`whisper/`** - Official OpenAI Whisper implementation
- **`data/`** - Dataset storage directory
- **`qasr_env/`** - Python virtual environment

## 🚀 **Model Information**

### **Base Model: Whisper Tiny**
- **Architecture**: 4 transformer layers, 384 hidden size, 6 attention heads
- **Parameters**: ~39M
- **Input**: 80 mel bins × 3000 time steps
- **Vocabulary**: 51,865 tokens

### **Quantum Enhancements**
- **Quantum Conv1d**: Replaces classical Conv1d in audio encoder
- **Quantum Circuit**: Amplitude embedding + rotations + CNOT gates
- **Backend**: PennyLane default.qubit simulator
- **Configurable**: Number of qubits (default: 4)

## 🎯 **Usage**

### **Training Quantum Model**
```bash
python train_quantum_whisper.py --dataset librispeech --epochs 50 --n_qubits 4
```

### **Evaluation**
```bash
python evaluate_pretrained_whisper.py --dataset librispeech
```

### **Classical Training**
```bash
python train_whisper_from_scratch.py
```

## ✅ **Cleaned Up**

Removed unnecessary test files:
- ~~`test_librispeech_simple.py`~~
- ~~`test_librispeech_fast.py`~~
- ~~`test_librispeech.py`~~
- ~~`test_google_speech_commands.py`~~
- ~~`test_imports.py`~~
- ~~`test_all_experiments.py`~~
- ~~`example_usage.py`~~

The project now contains only essential implementation files and comprehensive documentation.
