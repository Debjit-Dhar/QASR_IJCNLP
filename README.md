# Quantum Whisper: Quantum-Enhanced Speech Recognition

A quantum-enhanced implementation of OpenAI's **official Whisper Tiny model** for speech recognition, featuring quantum convolutional layers integrated with the classical transformer architecture.

## 🚀 Project Overview

This project implements a **quantum-enhanced Whisper model** where classical Conv1d layers in the audio encoder are replaced with **quantum convolutional layers** using PennyLane. The model extends the **official OpenAI Whisper Tiny implementation** while introducing quantum computing capabilities for enhanced feature extraction.

## ✨ Features

- **Official Whisper Tiny Model**: Based on OpenAI's official implementation
- **Quantum-Enhanced Audio Processing**: Quantum Conv1d layers for audio feature extraction
- **Multi-Dataset Support**: Google Speech Commands (classification) and LibriSpeech (ASR)
- **PennyLane Integration**: Quantum circuit simulation with `default.qubit` backend
- **Flexible Architecture**: Support for different numbers of qubits and quantum layers
- **Proper ASR Implementation**: Character-level prediction for LibriSpeech
- **Complete Training & Evaluation**: Both quantum and classical approaches
- **Dual Metrics**: Character Error Rate (CER) and Word Error Rate (WER)

## 🎯 Dataset Options

### 1. Google Speech Commands (Classification)
- **Task**: 35-class speech command classification
- **Format**: Audio → Command label
- **Usage**: Perfect for classification tasks
- **Splits**: Train/Validation/Test

### 2. LibriSpeech (ASR - Automatic Speech Recognition)
- **Task**: Continuous speech recognition
- **Format**: Audio → Text transcription
- **Usage**: Proper ASR with character-level prediction
- **Splits**: `train.100`/`validation`/`test`

## 🏗️ Model Architecture

### Base Model
- **Whisper Tiny**: **Official OpenAI implementation** (39M parameters)
- **Audio Encoder**: 4 transformer layers, 384 hidden size, 6 attention heads
- **Audio Input**: 80 mel bins × 3000 time steps
- **Vocabulary**: 51,865 tokens

### Quantum Enhancements
- **Quantum Conv1d**: Replaces classical Conv1d in audio encoder
- **PennyLane Backend**: `default.qubit` simulator
- **Configurable Qubits**: Adjustable number of qubits (default: 4)
- **Weight Transfer**: Copies pretrained weights from official model

### ASR Decoder (LibriSpeech)
- **LSTM Decoder**: Multi-layer LSTM for sequence generation
- **Character-Level**: Predicts text character by character
- **Teacher Forcing**: During training for stable learning

## 🚀 Quick Start

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv qasr_env
source qasr_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Test Model Loading
```bash
# Test that official Whisper models can be loaded
python test_official_whisper.py
```

## 📚 **Complete Usage Guide**

### **Classification Research (Google Speech Commands)**

#### Training Quantum Whisper for Classification
```bash
python train_quantum_whisper.py \
    --dataset google \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 16 \
    --n_qubits 4
```

#### Evaluating Classification Model
```bash
python evaluate_pretrained_whisper.py \
    --dataset google \
    --batch_size 16
```

### **ASR Research (LibriSpeech)**

#### Option 1: Quantum ASR Training (Recommended)
```bash
python train_quantum_whisper_asr.py \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 16 \
    --n_qubits 4 \
    --hidden_size 384 \
    --num_layers 2
```

#### Option 2: Official Whisper Tiny Training
```bash
python train_whisper_from_scratch.py \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 8
```

#### Evaluating ASR Models
```bash
# Evaluate quantum ASR model (best CER)
python evaluate_quantum_whisper_asr.py \
    --model_path best_quantum_whisper_asr_cer.pth \
    --n_qubits 4 \
    --hidden_size 384 \
    --num_layers 2

# Evaluate quantum ASR model (best WER)
python evaluate_quantum_whisper_asr.py \
    --model_path best_quantum_whisper_asr_wer.pth \
    --n_qubits 4 \
    --hidden_size 384 \
    --num_layers 2

# Evaluate official Whisper model
python evaluate_pretrained_whisper.py \
    --batch_size 16
```

## 🔧 **Model Loading Strategy**

The project uses a **three-tier loading strategy** to ensure compatibility:

## 📥 **Automatic Dataset Download**

All training and evaluation scripts automatically handle LibriSpeech dataset downloading:

- **Automatic Detection**: Scripts check if dataset is already downloaded locally
- **Smart Downloading**: Downloads only if not found locally
- **Caching**: Downloaded datasets are cached for future use
- **Fallback Handling**: Graceful fallbacks if download fails
- **Progress Updates**: Clear progress information during download

**Supported Datasets:**
- **LibriSpeech Clean**: `train-clean-100` (training), `dev-clean` (validation)
- **Automatic Splits**: Training/validation split handled automatically
- **Primary**: Torchaudio integration (like official notebook)
- **Fallback**: Hugging Face datasets library for reliable downloading

**First Run**: Will download ~1GB of LibriSpeech data automatically
**Subsequent Runs**: Uses cached local dataset for instant loading

1. **Local Whisper Directory**: First tries to load from the included `whisper/` directory
2. **Hugging Face**: Falls back to `openai/whisper-tiny` from Hugging Face
3. **Official Dimensions**: Creates model with official Whisper Tiny dimensions as last resort

This ensures that **only the official Whisper Tiny model** is used, with no simplified or dummy versions.

## 📊 **Training & Evaluation**

### **Training Process**

#### **Classical Whisper ASR Training**
- **From Scratch**: Creates Whisper model architecture without pretrained weights
- **Full Training**: Trains all model parameters on LibriSpeech dataset
- **ASR Focus**: Optimized for speech recognition with character-level prediction
- **Dual Model Saving**: Saves best models for both CER and WER

#### **Quantum Whisper ASR Training**
- **Official Model**: Loads pretrained Whisper Tiny weights
- **Quantum Layers**: Replaces Conv1d layers with quantum equivalents
- **Weight Transfer**: Copies pretrained weights to non-quantum layers
- **Fine-tuning**: Trains only quantum layers while preserving official weights
- **Dual Model Saving**: Saves two separate models:
  - **Best CER Model**: Model with lowest Character Error Rate
  - **Best WER Model**: Model with lowest Word Error Rate

### **Evaluation Metrics**
- **Character Error Rate (CER)**: Character-level accuracy for ASR
- **Word Error Rate (WER)**: Word-level accuracy for ASR
- **Classification Accuracy**: For Google Speech Commands

### **Model Saving Strategy**
During training, the system automatically tracks and saves the best performing models:

1. **Best CER Model**: `best_*_cer.pth` - Saved when validation CER improves
2. **Best WER Model**: `best_*_wer.pth` - Saved when validation WER improves
3. **Final Model**: `*_final.pth` - Model state after final training epoch

This ensures you have access to the optimal model for each metric, allowing you to choose the best model based on your specific requirements.

## 🎯 **Key Benefits of Official Model Usage**

1. **Proven Performance**: Uses OpenAI's tested and optimized architecture
2. **Pretrained Weights**: Leverages knowledge from massive training data
3. **Standard Dimensions**: Follows official model specifications exactly
4. **Compatibility**: Ensures consistency with other Whisper implementations
5. **Research Validity**: Results comparable to official benchmarks

## 🚨 **Important Notes**

- **No Dummy Models**: All implementations use the official Whisper Tiny model
- **Full Dataset**: Evaluation always uses the complete LibriSpeech test set
- **Weight Preservation**: Quantum enhancements don't affect pretrained weights
- **Standard Metrics**: Uses standard ASR evaluation metrics (CER, WER)

## 📁 **File Structure**

```
QASR IJCNLP/
├── whisper/                          # Official Whisper implementation
├── train_whisper_from_scratch.py    # Train official Whisper Tiny (demo)
├── train_classical_whisper_asr.py   # Train classical Whisper ASR from scratch (auto-downloads LibriSpeech, uses notebook insights)
├── train_quantum_whisper_asr.py     # Train quantum ASR model
├── evaluate_pretrained_whisper.py   # Evaluate official Whisper
├── evaluate_quantum_whisper_asr.py  # Evaluate quantum ASR model
├── quantum_whisper.py               # Quantum Whisper implementation
├── librispeech_asr.py               # ASR dataset and model
├── test_official_whisper.py         # Test model loading
├── utils.py                         # Utility functions
└── requirements.txt                 # Dependencies

# Training Output Files:
├── best_classical_whisper_asr_cer.pth    # Best classical model by CER
├── best_classical_whisper_asr_wer.pth    # Best classical model by WER
├── best_quantum_whisper_asr_cer.pth      # Best quantum model by CER
├── best_quantum_whisper_asr_wer.pth      # Best quantum model by WER
├── *_checkpoint_epoch_*.pth              # Periodic checkpoints
├── *_training_history.json               # Training metrics and history
└── *_training_results.png                # Training curves visualization
```

## 🔬 **Research Applications**

This implementation is designed for:
- **Quantum Machine Learning Research**: Studying quantum advantages in ASR
- **Model Compression**: Quantum layers as alternative to classical layers
- **Hybrid Architectures**: Combining classical and quantum computing
- **ASR Benchmarking**: Comparing quantum vs. classical approaches

## 📈 **Performance Expectations**

- **Base Performance**: Matches official Whisper Tiny baseline
- **Quantum Enhancement**: Potential improvements through quantum feature extraction
- **Training Efficiency**: Faster convergence with pretrained weights
- **Memory Usage**: Similar to official model with quantum overhead

## 🤝 **Contributing**

When contributing to this project:
1. **Maintain Official Model**: Always use the official Whisper Tiny architecture
2. **Preserve Weights**: Don't modify pretrained weights unnecessarily
3. **Test Loading**: Ensure models can be loaded with `test_official_whisper.py`
4. **Full Evaluation**: Always evaluate on complete datasets, not samples

## 📄 **License**

This project follows the same license as the official Whisper implementation.
