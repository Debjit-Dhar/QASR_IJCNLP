# Quantum Whisper: Quantum-Enhanced Speech Recognition

A quantum-enhanced implementation of OpenAI's Whisper model for speech recognition, featuring quantum convolutional layers integrated with classical transformer architecture.

## 🚀 Project Overview

This project implements a **quantum-enhanced Whisper model** where classical Conv1d layers in the audio encoder are replaced with **quantum convolutional layers** using PennyLane. The model maintains the original Whisper architecture while introducing quantum computing capabilities for enhanced feature extraction.

## ✨ Features

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
- **Whisper Tiny**: Official OpenAI implementation
- **Audio Encoder**: 4 transformer layers, 384 hidden size, 6 attention heads
- **Audio Input**: 80 mel bins × 3000 time steps

### Quantum Enhancements
- **Quantum Conv1d**: Replaces classical Conv1d in audio encoder
- **PennyLane Backend**: `default.qubit` simulator
- **Configurable Qubits**: Adjustable number of qubits (default: 4)

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

#### Option 1: Quantum ASR Training
```bash
python train_quantum_whisper_asr.py \
    --epochs 50 \
    --lr 1e-3 \
    --batch_size 16 \
    --n_qubits 4 \
    --hidden_size 384 \
    --num_layers 2
```

#### Option 2: Classical ASR Training from Scratch
```bash
python train_whisper_from_scratch.py \
    --model_size tiny \
    --epochs 50 \
    --lr 1e-4 \
    --batch_size 8
```

#### Evaluating ASR Models
```bash
# Evaluate trained quantum ASR model
python evaluate_quantum_whisper_asr.py \
    --model_path best_quantum_whisper_asr.pth \
    --batch_size 16 \
    --n_qubits 4

# Evaluate pretrained classical Whisper models
python evaluate_pretrained_whisper_asr.py \
    --model_size tiny \
    --max_samples 100
```

## 🔧 **Training Parameters Explained**

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `--epochs` | Training epochs | 50 | 50-100 for good results |
| `--lr` | Learning rate | 1e-3 | 1e-3 to 5e-4 |
| `--batch_size` | Batch size | 16 | 8-32 (memory dependent) |
| `--n_qubits` | Quantum qubits | 4 | 2-8 (performance vs speed) |
| `--hidden_size` | LSTM hidden size | 384 | 256-512 |
| `--num_layers` | LSTM layers | 2 | 2-4 |
| `--model_size` | Whisper size | tiny | tiny/base/small |

## 📊 **Understanding Results**

### **Classification Metrics**
- **Accuracy**: Overall classification accuracy
- **Class-wise Accuracy**: Performance per command class
- **WER**: Word Error Rate (less relevant for classification)

### **ASR Metrics**
- **CER (Character Error Rate)**: Character-level accuracy (0.0 = perfect, 1.0 = completely wrong)
- **WER (Word Error Rate)**: Word-level accuracy  
- **Implementation**: Uses `editdistance` library for CER and `jiwer` library for WER
- **Formula**: `(Insertions + Deletions + Substitutions) / Total Characters/Words`
- **Robustness**: Includes fallback implementations if external libraries fail

### **Performance Expectations**

#### Classification (Google Speech Commands)
- **Expected Accuracy**: 80-95% (depending on quantum configuration)
- **Training Time**: ~2-4 hours on CPU, ~30-60 minutes on GPU
- **Memory**: ~4-8 GB RAM

#### ASR (LibriSpeech)
- **Expected CER**: 0.1-0.3 (10-30% character error rate)
- **Expected WER**: 0.15-0.4 (15-40% word error rate)
- **Training Time**: ~4-8 hours on CPU, ~1-2 hours on GPU
- **Memory**: ~8-16 GB RAM

## 🔄 **Training Process**

### **Classification Training**
1. **Dataset Loading**: Google Speech Commands (35 classes)
2. **Model Initialization**: Quantum Whisper with frozen pretrained layers
3. **Training Loop**: Only quantum layers and classifier are trained
4. **Validation**: Accuracy and class-wise performance
5. **Model Saving**: Best model based on validation accuracy

### **ASR Training**
1. **Dataset Loading**: LibriSpeech `train.100`, `validation`, `test`
2. **Vocabulary Building**: Creates character vocabulary from training text
3. **Model Initialization**: Quantum Whisper + LSTM decoder
4. **Training Loop**: Teacher forcing with LSTM decoder
5. **Validation**: Character Error Rate (CER) on validation set
6. **Model Saving**: Best model based on validation CER

## 🔍 **Evaluation Process**

### **Classification Evaluation**
1. **Model Loading**: Loads trained quantum Whisper model
2. **Test Dataset**: Processes Google Speech Commands test split
3. **Inference**: Generates class predictions
4. **Metrics Calculation**: Accuracy, class-wise accuracy, WER
5. **Results**: Sample predictions and performance plots

### **ASR Evaluation**
1. **Model Loading**: Loads trained ASR model and character vocabulary
2. **Test Dataset**: Processes LibriSpeech test split
3. **Inference**: Generates predictions without teacher forcing
4. **Metrics Calculation**: CER and WER using `editdistance` and `jiwer` libraries
5. **Results**: Sample analysis and metrics distribution plots

## 📁 **Project Structure**

```
QASR IJCNLP/
├── quantum_whisper.py                    # Core quantum Whisper implementation
├── librispeech_asr.py                    # ASR dataset and decoder implementation
├── utils.py                              # Common utility functions
├── train_quantum_whisper.py             # Classification training script
├── train_quantum_whisper_asr.py         # Quantum ASR training script
├── train_whisper_from_scratch.py        # Classical ASR training script
├── evaluate_pretrained_whisper.py       # Classification evaluation
├── evaluate_quantum_whisper_asr.py      # Quantum ASR evaluation
├── evaluate_pretrained_whisper_asr.py   # Classical Whisper evaluation
├── whisper/                              # Official Whisper implementation
├── requirements.txt                      # Python dependencies
└── README.md                            # This comprehensive guide
```

## 🔧 **File Overview**

### **Core Implementation Files**
- **`quantum_whisper.py`**: Quantum Conv1d layers, QuantumAudioEncoder, QuantumWhisper
- **`librispeech_asr.py`**: ASR dataset, LSTM decoder, character vocabulary building
- **`utils.py`**: Common functions (CER/WER calculation, plotting, device management)

### **Training Scripts**
- **`train_quantum_whisper.py`**: Classification training (Google Speech Commands)
- **`train_quantum_whisper_asr.py`**: Quantum ASR training (LibriSpeech)
- **`train_whisper_from_scratch.py`**: Classical ASR training from scratch

### **Evaluation Scripts**
- **`evaluate_pretrained_whisper.py`**: Classification evaluation
- **`evaluate_quantum_whisper_asr.py`**: Quantum ASR evaluation
- **`evaluate_pretrained_whisper_asr.py`**: Classical Whisper ASR evaluation

## 🎯 **Research Applications**

### **Quantum Advantage Studies**
- Compare quantum vs classical Conv1d performance
- Analyze quantum circuit depth vs ASR performance
- Study quantum noise effects on speech recognition

### **ASR Benchmarking**
- Compare with classical Whisper models
- Analyze character-level vs token-level approaches
- Study LSTM vs Transformer decoder performance

### **Hybrid Models**
- Combine quantum audio processing with classical text generation
- Explore quantum attention mechanisms
- Study quantum-classical interface optimization

## 🚨 **Important Notes**

### **Task Alignment**
- **Google Speech Commands**: Use for classification only
- **LibriSpeech**: Use for ASR only (proper implementation)
- **Classification on LibriSpeech**: Not recommended (wrong task)

### **Model Paths**
- Ensure model paths match when evaluating
- Check for training history files for vocabulary
- Use correct model size parameters

### **Performance Considerations**
- **Quantum Simulation**: PennyLane `default.qubit` backend for CPU compatibility
- **Memory Management**: Reduce batch size for large models
- **Training Time**: Quantum models may take longer due to simulation overhead

## 🔧 **Troubleshooting**

### **Common Issues**

#### 1. Memory Errors
```bash
# Reduce batch size
python train_quantum_whisper_asr.py --batch_size 8

# Reduce hidden size
python train_quantum_whisper_asr.py --hidden_size 256
```

#### 2. Slow Training
```bash
# Use GPU if available
python train_quantum_whisper_asr.py --device cuda

# Reduce number of qubits
python train_quantum_whisper_asr.py --n_qubits 2
```

#### 3. Poor Convergence
```bash
# Lower learning rate
python train_quantum_whisper_asr.py --lr 5e-4

# More training epochs
python train_quantum_whisper_asr.py --epochs 100
```

### **Dependencies**
- **Core**: PyTorch, PennyLane, Transformers
- **Audio**: torchaudio, soundfile, librosa
- **Data**: datasets, numpy, tqdm
- **Visualization**: matplotlib

## 📊 **Output Files**

### **Training Outputs**
- `best_quantum_whisper_val_acc.pth`: Best classification model
- `best_quantum_whisper_asr.pth`: Best quantum ASR model
- `best_whisper_from_scratch.pth`: Best classical ASR model
- `*_final.pth`: Final models after training
- `*_training_history.json`: Training logs and metrics
- `*_training_results.png`: Training curves and metrics plots

### **Evaluation Outputs**
- `*_evaluation_results.json`: Detailed evaluation metrics
- `*_evaluation_results.png`: Performance visualization
- `*_metrics_distribution.png`: CER/WER distribution plots

## 🔮 **Future Enhancements**

### **Short Term**
- **Beam Search**: Improve text generation quality
- **Attention Mechanism**: Add attention to LSTM decoder
- **Subword Tokenization**: BPE or SentencePiece integration

### **Long Term**
- **Quantum Attention**: Quantum-enhanced attention mechanisms
- **Quantum LSTM**: Quantum LSTM cells
- **End-to-End Quantum**: Fully quantum ASR pipeline

## 📖 **Citation**

If you use this code in your research, please cite:

```bibtex
@misc{quantum_whisper_2024,
  title={Quantum Whisper: Quantum-Enhanced Speech Recognition},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/quantum-whisper}
}
```

## 🤝 **Contributing**

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🎯 **Quick Reference Commands**

### **Classification Research**
```bash
# Train
python train_quantum_whisper.py --dataset google --epochs 50 --n_qubits 4
# Evaluate
python evaluate_pretrained_whisper.py --dataset google
```

### **Quantum ASR Research**
```bash
# Train
python train_quantum_whisper_asr.py --epochs 50 --n_qubits 4
# Evaluate
python evaluate_quantum_whisper_asr.py --model_path best_quantum_whisper_asr.pth
```

### **Classical ASR Research**
```bash
# Train from scratch
python train_whisper_from_scratch.py --model_size tiny --epochs 50
# Evaluate pretrained
python evaluate_pretrained_whisper_asr.py --model_size tiny
```

This comprehensive guide covers all aspects of the Quantum Whisper project, from basic usage to advanced research applications. The modular architecture and utility functions ensure code reusability and maintainability.
