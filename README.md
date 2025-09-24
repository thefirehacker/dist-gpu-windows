# Distributed PyTorch Setup: Mac + Windows GPU

A distributed PyTorch setup for running computations across multiple machines - Mac as coordinator and Windows with RTX 2080 as worker.

## 🎯 Overview

This project demonstrates how to set up distributed PyTorch training/inference across heterogeneous systems:
- **Mac (Coordinator)**: Runs on CPU/MPS and coordinates operations
- **Windows RTX 2080 (Worker)**: Provides GPU computation power

## 📁 Project Structure

```
.
├── L1-nbdistributed/
│   └── Ops.ipynb              # Main Jupyter notebook (Mac coordinator)
├── worker.py                   # Windows worker script
├── test_connection.py          # Network connectivity test
├── archive/                    # Previous versions and backups
├── WINDOWS_SETUP.md           # Detailed Windows setup instructions
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

**Mac (Coordinator):**
- Python 3.11+
- PyTorch 2.8.0+
- Jupyter notebook
- Virtual environment activated

**Windows (Worker):**
- Python 3.11+
- PyTorch with CUDA support
- NVIDIA RTX 2080 with drivers
- CUDA toolkit

### Setup Steps

#### 1. Mac Setup (Coordinator)

```bash
# Clone and navigate to project
git clone <repository-url>
cd Code-Scratch

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install torch torchvision jupyter nbdistributed

# Start Jupyter
jupyter notebook
```

#### 2. Windows Setup (Worker)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run the worker script
python worker.py
```

#### 3. Connection Sequence

**IMPORTANT: Follow this exact order:**

1. **Run Mac cells in order** (in Jupyter notebook):
   - Cell 1: Install packages
   - Cell 2: Setup and IP detection  
   - Cell 3: Configuration
   - Cell 4: IP verification
   - Cell 5: Function definitions
   - Cell 6: **START COORDINATOR** 

2. **After Mac shows "Ready"**, run on Windows:
   ```bash
   python worker.py
   ```
   Enter Mac IP when prompted (auto-detected, e.g., `192.168.29.234`)

3. **Connection established** - both machines will show success messages

## 🔧 Configuration

### Network Settings
- **Port**: 12355 (configurable)
- **Backend**: Gloo (CPU-GPU mixed setup)
- **IP Detection**: Automatic (no manual configuration needed)

### Environment Variables
```bash
MASTER_ADDR=<auto-detected-mac-ip>
MASTER_PORT=12355
WORLD_SIZE=2
RANK=0  # Mac coordinator
RANK=1  # Windows worker
```

## 🧪 Testing

### Basic Connection Test
```python
# In Jupyter notebook after connection
result = test_all_gather_basic()
```

### Network Connectivity Test
```bash
# On Windows, test connection to Mac
python test_connection.py
```

## 📊 Features

- **Heterogeneous Setup**: Mac CPU/MPS + Windows GPU
- **Automatic IP Detection**: No manual IP configuration
- **IP-only Connections**: Avoids hostname resolution issues
- **Connection Retry Logic**: Robust connection handling
- **Device-aware Operations**: Automatic device allocation based on rank

## 🐛 Troubleshooting

### Common Issues

#### Connection Timeout
```
Error: The client socket has failed to connect
```
**Solution**: Ensure Mac coordinator is running first, then start Windows worker

#### Hostname Resolution Error
```
Error: failed to connect to AMARDEEPS-IMAC
```
**Solution**: Use the updated scripts that force IP-only connections

#### Port Already in Use
```
Error: Port 12355 not available
```
**Solution**: 
- Check if another process is using the port: `lsof -i :12355`
- Restart both coordinator and worker

#### CUDA Not Available on Windows
```
Warning: CUDA not available! Running on CPU
```
**Solution**: 
- Install CUDA toolkit
- Install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Network Requirements

- Both machines must be on the same network
- Port 12355 must be open for communication
- Firewall may need configuration
- No VPN interference

## 🔄 Development Workflow

1. **Make changes** to coordinator notebook or worker script
2. **Test locally** with connection verification
3. **Commit changes** (Git automatically ignores temporary files)
4. **Deploy** to both machines

## 📈 Performance Tips

- Use appropriate tensor sizes for network transfer
- Leverage device-specific optimizations (MPS on Mac, CUDA on Windows)
- Monitor network latency between machines
- Consider batch sizes for distributed operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test changes on both Mac and Windows
4. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

- PyTorch distributed documentation
- Gloo backend for cross-platform compatibility
- Community contributions and testing

---

For detailed Windows setup instructions, see [`WINDOWS_SETUP.md`](WINDOWS_SETUP.md).

For issues and questions, please open a GitHub issue.
