#!/usr/bin/env python3
"""
Simple test script to verify Windows RTX 2050 setup without distributed training
"""

import torch
import sys

def test_cuda_setup():
    """Test CUDA installation and GPU availability"""
    print("=== Windows RTX 2050 Setup Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ CUDA not available!")
        return False

def test_tensor_operations():
    """Test basic tensor operations on GPU"""
    print("\n=== GPU Tensor Operations Test ===")
    
    # Test tensor creation and movement to GPU
    device = torch.device("cuda:0")
    
    # Create test tensors
    a = torch.randn(3, 3).to(device)
    b = torch.randn(3, 3).to(device)
    
    print(f"Tensor A device: {a.device}")
    print(f"Tensor B device: {b.device}")
    
    # Test operations
    c = torch.matmul(a, b)
    print(f"Matrix multiplication result device: {c.device}")
    print(f"Result shape: {c.shape}")
    
    # Test memory usage
    torch.cuda.synchronize()
    memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
    print(f"GPU memory allocated: {memory_allocated:.2f} MB")
    
    return True

def test_distributed_imports():
    """Test if distributed training imports work"""
    print("\n=== Distributed Training Imports Test ===")
    
    try:
        import torch.distributed as dist
        print("✅ torch.distributed imported successfully")
        
        # Test backend availability
        backends = []
        if dist.is_gloo_available():
            backends.append("gloo")
        if dist.is_nccl_available():
            backends.append("nccl")
        if dist.is_mpi_available():
            backends.append("mpi")
        
        print(f"Available backends: {backends}")
        
        return True
        
    except Exception as e:
        print(f"❌ Distributed imports failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Windows RTX 2050 Setup Verification")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: CUDA Setup
    if test_cuda_setup():
        tests_passed += 1
        print("✅ CUDA test passed")
    else:
        print("❌ CUDA test failed")
        return
    
    # Test 2: GPU Operations
    try:
        if test_tensor_operations():
            tests_passed += 1
            print("✅ GPU operations test passed")
        else:
            print("❌ GPU operations test failed")
    except Exception as e:
        print(f"❌ GPU operations test failed: {e}")
    
    # Test 3: Distributed Imports
    if test_distributed_imports():
        tests_passed += 1
        print("✅ Distributed imports test passed")
    else:
        print("❌ Distributed imports test failed")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your setup is ready for distributed training.")
        print("\nNext steps:")
        print("1. Start the Mac coordinator (run the Jupyter notebook)")
        print("2. Run: python worker.py")
        print("3. Enter the Mac's IP address when prompted")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
