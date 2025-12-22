"""
Utility script for checking system compatibility and setup
Run this before starting training to verify everything is configured correctly
"""

import sys
import platform
import subprocess


def check_uv():
    """Check if uv is installed."""
    print("\n🚀 Checking uv package manager...")
    try:
        subprocess.run(['uv', '--version'], check=True, capture_output=True)
        print("   ✅ uv is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("   ⚠️  uv not found (optional but recommended)")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.9+")
        return False


def check_architecture():
    """Check if running on Apple Silicon."""
    print("\n💻 Checking system architecture...")
    arch = platform.machine()
    system = platform.system()

    print(f"   System: {system}")
    print(f"   Architecture: {arch}")

    if arch == "arm64":
        print(f"   ✅ Running on Apple Silicon (M1/M2/M3)")
        return True
    else:
        print(f"   ⚠️  Not on Apple Silicon - May have reduced performance")
        return False


def check_torch():
    """Check if PyTorch is installed and MPS is available."""
    print("\n🔥 Checking PyTorch...")
    try:
        import torch
        print(f"   ✅ PyTorch {torch.__version__} installed")

        # Check MPS availability
        if torch.backends.mps.is_available():
            print(f"   ✅ MPS (Metal Performance Shaders) is available")
            print(f"   ✅ GPU acceleration enabled!")
            return True
        else:
            print(f"   ⚠️  MPS not available - will use CPU")
            return False
    except ImportError:
        print(f"   ❌ PyTorch not installed")
        return False


def check_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")

    required_packages = [
        'transformers',
        'accelerate',
        'peft',
        'datasets',
        'trl',
        'yaml'
    ]

    all_installed = True
    for package in required_packages:
        try:
            if package == 'yaml':
                __import__('yaml')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - Not installed")
            all_installed = False

    # Note about bitsandbytes
    print(f"   ⚠️  bitsandbytes - Not available on macOS (using float16 instead)")

    return all_installed


def check_memory():
    """Check system memory."""
    print("\n💾 Checking system memory...")
    try:
        # macOS specific
        result = subprocess.run(
            ['sysctl', 'hw.memsize'],
            capture_output=True,
            text=True
        )
        memory_bytes = int(result.stdout.split(':')[1].strip())
        memory_gb = memory_bytes / (1024**3)

        print(f"   Total RAM: {memory_gb:.1f} GB")

        if memory_gb >= 16:
            print(f"   ✅ Sufficient memory for most models")
        elif memory_gb >= 8:
            print(f"   ⚠️  8-16GB: Use TinyLlama or small models")
        else:
            print(f"   ❌ <8GB: May struggle with training")

        return True
    except Exception as e:
        print(f"   ⚠️  Could not determine memory: {e}")
        return False


def check_disk_space():
    """Check available disk space."""
    print("\n💽 Checking disk space...")
    try:
        result = subprocess.run(
            ['df', '-h', '.'],
            capture_output=True,
            text=True
        )
        lines = result.stdout.split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            available = parts[3]
            print(f"   Available space: {available}")

            # Try to parse space (rough estimate)
            if 'G' in available:
                space_gb = float(available.replace('G', '').replace('i', ''))
                if space_gb >= 50:
                    print(f"   ✅ Sufficient disk space")
                else:
                    print(f"   ⚠️  <50GB: May need more for models and outputs")
        return True
    except Exception as e:
        print(f"   ⚠️  Could not determine disk space: {e}")
        return False


def check_config_file():
    """Check if config file exists."""
    print("\n⚙️  Checking configuration...")
    import os

    if os.path.exists('config.yaml'):
        print(f"   ✅ config.yaml found")
        try:
            import yaml
            with open('config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            print(f"   ✅ config.yaml is valid YAML")
            print(f"   Model: {config.get('model', {}).get('name', 'Not specified')}")
            return True
        except Exception as e:
            print(f"   ❌ Error reading config.yaml: {e}")
            return False
    else:
        print(f"   ❌ config.yaml not found")
        return False


def estimate_training_time():
    """Estimate training time based on configuration."""
    print("\n⏱️  Training Time Estimates:")
    print("   (Based on 3 epochs, full Alpaca dataset)")
    print("")
    print("   TinyLlama (1.1B):  ~2-4 hours")
    print("   Phi-2 (2.7B):      ~4-6 hours")
    print("   Mistral-7B:        ~6-10 hours")
    print("   Llama-2-7B:        ~8-12 hours")
    print("")
    print("   💡 Tip: Start with 100 samples for quick testing")


def print_recommendations():
    """Print recommendations based on checks."""
    print("\n📋 Recommendations:")
    print("")
    print("For first-time setup:")
    print("   1. Run: ./setup.sh")
    print("   2. Activate venv: source venv/bin/activate")
    print("   3. Test data: python main.py prepare-data")
    print("   4. Quick train: Edit config.yaml, set max_samples: 100")
    print("   5. Start training: python main.py train")
    print("")
    print("For memory issues:")
    print("   - Use TinyLlama model")
    print("   - Set per_device_train_batch_size: 1")
    print("   - Set gradient_accumulation_steps: 8")
    print("   - Reduce max_seq_length: 256")
    print("")
    print("For faster results:")
    print("   - Set max_samples: 100-1000")
    print("   - Set num_train_epochs: 1-2")
    print("   - Use smaller model (TinyLlama)")


def main():
    """Run all checks."""
    print("=" * 60)
    print("LLM Fine-tuning Project - System Check")
    print("=" * 60)

    checks = []

    checks.append(("UV Package Manager", check_uv()))
    checks.append(("Python Version", check_python_version()))
    checks.append(("Architecture", check_architecture()))
    checks.append(("PyTorch & MPS", check_torch()))
    checks.append(("Required Packages", check_packages()))
    checks.append(("System Memory", check_memory()))
    checks.append(("Disk Space", check_disk_space()))
    checks.append(("Configuration", check_config_file()))

    estimate_training_time()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n🎉 All checks passed! You're ready to start training!")
        print("\nNext steps:")
        print("   python main.py prepare-data")
        print("   python main.py train")
    elif passed >= total - 2:
        print("\n⚠️  Most checks passed. Review warnings above.")
        print("   You should be able to proceed with training.")
    else:
        print("\n❌ Several checks failed. Please fix the issues above.")
        print("   Run ./setup.sh to install dependencies.")

    print_recommendations()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()

