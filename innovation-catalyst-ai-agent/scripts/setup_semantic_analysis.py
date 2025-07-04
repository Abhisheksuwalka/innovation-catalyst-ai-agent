# scripts/setup_semantic_analysis.py
"""
Setup script for semantic analysis dependencies.
Handles installation and validation of required packages.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package_name: str, pip_name: str = None) -> bool:
    """Install a package using pip."""
    pip_name = pip_name or package_name
    
    try:
        print(f"Installing {package_name}...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", pip_name
        ])
        print(f"✅ {package_name} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}: {e}")
        return False

def download_spacy_model(model_name: str) -> bool:
    """Download spaCy model."""
    try:
        print(f"Downloading spaCy model: {model_name}...")
        subprocess.check_call([
            sys.executable, "-m", "spacy", "download", model_name
        ])
        print(f"✅ spaCy model {model_name} downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model {model_name}: {e}")
        return False

def setup_semantic_analysis():
    """Setup all semantic analysis dependencies."""
    print("🚀 Setting up Semantic Analysis Dependencies...")
    
    # Required packages
    packages = [
        ("sentence-transformers", "sentence-transformers>=2.2.0"),
        ("torch", "torch>=2.0.0"),
        ("spacy", "spacy>=3.6.0"),
        ("nltk", "nltk>=3.8.0"),
        ("yake", "yake>=0.4.8"),
        ("scikit-learn", "scikit-learn>=1.3.0"),
        ("pytest", "pytest>=7.0.0"),
        ("pytest-cov", "pytest-cov>=3.0.0"),
        ("pytest-mock", "pytest-mock>=3.10.0"),
        ("pytest-asyncio", "pytest-asyncio>=0.20.0"),
        ("pytest-xdist", "pytest-xdist>=3.0.0"),
        ("smolagents", "smolagents>=1.19.0,<2.0.0"),
        ("libmagic", "libmagic"),
        ("python-magic", "python-magic>=0.4.27"),
        # ("python-magic-bin", "python-magic-bin"),
        ("python-dotenv", "python-dotenv>=1.0.0"),
        ("PyPDF2", "PyPDF2>=3.0.0"),
        ("pdfplumber", "pdfplumber>=0.10.0"),
        ("beautifulsoup4", "beautifulsoup4>=4.12.0"),
        ("docx", "docx"),
        ("pandas", "pandas>=2.0.0"),
        ("chardet", "chardet>=5.0.0"),
        ("duckduckgo-search", "duckduckgo-search>=0.6.0"),
        ("psutil", "psutil>=5.9.0"),
        ("litellm", "litellm"),
        ("azure-ai-inference", "azure-ai-inference"),
        ("nltk", "nltk>=3.8.0"),
        ("time", "time"),
    ]
    
    success_count = 0
    
    # Install packages
    for package_name, pip_name in packages:
        if install_package(package_name, pip_name):
            success_count += 1
    
    print(f"\n📦 Installed {success_count}/{len(packages)} packages")
    
    # Download spaCy model
    if success_count >= len(packages) - 1:  # Allow for some failures
        download_spacy_model("en_core_web_sm")
    
    # Test installations
    print("\n🧪 Testing installations...")
    test_imports()
    
    print("\n✅ Semantic analysis setup complete!")

def test_imports():
    """Test that all packages can be imported."""
    test_packages = [
        ("sentence_transformers", "SentenceTransformer"),
        ("torch", "torch"),
        ("spacy", "spacy"),
        ("nltk", "nltk"),
        ("yake", "yake"),
        ("sklearn", "scikit-learn"),
    ]
    
    for module_name, display_name in test_packages:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name}")

if __name__ == "__main__":
    setup_semantic_analysis()
