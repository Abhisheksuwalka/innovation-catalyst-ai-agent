# project_structure.py
"""
Innovation Catalyst Agent - Project Structure Setup
Creates the complete project directory structure with proper organization.
"""

import os
from pathlib import Path
from typing import List, Dict

class ProjectStructure:
    """
    Manages the complete project directory structure for Innovation Catalyst Agent.
    
    Features:
        - Creates organized directory structure
        - Generates necessary configuration files
        - Sets up proper Python package structure
        - Includes testing and documentation directories
    """
    
    def __init__(self, project_root: str = "innovation_catalyst"):
        self.project_root = Path(project_root)
        self.structure = {
            "src": {
                "innovation_catalyst": {
                    "tools": ["__init__.py"],
                    "agents": ["__init__.py"],
                    "utils": ["__init__.py"],
                    "models": ["__init__.py"],
                    "__init__.py": None
                }
            },
            "tests": {
                "unit": ["__init__.py"],
                "integration": ["__init__.py"],
                "fixtures": ["__init__.py"],
                "__init__.py": None
            },
            "docs": ["README.md"],
            "examples": ["sample_documents"],
            "configs": ["logging.yaml"],
            "scripts": ["setup.py"],
            ".github": {
                "workflows": ["ci.yml"]
            }
        }
    
    def create_structure(self) -> None:
        """Create the complete project directory structure."""
        self._create_directories(self.structure, self.project_root)
        self._create_root_files()
        print(f"✅ Project structure created at: {self.project_root.absolute()}")
    
    def _create_directories(self, structure: Dict, base_path: Path) -> None:
        """Recursively create directory structure."""
        for name, content in structure.items():
            current_path = base_path / name
            
            if isinstance(content, dict):
                current_path.mkdir(parents=True, exist_ok=True)
                self._create_directories(content, current_path)
            elif isinstance(content, list):
                current_path.mkdir(parents=True, exist_ok=True)
                for file_name in content:
                    (current_path / file_name).touch()
            else:
                current_path.touch()
    
    def _create_root_files(self) -> None:
        """Create essential root-level files."""
        files_content = {
            "requirements.txt": self._get_requirements(),
            "setup.py": self._get_setup_py(),
            ".gitignore": self._get_gitignore(),
            "README.md": self._get_readme(),
            "pyproject.toml": self._get_pyproject_toml()
        }
        
        for filename, content in files_content.items():
            with open(self.project_root / filename, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def _get_requirements(self) -> str:
        """Generate requirements.txt content."""
        return """# Core Dependencies
smolagents>=0.3.0
transformers>=4.30.0
torch>=2.0.0
sentence-transformers>=2.2.0
numpy>=1.24.0
pandas>=2.0.0

# Document Processing
PyPDF2>=3.0.0
python-docx>=0.8.11
python-magic>=0.4.27

# NLP Libraries
spacy>=3.6.0
nltk>=3.8.0
scikit-learn>=1.3.0

# Web Interface
gradio>=3.40.0
streamlit>=1.25.0

# Utilities
pydantic>=2.0.0
loguru>=0.7.0
tqdm>=4.65.0
click>=8.1.0

# Development Dependencies
pytest>=7.4.0
pytest-cov>=4.1.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.5.0
pre-commit>=3.3.0

# Optional Dependencies
faiss-cpu>=1.7.4  # For vector similarity search
plotly>=5.15.0    # For visualization
"""
    
    def _get_setup_py(self) -> str:
        """Generate setup.py content."""
        return '''"""Setup script for Innovation Catalyst Agent."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="innovation-catalyst",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI Agent for discovering innovative connections between ideas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/innovation-catalyst",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "innovation-catalyst=innovation_catalyst.cli:main",
        ],
    },
)
'''
    
    def _get_gitignore(self) -> str:
        """Generate .gitignore content."""
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project Specific
logs/
cache/
temp/
*.log
.pytest_cache/
.coverage
htmlcov/
.mypy_cache/
.tox/

# Model files
models/
*.bin
*.safetensors

# Data files
data/
uploads/
outputs/

# Secrets
.env.local
.env.production
secrets.yaml
"""
    
    def _get_readme(self) -> str:
        """Generate README.md content."""
        return """# 🚀 Innovation Catalyst Agent

An AI-powered agent that discovers novel connections between ideas and generates actionable innovation insights from uploaded documents.

## Features

- **Multi-format Document Processing**: Supports PDF, DOCX, TXT, and Markdown files
- **Semantic Analysis**: Advanced NLP for entity extraction and topic modeling
- **Connection Discovery**: Finds unexpected relationships between different concepts
- **Innovation Synthesis**: Generates comprehensive innovation reports with actionable steps
- **SmolAgent Integration**: Built on the robust SmolAgent framework
- **Web Interface**: User-friendly Gradio interface for easy interaction

## Quick Start
```
Clone the repository
git clone https://github.com/yourusername/innovation-catalyst.git
cd innovation-catalyst
Install dependencies

pip install -r requirements.txt
Run the agent

python -m innovation_catalyst.app
```
## Architecture

The Innovation Catalyst Agent follows a modular architecture:

```
┌─────────────────┐ ┌──────────────────────┐ ┌─────────────────┐
│ Web Interface │───▶│ SmolAgent Core │───▶│ Tool Ecosystem│
│ (Gradio UI) │ │ (Orchestration) │ │ (Processing) │
└─────────────────┘ └──────────────────────┘ └─────────────────┘
```


## Development
```
Install development dependencies

pip install -e ".[dev]"
Run tests

pytest
Format code

black src/ tests/
Type checking

mypy src/
```


## License

MIT License - see LICENSE file for details.
"""
    
    def _get_pyproject_toml(self) -> str:
        """Generate pyproject.toml content."""
        return """[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "innovation-catalyst"
version = "0.1.0"
description = "AI Agent for discovering innovative connections between ideas"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = ["ai", "agent", "innovation", "nlp", "document-analysis"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
]

[tool.black]
line-length = 88
target-version = ['py38', 'py39', 'py310', 'py311']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--verbose",
    "--cov=src/innovation_catalyst",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-fail-under=80",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["src"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/conftest.py",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\bProtocol\\):",
    "@(abc\\.)?abstractmethod",
]
"""

if __name__ == "__main__":
    project = ProjectStructure(project_root="/Users/abhisheksuwalka/project/innovation-catalyst-ai/first-draft-HuggingFaceHub/innovation-catalyst-ai-agent")
    project.create_structure()
