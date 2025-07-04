# src/innovation_catalyst/__init__.py
"""
Initialization for the Innovation Catalyst package.

This file ensures that environment variables are loaded from the .env file
as soon as any module from this package is imported. This provides a single,
reliable point for environment configuration.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

import nltk
nltk.download('punkt_tab')

import time

# Determine the project root to find the .env file reliably
# This assumes your .env file is in the project root, one level above 'src'
project_root = Path(__file__).parent.parent.parent
dotenv_path = project_root / '.env'

if dotenv_path.exists():
    load_dotenv(dotenv_path=dotenv_path)
    logging.info(f"Loaded environment variables from: {dotenv_path}")
else:
    logging.warning(f".env file not found at {dotenv_path}. Relying on system environment variables.")