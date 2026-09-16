"""
pytest configuration — adds the project root to sys.path so all imports work.
"""

import sys
import os

# Ensure the project root is in the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
