#!/usr/bin/env python3
"""
Configuration settings for EAGLE-2 SE OmniDraft project.
"""

# Repository configuration
REPO_URL = "https://github.com/tenet-diver/EAGLE-2_SE_OmniDraft.git"
PROJECT_DIR = "EAGLE-2_SE_OmniDraft"
SCRIPT_NAME = "heterogeneous_spd.py"

# Model configuration
TINY_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LARGE_MODEL_ID = "Qwen/Qwen3-0.6B"

# Execution parameters
DEFAULT_MAX_NEW_TOKENS = 50
DEFAULT_K = 32
DEFAULT_ALPHA = 0.15
SCRIPT_TIMEOUT = 300  # seconds

# Test configuration
TEST_PROMPT = "The future of artificial intelligence is"
