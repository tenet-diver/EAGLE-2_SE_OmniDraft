#!/usr/bin/env python3
"""
Setup utilities for EAGLE-2 SE OmniDraft project.
Handles environment setup, model loading, and script execution.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class ProjectSetup:
    """Handles project setup and configuration."""
    
    def __init__(self, project_dir: str = "EAGLE-2_SE_OmniDraft"):
        self.project_dir = project_dir
        self.script_name = "heterogeneous_spd.py"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def check_environment(self) -> Dict[str, Any]:
        """Check Python environment and dependencies."""
        env_info = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "cuda_available": torch.cuda.is_available(),
            "device": self.device
        }
        
        if torch.cuda.is_available():
            env_info.update({
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count()
            })
            
        return env_info
    
    def clone_repository(self, repo_url: str) -> bool:
        """Clone repository if it doesn't exist."""
        if os.path.exists(self.project_dir):
            print(f"✅ Repository directory '{self.project_dir}' already exists")
            return True
            
        print(f"Cloning repository from {repo_url}...")
        result = subprocess.run(
            ["git", "clone", repo_url], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Repository cloned successfully")
            return True
        else:
            print(f"❌ Error cloning repository: {result.stderr}")
            return False
    
    def install_dependencies(self) -> bool:
        """Install dependencies from requirements.txt."""
        requirements_path = "requirements.txt"
        
        if not os.path.exists(requirements_path):
            print(f"❌ {requirements_path} not found")
            return False
            
        print("Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", requirements_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Error installing dependencies: {result.stderr}")
            return False
    
    def verify_imports(self) -> bool:
        """Verify that required packages can be imported."""
        try:
            import torch
            import transformers
            import numpy as np
            
            print(f"✅ PyTorch version: {torch.__version__}")
            print(f"✅ Transformers version: {transformers.__version__}")
            print(f"✅ NumPy version: {np.__version__}")
            return True
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False


class ModelManager:
    """Handles model loading and verification."""
    
    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tiny_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.large_id = "Qwen/Qwen3-0.6B"
        
    def load_tiny_model(self) -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]:
        """Load TinyLlama model and tokenizer."""
        try:
            print("Loading TinyLlama tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.tiny_id)
            print("✅ TinyLlama tokenizer loaded")
            
            print("Loading TinyLlama model...")
            model = AutoModelForCausalLM.from_pretrained(self.tiny_id).to(self.device).eval()
            print("✅ TinyLlama model loaded")
            
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Error loading TinyLlama: {e}")
            return None, None
    
    def load_large_model(self) -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]:
        """Load Meta-Llama model and tokenizer."""
        try:
            print("Loading Meta-Llama tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(self.large_id)
            print("✅ Meta-Llama tokenizer loaded")
            
            print("Loading Meta-Llama model (this may take a while)...")
            model = AutoModelForCausalLM.from_pretrained(self.large_id).to(self.device).eval()
            print("✅ Meta-Llama model loaded")
            
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Error loading Meta-Llama: {e}")
            print("Note: You may need to:")
            print("1. Accept the license agreement on HuggingFace")
            print("2. Login with: huggingface-cli login")
            print("3. Or use a different model ID")
            return None, None
    
    def test_tokenization(self, tokenizer: AutoTokenizer, model_name: str) -> bool:
        """Test tokenizer with sample text."""
        try:
            test_text = "Hello, world!"
            tokens = tokenizer.encode(test_text)
            print(f"Test tokenization ({model_name}): '{test_text}' -> {tokens}")
            return True
        except Exception as e:
            print(f"❌ Tokenization test failed for {model_name}: {e}")
            return False


class ScriptExecutor:
    """Handles script execution and module importing."""
    
    def __init__(self, script_name: str = "heterogeneous_spd.py"):
        self.script_name = script_name
        
    def check_script_exists(self) -> bool:
        """Check if the target script exists."""
        if os.path.exists(self.script_name):
            script_path = Path(self.script_name)
            stat = script_path.stat()
            print(f"✅ Found script: {self.script_name}")
            print(f"   Size: {stat.st_size} bytes")
            return True
        else:
            print(f"❌ Script not found: {self.script_name}")
            print("Available files:")
            for file in os.listdir("."):
                print(f"  - {file}")
            return False
    
    def execute_script(self, timeout: int = 300) -> Tuple[bool, str, str]:
        """Execute the script as subprocess."""
        if not self.check_script_exists():
            return False, "", "Script not found"
            
        print(f"Executing {self.script_name}...")
        print("=" * 50)
        
        try:
            result = subprocess.run(
                [sys.executable, self.script_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            success = result.returncode == 0
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            return False, "", f"Script execution timed out after {timeout} seconds"
        except Exception as e:
            return False, "", f"Error executing script: {e}"
    
    def import_script_module(self) -> Optional[Any]:
        """Import the script as a module for interactive use."""
        if not self.check_script_exists():
            return None
            
        try:
            spec = importlib.util.spec_from_file_location(
                "heterogeneous_spd", 
                self.script_name
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            print("✅ Script imported successfully")
            print("Available functions:")
            for attr in dir(module):
                if not attr.startswith('_') and callable(getattr(module, attr)):
                    print(f"  - {attr}")
                    
            return module
            
        except Exception as e:
            print(f"❌ Error importing script: {e}")
            return None
    
    def run_interactive_test(self, module: Any, prompt: str = "The future of artificial intelligence is") -> Optional[str]:
        """Run an interactive test with the imported module."""
        try:
            print(f"Running interactive test with prompt: '{prompt}'")
            
            result = module.heterogeneous_spec_decode(
                prompt=prompt,
                max_new_tokens=50,
                K=32,
                alpha=0.15
            )
            
            print(f"Generated text: {result}")
            print("✅ Interactive execution completed")
            return result
            
        except Exception as e:
            print(f"❌ Error in interactive execution: {e}")
            return None
