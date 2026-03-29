"""
Runtime module - loads and runs .paw programs.

Currently uses PyTorch backend.
ONNX backend available but commented out for simplicity.
"""

# Use PyTorch runtime
from .interpreter import function

# ONNX runtime (uncomment when ready):
# import os
# _runtime = os.getenv('PROGRAMASWEIGHTS_RUNTIME', 'pytorch').lower()
# if _runtime == 'onnx':
#     try:
#         from .interpreter_onnx import function
#     except Exception:
#         from .interpreter import function
# else:
#     from .interpreter import function

__all__ = ['function']
