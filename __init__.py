"""Top-level package for cui-generalized-guidance-forms."""

__all__ = [
    "CUIGeneralizedGuidanceForms",
    "comfy_entrypoint",
]

__author__ = """xxiiyu"""
__email__ = ""
__version__ = "0.0.1"

from .src.cui_generalized_guidance_forms import (
    CUIGeneralizedGuidanceForms,
    comfy_entrypoint,
)

print(
    "\033[92m"
    + "[cui-ggf] Thank you for installing CUI-GeneralizedGuidanceForms!"
    + "\033[0m"
)
