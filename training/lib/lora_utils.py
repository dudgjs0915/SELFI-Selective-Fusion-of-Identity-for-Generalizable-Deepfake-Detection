"""
LoRA (Low-Rank Adaptation) utilities for two-phase training.
Supports adding LoRA layers to pre-trained models and managing trainable parameters.
"""

import torch
import torch.nn as nn
import loralib as lora
import logging

logger = logging.getLogger(__name__)


def replace_linear_with_lora(model, r=8, lora_alpha=16, lora_dropout=0.1, target_modules=None):
    """
    Replace nn.Linear layers with LoRA layers.
    
    Args:
        model: PyTorch model to modify
        r: LoRA rank (default: 8)
        lora_alpha: LoRA alpha scaling parameter (default: 16)
        lora_dropout: Dropout probability for LoRA layers (default: 0.1)
        target_modules: List of module name patterns to apply LoRA to.
                       If None, applies to all Linear layers in the model.
    
    Returns:
        Modified model with LoRA layers
    """
    replaced_count = 0
    
    def _replace_module(parent_module, name):
        nonlocal replaced_count
        module = getattr(parent_module, name)
        
        if isinstance(module, nn.Linear) and not isinstance(module, lora.Linear):
            # Check if this module should be replaced
            if target_modules is not None:
                full_name = get_module_name(model, module)
                should_replace = any(pattern in full_name for pattern in target_modules)
                if not should_replace:
                    return
            
            # Get the device of the original module
            device = next(module.parameters()).device
            
            # Create LoRA layer with the same dimensions
            lora_layer = lora.Linear(
                module.in_features,
                module.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=module.bias is not None
            )
            
            # Move LoRA layer to the same device as the original module
            lora_layer = lora_layer.to(device)
            
            # Copy weights and bias from original layer
            with torch.no_grad():
                lora_layer.weight.copy_(module.weight)
                if module.bias is not None:
                    lora_layer.bias.copy_(module.bias)
            
            # Replace the module
            setattr(parent_module, name, lora_layer)
            replaced_count += 1
            logger.debug(f"Replaced {name} with LoRA layer")
    
    # Recursively replace Linear layers
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            _replace_module(model, name)
        else:
            replace_linear_with_lora(child, r, lora_alpha, lora_dropout, target_modules)
    
    logger.info(f"Replaced {replaced_count} Linear layers with LoRA layers")
    return model


def get_module_name(model, target_module):
    """
    Get the full name of a module within a model.
    
    Args:
        model: PyTorch model
        target_module: Module to find
    
    Returns:
        Full module name as string
    """
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return ""


def freeze_non_lora_params(model):
    """
    Freeze all parameters except LoRA parameters.
    
    Args:
        model: PyTorch model with LoRA layers
    
    Returns:
        Number of frozen parameters, number of trainable parameters
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        # LoRA parameters typically have 'lora_' in their name
        if 'lora_' in name:
            param.requires_grad = True
            trainable_count += param.numel()
            logger.debug(f"Trainable (LoRA): {name}")
        else:
            param.requires_grad = False
            frozen_count += param.numel()
    
    logger.info(f"Frozen {frozen_count:,} non-LoRA parameters")
    logger.info(f"Trainable {trainable_count:,} LoRA parameters")
    logger.info(f"Trainable ratio: {trainable_count / (frozen_count + trainable_count) * 100:.2f}%")
    
    return frozen_count, trainable_count


def unfreeze_all_params(model):
    """
    Unfreeze all parameters in the model.
    
    Args:
        model: PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True
    logger.info("All parameters unfrozen")


def mark_only_lora_as_trainable(model, bias='none'):
    """
    Mark only LoRA parameters as trainable.
    This is an alternative implementation using loralib's built-in functionality.
    
    Args:
        model: PyTorch model with LoRA layers
        bias: How to handle bias terms ('none', 'all', or 'lora_only')
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then, unfreeze LoRA parameters
    trainable_params = 0
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        elif bias == 'all' and 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()
        elif bias == 'lora_only' and 'lora_' in name and 'bias' in name:
            param.requires_grad = True
            trainable_params += param.numel()
    
    logger.info(f"Marked {trainable_params:,} parameters as trainable")


def merge_lora_weights(model):
    """
    Merge LoRA weights into the base model weights.
    This creates a single model without LoRA layers.
    
    Args:
        model: PyTorch model with LoRA layers
    
    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, lora.Linear):
            # loralib's Linear has a merge_weights method
            if hasattr(module, 'merge'):
                module.merge()
    
    logger.info("LoRA weights merged into base model")
    return model


def print_trainable_parameters(model):
    """
    Print statistics about trainable parameters.
    
    Args:
        model: PyTorch model
    """
    trainable_params = 0
    all_param = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'lora_' in name:
                lora_params += param.numel()
    
    logger.info(
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}% || "
        f"LoRA params: {lora_params:,}"
    )
    
    return {
        'trainable_params': trainable_params,
        'all_params': all_param,
        'trainable_percentage': 100 * trainable_params / all_param,
        'lora_params': lora_params
    }


def get_lora_state_dict(model):
    """
    Get only the LoRA parameters from the model.
    
    Args:
        model: PyTorch model with LoRA layers
    
    Returns:
        Dictionary of LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.named_parameters():
        if 'lora_' in name:
            lora_state_dict[name] = param
    
    return lora_state_dict


def load_lora_weights(model, lora_checkpoint_path):
    """
    Load LoRA weights from a checkpoint.
    
    Args:
        model: PyTorch model with LoRA layers
        lora_checkpoint_path: Path to LoRA checkpoint
    
    Returns:
        Model with loaded LoRA weights
    """
    checkpoint = torch.load(lora_checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Load only LoRA parameters
    lora_params = {k: v for k, v in state_dict.items() if 'lora_' in k}
    
    # Load the weights (strict=False to allow partial loading)
    missing_keys, unexpected_keys = model.load_state_dict(lora_params, strict=False)
    
    logger.info(f"Loaded LoRA weights from {lora_checkpoint_path}")
    logger.info(f"Loaded {len(lora_params)} LoRA parameters")
    
    return model
