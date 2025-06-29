"""
Typing compatibility fix for Python 3.12+ and Dask/Distributed
"""

import sys
import warnings

# Apply typing compatibility fixes
def apply_typing_fixes():
    """Apply fixes for Python 3.12+ typing compatibility issues."""
    
    # Fix for the 'unhashable type: list' error in distributed
    if sys.version_info >= (3, 12):
        import typing
        
        # Monkey patch the Union type to handle lists properly
        original_union = typing.Union
        
        def safe_union(*args):
            try:
                return original_union(*args)
            except TypeError as e:
                if "unhashable type: 'list'" in str(e):
                    # Convert lists to tuples for hashing
                    processed_args = []
                    for arg in args:
                        if isinstance(arg, list):
                            processed_args.append(tuple(arg))
                        else:
                            processed_args.append(arg)
                    return original_union(*processed_args)
                else:
                    raise
        
        # Apply the fix
        typing.Union = safe_union
        
        # Also fix TypeAlias if needed
        if hasattr(typing, 'TypeAlias'):
            original_type_alias = typing.TypeAlias
            
            def safe_type_alias(*args, **kwargs):
                try:
                    return original_type_alias(*args, **kwargs)
                except TypeError as e:
                    if "unhashable type: 'list'" in str(e):
                        warnings.warn("TypeAlias compatibility issue detected, using fallback")
                        return object  # Fallback to object
                    else:
                        raise
            
            typing.TypeAlias = safe_type_alias

# Apply fixes immediately when imported
apply_typing_fixes() 