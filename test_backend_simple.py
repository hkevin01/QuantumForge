#!/usr/bin/env python3
"""Simple test to verify backend adapter recovery without torch dependency."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports_only():
    """Test that backend adapters can be imported without torch."""
    print("Testing backend adapter imports (no torch)...")

    try:
        # This should work even without torch
        import quantumforge.core.backends.adapters as adapters
        print("✓ Successfully imported adapters module")

        # Test individual classes exist
        assert hasattr(adapters, 'BackendAdapterBase')
        assert hasattr(adapters, 'PySCFAdapter')
        assert hasattr(adapters, 'CP2KAdapter')
        assert hasattr(adapters, 'create_backend_adapter')
        print("✓ All expected classes and functions are available")

        # Test main package import
        from quantumforge.core.backends import create_backend_adapter
        print("✓ Package-level import works")

        # Test __all__ exports
        from quantumforge.core import backends
        expected_exports = [
            "BackendAdapterBase",
            "PySCFAdapter",
            "CP2KAdapter",
            "create_backend_adapter"
        ]

        for export in expected_exports:
            assert hasattr(backends, export), f"Missing export: {export}"
        print("✓ All expected exports are available")

        return True

    except Exception as e:
        print(f"✗ Import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all backend files exist and are readable."""
    print("\nTesting file structure...")

    backend_dir = Path(__file__).parent / "src/quantumforge/core/backends"

    expected_files = [
        "__init__.py",
        "adapters.py"
    ]

    for filename in expected_files:
        filepath = backend_dir / filename
        if not filepath.exists():
            print(f"✗ Missing file: {filepath}")
            return False
        print(f"✓ Found: {filename}")

    # Check that __init__.py is not empty and has reasonable size
    init_file = backend_dir / "__init__.py"
    init_size = init_file.stat().st_size
    if init_size < 50:  # Should have imports and __all__
        print(f"✗ __init__.py seems too small: {init_size} bytes")
        return False
    elif init_size > 1000:  # Should not be huge
        print(f"⚠ __init__.py seems large: {init_size} bytes")
    else:
        print(f"✓ __init__.py has reasonable size: {init_size} bytes")

    return True

if __name__ == "__main__":
    print("Backend Adapter File Recovery Test")
    print("=" * 45)

    success = True
    success &= test_file_structure()
    success &= test_imports_only()

    print("\n" + "=" * 45)
    if success:
        print("✓ Backend adapter file recovery successful!")
        print("  - Files exist and have correct structure")
        print("  - Imports work without dependency issues")
        print("  - Package exports are correctly configured")
    else:
        print("✗ Backend adapter recovery has issues.")

    sys.exit(0 if success else 1)
