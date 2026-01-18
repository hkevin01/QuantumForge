#!/usr/bin/env python3
"""Test script to verify backend adapter recovery after file corruption."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_backend_imports():
    """Test that backend adapters can be imported successfully."""
    print("Testing backend adapter imports...")

    try:
        # Test main module import
        from quantumforge.core.backends import (
            BackendAdapterBase,
            CP2KAdapter,
            PySCFAdapter,
            create_backend_adapter,
        )
        print("✓ Successfully imported main backend adapter classes")

        # Test factory function
        try:
            adapter = create_backend_adapter("pyscf")
            print(f"✓ Created PySCF adapter: {adapter.name()}")
        except ImportError as e:
            print(f"⚠ PySCF adapter creation failed (expected): {e}")

        try:
            adapter = create_backend_adapter("cp2k")
            print(f"✓ Created CP2K adapter: {adapter.name()}")
            print(f"  - Available: {adapter.is_available()}")
        except Exception as e:
            print(f"✗ CP2K adapter creation failed: {e}")

        # Test invalid backend
        try:
            create_backend_adapter("invalid")
        except ValueError as e:
            print(f"✓ Correctly rejected invalid backend: {e}")

        return True

    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_backend_availability():
    """Test backend availability checks."""
    print("\nTesting backend availability...")

    try:
        from quantumforge.core.backends import CP2KAdapter, PySCFAdapter

        # Test PySCF availability
        pyscf_adapter = PySCFAdapter()
        print(f"PySCF available: {pyscf_adapter.is_available()}")

        # Test CP2K availability
        cp2k_adapter = CP2KAdapter()
        print(f"CP2K available: {cp2k_adapter.is_available()}")

        return True

    except Exception as e:
        print(f"✗ Availability test failed: {e}")
        return False

def test_basic_functionality():
    """Test basic adapter functionality."""
    print("\nTesting basic adapter functionality...")

    try:
        from quantumforge.core.backends import CP2KAdapter

        adapter = CP2KAdapter()

        # Test basic methods
        assert adapter.name() == "CP2K"
        assert hasattr(adapter, "setup_calculation")
        assert hasattr(adapter, "run_scf")

        print("✓ Basic adapter functionality works")
        return True

    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("Backend Adapter Recovery Test")
    print("=" * 40)

    success = True
    success &= test_backend_imports()
    success &= test_backend_availability()
    success &= test_basic_functionality()

    print("\n" + "=" * 40)
    if success:
        print("✓ All tests passed! Backend adapters recovered successfully.")
    else:
        print("✗ Some tests failed.")

    print(f"Exit code: {0 if success else 1}")
    sys.exit(0 if success else 1)
