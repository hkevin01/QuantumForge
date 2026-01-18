#!/usr/bin/env python3
"""Direct test of backend adapter files without package imports."""

import ast
import sys
from pathlib import Path


def test_file_syntax():
    """Test that all backend files have valid Python syntax."""
    print("Testing file syntax...")

    backend_dir = Path(__file__).parent / "src/quantumforge/core/backends"

    files_to_test = ["__init__.py", "adapters.py"]

    for filename in files_to_test:
        filepath = backend_dir / filename

        if not filepath.exists():
            print(f"✗ File not found: {filename}")
            return False

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            # Test that it parses as valid Python
            ast.parse(content)

            print(f"✓ {filename}: Valid Python syntax")

            # Additional checks
            if filename == "__init__.py":
                if "__all__" not in content:
                    print(f"⚠ {filename}: Missing __all__ declaration")
                else:
                    print(f"✓ {filename}: Has __all__ declaration")

                if "from .adapters import" not in content:
                    print(f"⚠ {filename}: Missing adapters import")
                else:
                    print(f"✓ {filename}: Has adapters import")

            elif filename == "adapters.py":
                if "class BackendAdapterBase" not in content:
                    print(f"✗ {filename}: Missing BackendAdapterBase class")
                    return False
                else:
                    print(f"✓ {filename}: Has BackendAdapterBase class")

                if "class PySCFAdapter" not in content:
                    print(f"✗ {filename}: Missing PySCFAdapter class")
                    return False
                else:
                    print(f"✓ {filename}: Has PySCFAdapter class")

                if "def create_backend_adapter" not in content:
                    print(f"✗ {filename}: Missing create_backend_adapter function")
                    return False
                else:
                    print(f"✓ {filename}: Has create_backend_adapter function")

        except SyntaxError as e:
            print(f"✗ {filename}: Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            print(f"✗ {filename}: Error reading file: {e}")
            return False

    return True

def test_file_sizes():
    """Test that files have reasonable sizes (not corrupted)."""
    print("\nTesting file sizes...")

    backend_dir = Path(__file__).parent / "src/quantumforge/core/backends"

    # Expected reasonable sizes (in bytes)
    expected_sizes = {
        "__init__.py": (100, 500),  # Should be small
        "adapters.py": (5000, 20000),  # Should be substantial but not huge
    }

    for filename, (min_size, max_size) in expected_sizes.items():
        filepath = backend_dir / filename

        if not filepath.exists():
            print(f"✗ File not found: {filename}")
            return False

        size = filepath.stat().st_size

        if size < min_size:
            print(f"✗ {filename}: Too small ({size} bytes, expected >= {min_size})")
            return False
        elif size > max_size:
            print(f"⚠ {filename}: Very large ({size} bytes, expected <= {max_size})")
        else:
            print(f"✓ {filename}: Good size ({size} bytes)")

    return True

def test_no_corruption_markers():
    """Test that files don't contain corruption markers."""
    print("\nTesting for corruption markers...")

    backend_dir = Path(__file__).parent / "src/quantumforge/core/backends"

    corruption_markers = [
        "unterminated string literal",
        "unexpected indent",
        "SyntaxError:",
        "String literal is unterminated",
        "class BackendAdapterBase(abc.ABC):\n    class BackendAdapterBase",  # Duplication
    ]

    for filename in ["__init__.py", "adapters.py"]:
        filepath = backend_dir / filename

        if not filepath.exists():
            print(f"✗ File not found: {filename}")
            return False

        try:
            with open(filepath, 'r') as f:
                content = f.read()

            for marker in corruption_markers:
                if marker in content:
                    print(f"✗ {filename}: Found corruption marker: {marker[:50]}...")
                    return False

            print(f"✓ {filename}: No corruption markers found")

        except Exception as e:
            print(f"✗ {filename}: Error reading file: {e}")
            return False

    return True

if __name__ == "__main__":
    print("Backend File Integrity Test")
    print("=" * 35)

    success = True
    success &= test_file_syntax()
    success &= test_file_sizes()
    success &= test_no_corruption_markers()

    print("\n" + "=" * 35)
    if success:
        print("✓ All backend files are intact and uncorrupted!")
        print("  - Valid Python syntax")
        print("  - Reasonable file sizes")
        print("  - No corruption markers")
        print("  - Required classes and functions present")
    else:
        print("✗ Backend file integrity issues detected.")

    sys.exit(0 if success else 1)
