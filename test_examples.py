#!/usr/bin/env python3
"""
Test Script - Verify all example files are syntactically correct
"""

import os
import py_compile
import sys
from pathlib import Path

def test_file(file_path):
    """Test if a Python file compiles without syntax errors."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    """Test all Python files in examples directory."""
    examples_dir = Path("examples")
    
    # Find all Python files
    python_files = sorted(examples_dir.rglob("*.py"))
    
    print("=" * 60)
    print("Testing Example Files for Syntax Errors")
    print("=" * 60)
    print()
    
    passed = []
    failed = []
    
    for file_path in python_files:
        relative_path = file_path.relative_to(examples_dir)
        print(f"Testing {relative_path}...", end=" ")
        
        success, error = test_file(file_path)
        
        if success:
            print("✓ PASS")
            passed.append(relative_path)
        else:
            print("✗ FAIL")
            print(f"  Error: {error}")
            failed.append(relative_path)
    
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Total files: {len(python_files)}")
    print(f"Passed: {len(passed)} ✓")
    print(f"Failed: {len(failed)} ✗")
    
    if failed:
        print("\nFailed files:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("\n✓ All files passed syntax check!")
        sys.exit(0)

if __name__ == "__main__":
    main()
