import os
import sys
import py_compile
import traceback

def check_python_file(filepath):
    try:
        py_compile.compile(filepath, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)
    except Exception as e:
        return False, traceback.format_exc()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(base_dir, 'src')

    print("=" * 70)
    print("Defect Detection Service - Python Syntax Check")
    print("=" * 70)
    print()

    py_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    py_files.sort()

    print(f"Found {len(py_files)} Python files to check")
    print()

    passed = 0
    failed = 0

    for filepath in py_files:
        rel_path = os.path.relpath(filepath, base_dir)
        print(f"Checking: {rel_path:<50} ", end="")

        success, error = check_python_file(filepath)
        if success:
            print("✓ PASS")
            passed += 1
        else:
            print("✗ FAIL")
            failed += 1
            if error:
                print(f"  Error: {error}")
            print()

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed > 0:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
