#!/usr/bin/env python3
"""
Manual script to run all quality checks before committing.
This script mimics what pre-commit hooks do automatically.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print("-" * 50)

    try:
        result = subprocess.run(
            cmd.split(),
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr.strip():
                print("Error:", result.stderr)
            if result.stdout.strip():
                print("Output:", result.stdout)
            return False

    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main() -> None:
    """Run all quality checks."""
    print("🎯 Running Pre-Commit Quality Checks")
    print("=" * 60)

    checks = [
        ("uv run ruff check src tests", "Code Linting (ruff)"),
        ("uv run ruff format --check src tests", "Code Formatting (ruff)"),
        ("uv run mypy src", "Type Checking (mypy)"),
        ("python -m pytest tests/ -v", "Unit Tests (pytest)"),
    ]

    all_passed = True
    passed_checks = []
    failed_checks = []

    for cmd, description in checks:
        if run_command(cmd, description):
            passed_checks.append(description)
        else:
            failed_checks.append(description)
            all_passed = False

    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS")
    print("=" * 60)

    print(f"\n✅ PASSED ({len(passed_checks)}):")
    for check in passed_checks:
        print(f"  • {check}")

    if failed_checks:
        print(f"\n❌ FAILED ({len(failed_checks)}):")
        for check in failed_checks:
            print(f"  • {check}")

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED! Ready to commit! 🚀")
        print("\nYou can now safely run:")
        print("  git add .")
        print("  git commit -m 'your commit message'")
        print("  git push origin main")
        sys.exit(0)
    else:
        print("\n⚠️  Some checks failed. Please fix the issues before committing.")
        print("\nTo fix common issues:")
        print("  • Format code: uv run black src tests")
        print("  • Fix imports: uv run isort src tests")
        print("  • Check specific errors above")
        sys.exit(1)


if __name__ == "__main__":
    main()
