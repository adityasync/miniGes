"""Placeholder tests for the sinGes-mini project.

These tests are intentionally lightweight and exist solely to verify that the test
infrastructure resolves imports correctly. Replace them with meaningful tests as
soon as the corresponding modules are implemented.
"""

from pathlib import Path


def test_project_structure_exists():
    """Ensure the core directories defined in the roadmap exist."""

    expected_directories = [
        Path("data/raw"),
        Path("data/processed"),
        Path("models/checkpoints/sign_recognition"),
        Path("models/checkpoints/transformer"),
        Path("src"),
        Path("app"),
        Path("config"),
    ]

    missing = [directory for directory in expected_directories if not directory.exists()]
    assert not missing, f"Missing directories: {missing}"


def test_requirements_file_created():
    """Verify that the dependency list is present and non-empty."""

    requirements = Path("requirements.txt")
    assert requirements.exists(), "requirements.txt is missing"
    assert requirements.read_text().strip(), "requirements.txt should not be empty"
