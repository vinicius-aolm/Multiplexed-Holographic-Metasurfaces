# Contributing to Multiplexed Holographic Metasurfaces

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the codebase.

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Multiplexed-Holographic-Metasurfaces.git
   cd Multiplexed-Holographic-Metasurfaces
   ```
3. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 💻 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Optional Development Tools

```bash
pip install pytest black flake8
```

## 📁 Code Organization

The project follows a structured organization:

- **`src/`**: Production code organized by functionality
  - `cli/`: Command-line interface tools
  - `holography/`: Hologram generation algorithms
  - `dammann/`: Dammann grating generation
  - `meta_library/`: Metasurface library processing
  - `optimization/`: Optimization algorithms
  - `simulation/`: Simulation tools
  - `utils/`: Shared utilities

- **`notebooks/`**: Jupyter notebooks for exploration and tutorials
- **`tests/`**: Test suite
- **`data/`**: Data files (not all committed to git)
- **`results/`**: Generated outputs (typically not committed)

### Adding New Features

1. **For new algorithms**: Add to the appropriate module in `src/`
2. **For new CLI tools**: Add to `src/cli/` following existing patterns
3. **For tutorials**: Add notebooks to `notebooks/`
4. **For tests**: Add to `tests/`

## 🎨 Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and concise

### Example Function Documentation

```python
def calculate_phase_map(target_image: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Calculate phase map for target image using GS algorithm.
    
    Args:
        target_image: Target intensity pattern (normalized 0-1)
        wavelength: Wavelength in meters
        
    Returns:
        Phase map in radians (0 to 2π)
        
    Raises:
        ValueError: If target_image is not 2D or wavelength is non-positive
    """
    # Implementation
    pass
```

### CLI Tools Patterns

When creating new CLI tools, follow the established pattern:

1. Use `argparse` for command-line arguments
2. Create timestamped output directories
3. Save metadata as `run_meta.json`
4. Generate `README.md` for each run
5. Use logging for progress messages

Example structure:
```python
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Tool description")
    # Add arguments
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_args()
    # Setup output directory
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Run algorithm
    # Save results with metadata
    
if __name__ == "__main__":
    main()
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_meta_library.py

# Run with coverage
pytest --cov=src tests/
```

### Writing Tests

- Add tests for new functionality in `tests/`
- Use descriptive test names
- Test edge cases and error conditions
- Keep tests independent

Example test:
```python
def test_phase_wrapping():
    """Test that phase values are correctly wrapped to [0, 2π]."""
    phase_in = np.array([-np.pi, 0, np.pi, 2*np.pi, 3*np.pi])
    phase_out = wrap_phase(phase_in)
    assert np.all((phase_out >= 0) & (phase_out < 2*np.pi))
```

## 📚 Documentation

### Module Documentation

Each module should have a `README.md` explaining:

- Purpose and functionality
- Usage examples (CLI and Python API)
- Input/output formats
- Key parameters

### Code Comments

- Use docstrings for functions, classes, and modules
- Add inline comments for complex logic
- Keep comments up-to-date with code changes

### Notebooks

Educational notebooks should:

- Have clear markdown explanations
- Include both English and Portuguese text (bilingual)
- Show complete examples
- Include reproducibility information

## 📤 Submitting Changes

### Commit Messages

Write clear, descriptive commit messages:

```
Add phase unwrapping feature to library cleaning

- Implement phase unwrapping per nanopillar group
- Add --unwrap-phase CLI flag
- Update documentation and tests
```

### Pull Request Process

1. **Update documentation** for any changed functionality
2. **Add or update tests** as needed
3. **Ensure tests pass**: `pytest tests/`
4. **Update CHANGELOG.md** (if exists) with notable changes
5. **Create pull request** with description of changes

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
How has this been tested?

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] All tests pass
```

## 🔍 Code Review

All submissions require review. Reviewers will check:

- Code quality and style
- Test coverage
- Documentation completeness
- Compatibility with existing code

## ❓ Questions?

If you have questions about contributing:

1. Check existing issues and documentation
2. Open a new issue for discussion
3. Reach out to project maintainers

## 📜 License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

---

Thank you for contributing to Multiplexed Holographic Metasurfaces! 🎉
