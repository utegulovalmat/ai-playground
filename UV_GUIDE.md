# UV Quick Reference

UV is a fast Python package manager written in Rust. It's significantly faster than pip.

## Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## Common Commands

### Virtual Environment

```bash
# Create virtual environment
uv venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### Package Management

```bash
# Install a package
uv pip install package-name

# Install from requirements.txt
uv pip install -r requirements.txt

# Install specific version
uv pip install package-name==1.0.0

# Sync exact dependencies (recommended for reproducibility)
uv pip sync requirements.txt

# Uninstall a package
uv pip uninstall package-name

# List installed packages
uv pip list

# Show package info
uv pip show package-name
```

### For This Project

```bash
# Initial setup
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Add a new package
uv pip install new-package
uv pip freeze > requirements.txt  # Update requirements
```

## Why UV?

- ⚡ **10-100x faster** than pip
- 🔒 **Better dependency resolution**
- 🎯 **Drop-in replacement** for pip
- 🚀 **Written in Rust** for performance

## Learn More

- [UV Documentation](https://github.com/astral-sh/uv)
- [UV vs pip Comparison](https://github.com/astral-sh/uv#highlights)
