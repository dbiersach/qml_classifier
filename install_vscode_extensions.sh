#!/usr/bin/env bash
# Install the VS Code extensions used by the qml_classifier workspace.
# Run from a terminal where the `code` command is available:
#   bash install_vscode_extensions.sh

# Linting and formatting (Ruff is the default formatter for this workspace)
code --install-extension charliermarsh.ruff --force

# Python language support
code --install-extension ms-python.python --force
code --install-extension ms-python.vscode-pylance --force
code --install-extension ms-python.debugpy --force

# Jupyter notebooks
code --install-extension ms-toolsai.jupyter --force
code --install-extension ms-toolsai.jupyter-keymap --force
code --install-extension ms-toolsai.jupyter-renderers --force
code --install-extension ms-toolsai.vscode-jupyter-cell-tags --force
code --install-extension ms-toolsai.vscode-jupyter-slideshow --force

# Markdown (README.md, future_quantum_advantage.md) and general formatting
code --install-extension davidanson.vscode-markdownlint --force
code --install-extension esbenp.prettier-vscode --force

# GitHub Copilot Chat (this repo ships .github/copilot-instructions.md)
code --install-extension github.copilot-chat --force

# Quality-of-life helpers
code --install-extension streetsidesoftware.code-spell-checker --force
code --install-extension usernamehw.errorlens --force
code --install-extension oderwat.indent-rainbow --force
code --install-extension emmanuelbeziat.vscode-great-icons --force

# PowerShell support (for the .ps1 install script)
code --install-extension ms-vscode.powershell --force

# Remove extensions that conflict with this Ruff + uv (.venv) workflow
code --uninstall-extension ms-python.isort --force
code --uninstall-extension ms-python.vscode-python-envs --force
