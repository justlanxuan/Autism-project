#!/bin/bash

set -e  

CONDA_ENV_NAME="mmact"
PYTHON_VERSION="3.9"

echo "========== MMAct Setup =========="

if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "[1/3] Activating existing environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}
else
    echo "[1/3] Creating new environment..."
    conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION} -y > /dev/null
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate ${CONDA_ENV_NAME}
fi

if pip show pandas &> /dev/null; then
    echo "[2/3] Upgrading dependencies..."
    pip install --upgrade -r requirements.txt -q
else
    echo "[2/3] Installing dependencies..."
    pip install -r requirements.txt -q
fi

echo "✓ Setup complete"
echo "Environment: ${CONDA_ENV_NAME}"