#!/usr/bin/env bash
# ==============================================================================
# setup_mac.sh - Automated Setup and Verification for macOS Apple Silicon
# ==============================================================================
set -e

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
BOLD="\033[1m"
NC="\033[0m"

echo -e "${BOLD}${CYAN}=== Ollama BenchRig - macOS Apple Silicon Setup ===${NC}\n"

# 1. Check OS and architecture
OS=$(uname -s)
ARCH=$(uname -m)

if [ "$OS" != "Darwin" ]; then
    echo -e "${RED}✖ This script is intended for macOS. Detected: $OS${NC}"
    exit 1
fi

if [ "$ARCH" != "arm64" ]; then
    echo -e "${YELLOW}⚠ Warning: Detected architecture $ARCH instead of Apple Silicon (arm64).${NC}"
    echo -e "${YELLOW}  Metal acceleration and UMA memory telemetry operate optimally on M1/M2/M3/M4 chips.${NC}"
else
    CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon")
    RAM_BYTES=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    RAM_GB=$(( RAM_BYTES / 1024 / 1024 / 1024 ))
    echo -e "${GREEN}✔ Detected Apple Silicon:${NC} $CHIP (${RAM_GB} GB Unified Memory)"
fi

# 2. Check Python 3
if ! command -v python3 &>/dev/null; then
    echo -e "${RED}✖ python3 command not found. Please install Python 3 (e.g. via Homebrew: brew install python).${NC}"
    exit 1
fi
PY_VER=$(python3 --version)
echo -e "${GREEN}✔ Python environment:${NC} $PY_VER"

# 3. Virtual environment setup
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n${CYAN}Creating Python virtual environment ($VENV_DIR)...${NC}"
    python3 -m venv "$VENV_DIR"
fi

echo -e "${CYAN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# 4. Install dependencies
echo -e "\n${CYAN}Installing dependencies from requirements.txt...${NC}"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo -e "${GREEN}✔ Python dependencies installed successfully.${NC}"

# 5. Check Ollama installation
echo -e "\n${CYAN}Checking Ollama service...${NC}"
if ! command -v ollama &>/dev/null; then
    echo -e "${YELLOW}⚠ Ollama is not found in PATH.${NC}"
    echo -e "  You can install it via: ${BOLD}brew install ollama${NC}"
    echo -e "  or download the app from: ${BOLD}https://ollama.com/download${NC}"
fi

# 6. Verify Ollama daemon connectivity on port 11434
if curl -s -f "http://localhost:11434/api/version" &>/dev/null; then
    OLLAMA_VER=$(curl -s "http://localhost:11434/api/version" | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}✔ Ollama REST API is accessible (Version: ${OLLAMA_VER})${NC}"
else
    echo -e "${YELLOW}⚠ Ollama service is not responding at http://localhost:11434.${NC}"
    echo -e "  Please launch the Ollama app or start it in a separate terminal: ${BOLD}ollama serve${NC}"
fi

# 7. Run diagnostic environment check
echo -e "\n${BOLD}${CYAN}Running environment diagnostics (benchmark.py --check):${NC}\n"
python3 benchmark.py --check || true

echo -e "\n${BOLD}${GREEN}=== Setup Complete! macOS environment is configured ===${NC}"
echo -e "To run a benchmark, execute:"
echo -e "  ${BOLD}source .venv/bin/activate${NC}"
echo -e "  ${BOLD}python3 benchmark.py --models qwen2.5-coder:7b,llama3.1:8b${NC}\n"
