#!/usr/bin/env bash
# setup_pointcept_env_ucloud.sh
# One-time setup script for the Pointcept Pixi environment.

# Exit immediately if any command fails
set -e

# ANSI color codes for readable output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Starting Pointcept environment setup...${NC}\n"

# 1. Sanity Check: Ensure Pixi exists
if ! command -v pixi >/dev/null 2>&1; then
    echo -e "${RED}❌ ERROR: 'pixi' command not found.${NC}"
    echo "Make sure your init script ran and Pixi is in your PATH."
    exit 1
fi

# 2. Dependency Resolution & Download
echo -e "${GREEN}[1/3] Resolving and installing dependencies...${NC}"
pixi install

# 3. Compilation & Setup Hooks
echo -e "\n${GREEN}[2/3] Running post-install hooks (compiling extensions)...${NC}"
pixi run post-install

# 4. Hardware Verification
echo -e "\n${GREEN}[3/3] Running smoke tests...${NC}"
# Temporarily disable 'set -e' just for the test so we can catch the failure gracefully
set +e
pixi run smoke-test
TEST_RESULT=$?
set -e

if [ $TEST_RESULT -ne 0 ]; then
    echo -e "\n${RED}❌ Setup completed, but smoke tests failed!${NC}"
    echo "Check the logs above for CUDA/PyTorch binding issues."
    exit 1
fi

echo -e "\n${GREEN}✅ Setup complete and verified!${NC}\n"

# 5. Activation reminder (bypassing the 3-second timeout issue)
echo "To enter the environment, bypass the subshell timeout by running:"
echo "  'pixi run bash' or 'pixi run zsh'"
echo "Or inject it into your current session:"
echo "  eval \"\$(pixi shell-hook)\""
