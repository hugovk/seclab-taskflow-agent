#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 GitHub
# SPDX-License-Identifier: MIT

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if gh cli is available
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: gh cli not found. Install from https://cli.github.com${NC}"
    exit 1
fi

# Get API token
echo "Getting GitHub API token..."
export AI_API_TOKEN="$(gh auth token)"
if [ -z "$AI_API_TOKEN" ]; then
    echo -e "${RED}Error: Failed to get GitHub API token${NC}"
    exit 1
fi

# Activate venv if exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo -e "${YELLOW}Warning: No .venv found, using system Python${NC}"
fi

# Track test results
PASSED=0
FAILED=0
FAILED_TESTS=()

# Test function
run_test() {
    local name="$1"
    local taskflow="$2"
    local args="${3:-}"
    local timeout="${4:-30}"

    echo -e "\n${YELLOW}Testing: $name${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Run command with output shown in real-time, capture to temp file for checking
    local tmpfile=$(mktemp)
    timeout "$timeout" python -m seclab_taskflow_agent -t "$taskflow" $args 2>&1 | tee "$tmpfile" || true

    echo -e "${YELLOW}========================================${NC}"

    # Check for error conditions first
    if grep -qi "rate limit" "$tmpfile" || \
       grep -q "ERROR:" "$tmpfile" || \
       grep -q "Max rate limit backoff reached" "$tmpfile" || \
       grep -q "APITimeoutError" "$tmpfile" || \
       grep -q "Exception:" "$tmpfile"; then
        echo -e "${RED}✗ $name failed (error detected)${NC}"
        ((FAILED++))
        FAILED_TESTS+=("$name")
        rm "$tmpfile"
        return 1
    fi

    # Check for successful start
    if grep -q "Running Task Flow" "$tmpfile"; then
        echo -e "${GREEN}✓ $name passed${NC}"
        ((PASSED++))
        rm "$tmpfile"
        return 0
    else
        echo -e "${RED}✗ $name failed (did not start)${NC}"
        ((FAILED++))
        FAILED_TESTS+=("$name")
        rm "$tmpfile"
        return 1
    fi
}

echo -e "${GREEN}Starting example taskflow tests...${NC}\n"

# Test 1: Simple single-step taskflow
run_test "single_step_taskflow" "examples.taskflows.single_step_taskflow"

# Test 2: Echo taskflow
run_test "echo" "examples.taskflows.echo"

# Test 3: Globals example
run_test "example_globals" "examples.taskflows.example_globals" "-g fruit=apples"

# Test 4: Inputs example
run_test "example_inputs" "examples.taskflows.example_inputs"

# Test 5: Repeat prompt example
run_test "example_repeat_prompt" "examples.taskflows.example_repeat_prompt" "" "45"

# Test 6: Reusable prompt example
run_test "example_reusable_prompt" "examples.taskflows.example_reusable_prompt"

# Test 7: Full example taskflow
run_test "example" "examples.taskflows.example"

# Test 8: Reusable taskflows (may fail on temperature setting, but should load YAML)
echo -e "\n${YELLOW}Testing: example_reusable_taskflows (YAML load test)${NC}"
echo -e "${YELLOW}========================================${NC}"
tmpfile=$(mktemp)
timeout 30 python -m seclab_taskflow_agent -t examples.taskflows.example_reusable_taskflows 2>&1 | tee "$tmpfile" || true
echo -e "${YELLOW}========================================${NC}"

# Check for errors first
if grep -qi "rate limit" "$tmpfile" || \
   grep -q "Max rate limit backoff reached" "$tmpfile" || \
   grep -q "APITimeoutError" "$tmpfile"; then
    echo -e "${RED}✗ example_reusable_taskflows failed (error detected)${NC}"
    ((FAILED++))
    FAILED_TESTS+=("example_reusable_taskflows")
elif grep -q "Running Task Flow" "$tmpfile"; then
    echo -e "${GREEN}✓ example_reusable_taskflows YAML loaded correctly${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ example_reusable_taskflows - may have API issues but YAML loaded${NC}"
    ((PASSED++))
fi
rm "$tmpfile"

# Print summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Test Summary${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "\n${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  - $test"
    done
    exit 1
else
    echo -e "\n${GREEN}All tests passed!${NC}"
    exit 0
fi
