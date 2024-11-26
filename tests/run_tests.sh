#!/bin/bash
# tests/run_tests.sh

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Starting trading system tests..."

# Create results directory
mkdir -p test_results
timestamp=$(date +%Y%m%d_%H%M%S)
result_dir="test_results/test_run_${timestamp}"
mkdir -p "${result_dir}"

# Run system validation
echo -e "\n${GREEN}Running system validation...${NC}"
python tests/validate_system.py 2>&1 | tee "${result_dir}/validation.log"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}System validation passed!${NC}"
    
    # Run test trading session
    echo -e "\n${GREEN}Running test trading session...${NC}"
    python tests/test_trading_session.py 2>&1 | tee "${result_dir}/trading_session.log"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Test trading session completed successfully!${NC}"
    else
        echo -e "${RED}Test trading session failed! Check ${result_dir}/trading_session.log for details${NC}"
        exit 1
    fi
else
    echo -e "${RED}System validation failed! Check ${result_dir}/validation.log for details${NC}"
    exit 1
fi

# Copy test results
cp test_trades_*.csv "${result_dir}/" 2>/dev/null || true
cp test_metrics_*.csv "${result_dir}/" 2>/dev/null || true

echo -e "\n${GREEN}All tests completed. Results saved in ${result_dir}${NC}"