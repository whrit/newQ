#!/bin/bash
# start_trading.sh

# Load environment variables
set -a
source .env
set +a

# Start trading system
python main.py