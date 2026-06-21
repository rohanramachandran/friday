#!/bin/bash
cd "$(dirname "$0")/../daemon"
source .venv/bin/activate
exec python3 main.py
