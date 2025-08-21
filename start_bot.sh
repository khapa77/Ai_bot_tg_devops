#!/bin/bash

python3 -m venv agent
source ./agent/bin/activate
exec python main.py
