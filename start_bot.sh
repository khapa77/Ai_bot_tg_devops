#!/bin/bash

python3 -m venv agent
source ./agent/bin/activate
exec python ii_sysadmin_v1.py
