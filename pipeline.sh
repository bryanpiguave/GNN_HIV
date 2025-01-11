#!/bin/bash
# This script is used to run the entire pipeline
cd src
python data_processing.py
cd ..
python training.py