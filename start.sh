#!/bin/bash

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting the app..."
python app.py
