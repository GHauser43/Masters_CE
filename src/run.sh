#!/bin/bash

# for argument help on NGRC.py: ./run.sh help
# to run NGRC.py: ./run.sh

if [ "$1" = "help" ]; then
    python NGRC.py --help
    exit 0
fi
echo "||"
echo "|| running"
echo "||"

python NGRC.py > output.txt

echo "||"
echo "|| finished"
echo "||"
