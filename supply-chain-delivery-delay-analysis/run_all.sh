#!/usr/bin/env bash
set -e
python src/generate_data.py
python src/clean_data.py
python src/analysis.py > /dev/null
python src/run_sql.py
echo "Done. See images/ and reports/"
