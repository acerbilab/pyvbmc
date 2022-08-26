#!/bin/bash
# Finds .py and .ini files to edit, in the appropriate directories, and
# writes them to files.
# Usage: ./find_files.sh
find ./pyvbmc/ \( -type d -name .git -prune \) -o -type f -and \( -name "*.py" -or -name "*.ini" \) > ./replace_files_pyvbmc.txt
find ./tests/ \( -type d -name .git -prune \) -o -type f -and \( -name "*.py" -or -name "*.ini" \) > ./replace_files_tests.txt
