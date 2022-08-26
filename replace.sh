#!/bin/bash
# Replaces specified strings with snake_case according to
# sorted_replacements.txt (order matters).
# Usage: ./replace.sh ./path_to/some_file.py
#   (mostly used via replace_all.sh)
echo "Changing to snake_case: $1"
while read p;
    do
        A="$(echo $p | cut -d ' ' -f1)"
        B="$(echo $p | cut -d ' ' -f2)"
        sed -i "s/$A/$B/g" $1
    done < sorted_replacements.txt
