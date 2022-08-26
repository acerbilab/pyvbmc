#!/bin/bash
# Calls replace.sh on each specified file.
# Usage: ./replace_all.sh replace_files_pyvbmc.txt
#        ./replace_all.sh replace_files_tests.txt
while read p;
    do
        ./replace.sh $p
    done < $1
