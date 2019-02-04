#!/bin/bash

clear & clear
for f in $(find . -name "unittest*" | grep -v ".pyc" | sed -e 's_\./__;s_/_._g;s/\.py//');
do
    python -m unittest $f;
done
