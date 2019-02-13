#!/bin/bash

if [[ "$1" == "sanity" ]];
then
    # The ! is to invert the status code for the whole line
    ! python -c "import dragonfly" | grep -q "No module named dragonfly"
    exit $?
fi

for f in $(find . -name "unittest*" | grep -v ".pyc" | sed -e 's_\./__;s_/_._g;s/\.py//');
do
    python -m unittest $f;
done
