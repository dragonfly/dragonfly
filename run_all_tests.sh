#!/bin/bash

if [[ "$1" == "sanity" ]];
then
    # Try importing dragonfly and save stdout and stderr
    res=$(python -c "import dragonfly" 2>&1)

    echo $res
    # Default value to reduce over
    code=1
    # Scan over all arguments
    for o in "$@"
    do
        case "$o" in
        sanity) echo "$res" | grep "No module named dragonfly" ;;
        direct) echo "$res" | grep "Could not import fortran direct" ;;
        optimal) echo "$res" | grep "Could not import Python optimal transport" ;;
        -direct) echo "$res" | grep -v "Could not import fortran direct" && echo "Has direct" ;;
        -optimal) echo "$res" | grep -v "Could not import Python optimal transport" && echo "Has optimal" ;;
        *) echo "The accepted inputs are 'sanity', 'direct', '-direct', 'optimal' and '-optimal'." ;;
        esac
        code=$(( $code & $?))
    done

    # Return nonzero exit code if any value was found
    if [[ $code == 0 ]];
    then
        exit 1;
    fi
    exit 0;
fi

for f in $(find . -name "unittest*" | grep -v ".pyc" | sed -e 's_\./__;s_/_._g;s/\.py//');
do
    python -m unittest $f;
done
