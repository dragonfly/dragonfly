#!/bin/bash

if [[ "$1" == "sanity" ]];
then
    # Try importing dragonfly and save stdout and stderr
    res=$(python -c "import dragonfly" 2>&1 | paste -sd " " - )

    echo "======================"
    echo $res
    echo "======================"
    echo
    # Default value to reduce over
    code=1
    # Scan over all arguments
    for o in "$@"
    do
        case "$o" in
        sanity) echo "$res" | grep "No module named dragonfly" ;;
        direct) echo "$res" | grep "Could not import Fortran direct" ;;
        optimal) echo "$res" | grep "Could not import Python optimal transport" ;;
        -direct) echo "$res" | grep -v "Could not import Fortran direct" && echo "Has direct" ;;
        -optimal) echo "$res" | grep -v "Could not import Python optimal transport" && echo "Has optimal" ;;
        *) echo "The accepted inputs are 'sanity', 'direct', '-direct', 'optimal' and '-optimal'." ;;
        esac
        code=$(( $code & $?))
    done

    # Return nonzero exit code if any value was found
    exit $(($code == 0 ));
fi

# This is to ensure that there will not be an error due to matplotlib not having a valid backend.
export MPLBACKEND="Agg"
nosetests dragonfly --all-modules -m '^test_' -e 'test_data'
