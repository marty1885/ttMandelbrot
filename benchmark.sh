#!/bin/zsh

executables=("cpu" "cpu_single_core" "tt_single_core" "tt_single_core_nullary" "tt_multi_core_nullary")
output_csv="benchmark.csv"

echo "executable,size,time" > $output_csv
outputpath=`realpath $output_csv`
cd build
for executable in "${executables[@]}"; do
    for size in $(seq 1024 1024 16384); do
        echo -n "Running $executable $size"
        if [[ "$executable" == "cpu_single_core" ]]; then
            t=$(env TT_METAL_LOGGER_LEVELt=FATAL ./cpu -t 1 --width $size --height $size -o /dev/null 2> /dev/null | grep -v '|' | awk -F' ' '{print $3}' | tr -d \\n)
        else
            t=$(env TT_METAL_LOGGER_LEVELt=FATAL ./$executable --width $size --height $size -o /dev/null 2> /dev/null | grep -v '|' | awk -F' ' '{print $3}' | tr -d \\n)
        fi
        echo " -> ${t}s"
        echo "$executable,$size,$t" >> $outputpath
    done
done
