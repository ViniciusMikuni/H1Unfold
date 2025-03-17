#!/bin/bash
# Pattern for the files
if [ $# -eq 1 ]; then
    num=$1
else
    echo "specify the job array ID"
    exit 1
   
fi

files=(slurm-"$num"_*.out)

if [ ${#files[@]} -eq 0 ]; then
    echo "No files processed."
    exit 1
fi

# Detect flag from the first file's first line
first_line=$(head -n1 "${files[0]}")
if echo "$first_line" | grep -q -- "--fine_tune"; then
    detected_flag="Fine-Tuned"
elif echo "$first_line" | grep -q -- "--load_pretrain"; then
    detected_flag="Pre-Trained"
else
    detected_flag="Baseline"
fi

# Variables for overall average computation
total_avg=0
file_count=0

echo ""
echo "Detected $detected_flag"

for file in "${files[@]}"; do
    # Extract time values from the last 50 lines
    times=$(tail -n 50 "$file" | grep "OmniFold Took" | grep -oP 'Took \K[0-9.]+')
    
    file_sum=0
    count=0
    for t in $times; do
        file_sum=$(echo "$file_sum + $t" | bc -l)
        count=$((count + 1))
    done

    if [ $count -gt 0 ]; then
        file_avg=$(echo "$file_sum / $count" | bc -l)
        # echo "File: $file, Average time: $file_avg seconds"
        total_avg=$(echo "$total_avg + $file_avg" | bc -l)
        file_count=$((file_count + 1))
    else
        echo "File: $file, No valid time entries found."
    fi
done

if [ $file_count -gt 0 ]; then
    overall_avg=$(echo "$total_avg / $file_count" | bc -l)
    echo ""
    echo "($num) $detected_flag Overall Average time: $overall_avg seconds"
    echo ""
else
    echo "No valid time entries found in any file."
fi
