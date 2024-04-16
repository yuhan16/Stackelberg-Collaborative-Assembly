#!/bin/bash

tasks=(1 2 3 4 5 6 7 8)
echo "Start training all tasks using bash parallel..."
date

for i in ${tasks[@]}
do 
(
    echo "Processing task $i..."
    python experiments/sg_train.py $i
    #python experiments/ind_train.py $i
    #python experiments/nash_train.py $i
) &
done

wait
echo "All training completed."
date