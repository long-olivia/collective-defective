#!/bin/bash

pairs=(
  # "self self"
  # "self collective"
  "self neutral"
)

for pair in "${pairs[@]}"
do
    for i in {1..100}
        do
            echo "Running with: $pair, round $i"
            python rephrased_basic_gc.py $pair
            sleep 1
        done
done

deactivate