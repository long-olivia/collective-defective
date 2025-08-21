#!/bin/bash

pairs=(
  # "neutral neutral"
  # "neutral self"
  "neutral collective"
)

for pair in "${pairs[@]}"
do
    for i in {1..34}
        do
            echo "Running with: $pair, round $i"
            python rephrased_basic.py $pair
            sleep 1
        done
done

deactivate