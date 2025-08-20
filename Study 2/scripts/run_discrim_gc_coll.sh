#!/bin/bash

pairs=(
  # "collective collective"
  # "collective self"
  "collective neutral"
)

for pair in "${pairs[@]}"
do
    for i in {1..100}
        do
            echo "Running with: $pair, round $i"
            python rephrased_discrim_gc.py $pair
            sleep 1
        done
done

deactivate