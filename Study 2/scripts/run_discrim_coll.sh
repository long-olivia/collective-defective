#!/bin/bash

pairs=(
  # "collective collective"
  # "collective neutral"
  "collective self"
)

for pair in "${pairs[@]}"
do
    for i in {1..100}
        do
            echo "Running with: $pair, round $i"
            python rephrased_discrim.py $pair
            sleep 1
        done
done

deactivate