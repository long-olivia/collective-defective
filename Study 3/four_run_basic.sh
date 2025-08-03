#!/bin/bash

pairs=(
  # "collective collective collective collective"
  # "neutral neutral neutral neutral"
  "self self self self"
)

for pair in "${pairs[@]}"
do
    for i in {1..50}
        do
            echo "Running with: $pair, round $i"
            python four_rephrased_basic.py $pair
            sleep 1
        done
done

deactivate