#!/bin/bash

pairs=(
  "collective collective"
  "collective self"
  "collective neutral"
  "neutral neutral"
  "neutral self"
  "neutral collective"
  "self self"
  "self collective"
  "self neutral"
)

for pair in "${pairs[@]}"
do
    for i in {1..100}
        do
            echo "Running with: $pair, round $i"
            python self_discrim.py $pair
            sleep 1
        done
done

deactivate