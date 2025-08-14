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


for i in {1..51}
  do
            echo "Running with: neutral self, round $i"
            python basic_setup.py neutral self
            sleep 1
  done

for i in {1..26}
  do 
    echo "Running with neutral collective, round $i"
    python basic_setup.py neutral collective
    sleep 1
  done