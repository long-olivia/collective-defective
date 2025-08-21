#!/bin/bash

pairs=(
  # "collective collective collective collective"
  # "neutral neutral neutral neutral"
  "self self self self"
)

models=(
  # "openai/gpt-4o"
  "anthropic/claude-sonnet-4"
  # "meta-llama/llama-4-maverick"
  # "qwen/qwen3-235b-a22b-2507"
)

for model in "${models[@]}"
do
  echo "Running with model $model"
  for pair in "${pairs[@]}"
  do
      for i in {1..7}
          do
              echo "Running with: $pair, round $i"
              python four_rephrased_discrim.py $pair $model
              sleep 1
          done
  done
done
deactivate