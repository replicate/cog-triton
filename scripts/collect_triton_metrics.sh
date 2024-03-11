#!/bin/bash

# if triton_metrics.log exists, remove it
if [ -f triton_metrics.log ]; then
  rm triton_metrics.log
fi
# The URL where Triton metrics are available
METRICS_URL="http://localhost:8002/metrics"

# The file where you want to store the metrics
OUTPUT_FILE="triton_metrics.log"

# Interval in seconds between each metrics collection
INTERVAL=1

# Infinite loop to collect metrics every few seconds
while true; do
  curl -s $METRICS_URL >> $OUTPUT_FILE
  echo "Collected metrics at $(date)" >> $OUTPUT_FILE
  sleep $INTERVAL
done