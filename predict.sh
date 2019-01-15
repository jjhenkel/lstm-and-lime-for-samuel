#!/bin/bash

# Grab N lines from err-traces-shuf reversed and trimmed

echo "Top-3 Predictions:"
cat err-traces-shuf.txt.gz \
  | gzip -cd \
  | head -n $1 \
  | awk '{for (i=NF; i>3; i--) printf("%s ", $i); printf("\n") }' \
  | docker run -i --rm lstmlime 2>/dev/null \
  | awk -F'[,|]' '{ printf("%-15s (", $1); printf("%8s%%", $2*100); printf(") | %15-s (", $3); printf("%8s%%", $4*100); printf(") | %-15s (", $5); printf("%8s%%", $6*100); print ")"; }'

echo ""
echo "Actuals:"
cat err-traces-shuf.txt.gz \
  | gzip -cd \
  | head -n $1 \
  | awk '{ print $2 }'
