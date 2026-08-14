#!/usr/bin/env bash
# 条件 F（揃えた生ドキュメント）の実測。12問 × 3回 × 2モデル = 72サンプル。
set -uo pipefail
ACCOUNT=008300324376; REGION=ap-northeast-1
HAIKU="arn:aws:bedrock:${REGION}:${ACCOUNT}:application-inference-profile/flnf7r9i5pgb"
SONNET="arn:aws:bedrock:${REGION}:${ACCOUNT}:application-inference-profile/pbrje5prrfgi"
export AWS_PROFILE=dev-vm AWS_REGION=$REGION
for pair in "haiku-4-5:$HAIKU" "sonnet-4-6:$SONNET"; do
  name=${pair%%:*}; arn=${pair#*:}
  for run in 1 2 3; do
    out="eval-results/phase7f-${name}-run${run}.jsonl"
    [ -s "$out" ] && { echo "skip $out"; continue; }
    echo "=== F $name run$run ==="
    ONSEN_BEDROCK_MODEL_ID="$arn" uv run onsen eval --conditions F --out "$out" \
      > "eval-results/phase7f-${name}-run${run}.summary.json" 2>&1
    echo "exit=$? lines=$(wc -l < "$out" 2>/dev/null || echo 0)"
  done
done
echo "ALL DONE"
