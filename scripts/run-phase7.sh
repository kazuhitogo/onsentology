#!/usr/bin/env bash
# Phase 7 の実測。2モデル × 4条件 × 12問 × 3回 = 288サンプル。
# 逐次で走らせる（並列にすると Bedrock のスロットリングで欠測が出る）。
set -uo pipefail

ACCOUNT=008300324376
REGION=ap-northeast-1
HAIKU="arn:aws:bedrock:${REGION}:${ACCOUNT}:application-inference-profile/flnf7r9i5pgb"
SONNET="arn:aws:bedrock:${REGION}:${ACCOUNT}:application-inference-profile/pbrje5prrfgi"

export AWS_PROFILE=dev-vm
export AWS_REGION=$REGION

mkdir -p eval-results

for pair in "haiku-4-5:$HAIKU" "sonnet-4-6:$SONNET"; do
  name=${pair%%:*}
  arn=${pair#*:}
  for run in 1 2 3; do
    out="eval-results/phase7-${name}-run${run}.jsonl"
    if [ -s "$out" ]; then
      echo "skip (already exists): $out"
      continue
    fi
    echo "=== $name run$run -> $out ==="
    ONSEN_BEDROCK_MODEL_ID="$arn" uv run onsen eval --out "$out" \
      > "eval-results/phase7-${name}-run${run}.summary.json" 2>&1
    echo "exit=$? lines=$(wc -l < "$out" 2>/dev/null || echo 0)"
  done
done
echo "ALL DONE"
