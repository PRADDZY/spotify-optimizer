#!/usr/bin/env bash
set -euo pipefail

REDIS_URL=${REDIS_URL:-redis://localhost:6379/0}
REDIS_CLI=(redis-cli -u "$REDIS_URL")

printf "Redis URL: %s\n" "$REDIS_URL"
printf "Ping: %s\n" "$("${REDIS_CLI[@]}" ping)"

printf "\nMemory:\n"
"${REDIS_CLI[@]}" info memory | grep -E 'used_memory_human|used_memory_peak_human' || true

printf "\nStats:\n"
"${REDIS_CLI[@]}" info stats | grep -E 'expired_keys|evicted_keys' || true

printf "\nKey count (approx):\n"
"${REDIS_CLI[@]}" --scan | wc -l

printf "\nPurging allocator (memory purge)\n"
"${REDIS_CLI[@]}" memory purge || true
