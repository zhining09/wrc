#!/usr/bin/env bash
set +x
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
key_file="$repo_dir/.kimi-code-api-key"

if [[ ! -t 0 ]]; then
    printf 'Run this script in an interactive terminal.\n' >&2
    exit 1
fi
if ! IFS= read -r -s -p 'Kimi Code API Key: ' api_key; then
    printf '\nNo key saved.\n' >&2
    exit 1
fi
printf '\n'
if [[ ! "$api_key" =~ [^[:space:]] ]]; then
    printf 'The key is empty. No key saved.\n' >&2
    exit 1
fi

umask 077
temp_file="$(mktemp "$key_file.XXXXXX")"
trap 'rm -f -- "$temp_file"' EXIT
printf '%s\n' "$api_key" > "$temp_file"
mv -fT -- "$temp_file" "$key_file"
printf 'Kimi Code API key saved for this WRC checkout.\n'
