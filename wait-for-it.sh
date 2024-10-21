#!/usr/bin/env bash
# wait-for-it.sh

set -e

TIMEOUT=30
WAITFORIT_VERSION="v2.0.0"
WAITFORIT_GITHUB="https://raw.githubusercontent.com/vishnubob/wait-for-it/${WAITFORIT_VERSION}/wait-for-it.sh"

# Check if script is downloaded
if [[ ! -f "wait-for-it.sh" ]]; then
    curl -o wait-for-it.sh "$WAITFORIT_GITHUB"
    chmod +x wait-for-it.sh
fi

# Use wait-for-it to wait for the database
exec ./wait-for-it.sh "$@"