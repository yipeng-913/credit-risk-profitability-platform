#!/bin/sh
set -eu
case "${1:-}" in
    webserver|scheduler) exec airflow "$@" ;;
    *) exec "$@" ;;
esac
