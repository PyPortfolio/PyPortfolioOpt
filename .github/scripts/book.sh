#!/bin/sh
# Assemble the combined documentation site into _book
# - Copies API docs (pdoc), coverage, test report, and marimushka exports
# - Generates a links.json consumed by minibook
#
# This script mirrors the logic previously embedded in the Makefile `book` target
# for maintainability and testability. It is POSIX-sh compatible.

set -e

BLUE="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

printf "%b[INFO] Building combined documentation...%b\n" "$BLUE" "$RESET"
printf "%b[INFO] Assembling book...%b\n" "$BLUE" "$RESET"

printf "%b[INFO] Delete the _book folder...%b\n" "$BLUE" "$RESET"
rm -rf _book
printf "%b[INFO] Create empty _book folder...%b\n" "$BLUE" "$RESET"
mkdir -p _book

# Start building links.json content without jq
LINKS_ENTRIES=""

#printf "%b[INFO] Copy API docs...%b\n" "$BLUE" "$RESET"
#if [ -f _pdoc/index.html ]; then
#  mkdir -p _book/pdoc
#  cp -r _pdoc/* _book/pdoc
#  LINKS_ENTRIES='"API": "./pdoc/index.html"'
#fi

printf "%b[INFO] Copy coverage report...%b\n" "$BLUE" "$RESET"
if [ -f _tests/html-coverage/index.html ]; then
  mkdir -p _book/tests/html-coverage
  cp -r _tests/html-coverage/* _book/tests/html-coverage
  #if [ -n "$LINKS_ENTRIES" ]; then
  #  LINKS_ENTRIES="$LINKS_ENTRIES, \"Coverage\": \"./tests/html-coverage/index.html\""
  #else
  LINKS_ENTRIES='"Coverage": "./tests/html-coverage/index.html"'
  #fi
else
  printf "%b[WARN] No coverage report found or directory is empty%b\n" "$YELLOW" "$RESET"
fi

printf "%b[INFO] Copy test report...%b\n" "$BLUE" "$RESET"
if [ -f _tests/html-report/report.html ]; then
  mkdir -p _book/tests/html-report
  cp -r _tests/html-report/* _book/tests/html-report
  if [ -n "$LINKS_ENTRIES" ]; then
    LINKS_ENTRIES="$LINKS_ENTRIES, \"Test Report\": \"./tests/html-report/report.html\""
  else
    LINKS_ENTRIES='"Test Report": "./tests/html-report/report.html"'
  fi
else
  printf "%b[WARN] No test report found or directory is empty%b\n" "$YELLOW" "$RESET"
fi

#printf "%b[INFO] Copy notebooks...%b\n" "$BLUE" "$RESET"
#if [ -f _marimushka/index.html ]; then
#  mkdir -p _book/marimushka
#  cp -r _marimushka/* _book/marimushka
#  if [ -n "$LINKS_ENTRIES" ]; then
#    LINKS_ENTRIES="$LINKS_ENTRIES, \"Notebooks\": \"./marimushka/index.html\""
#  else
#    LINKS_ENTRIES='"Notebooks": "./marimushka/index.html"'
#  fi
#  printf "%b[INFO] Copied notebooks into _book/marimushka%b\n" "$BLUE" "$RESET"
#else
#  printf "%b[WARN] No notebooks found or directory is empty%b\n" "$YELLOW" "$RESET"
#fi

# Write final links.json
if [ -n "$LINKS_ENTRIES" ]; then
  printf '{%s}\n' "$LINKS_ENTRIES" > _book/links.json
else
  printf '{}\n' > _book/links.json
fi

printf "%b[INFO] Generated links.json:%b\n" "$BLUE" "$RESET"
cat _book/links.json
