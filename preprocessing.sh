#!/usr/bin/env sh

set -eu

DATA_DIR="data"

# Input file: WhatsApp text dump
INPUT_FILE="${DATA_DIR}/raw_input.txt"

# Output files:
#  - Vocab file mapping word to token
#  - Token file: conversion of input file to tokens
#  - Temp file for storing split words

VOCAB_FILE="${DATA_DIR}/vocab.txt"
TOKEN_FILE="${DATA_DIR}/network_input.txt"

TEMP_FILE=$(mktemp)

# Scripts:
SCRIPTS_DIR="scripts"

SPLIT_WORDS="${SCRIPTS_DIR}/split_words.sed"
GENERATE_VOCAB="${SCRIPTS_DIR}/generate_vocab.py"
TOKENIZE_DATA="${SCRIPTS_DIR}/tokenize_data.awk"


function split_words() {
    # This function achieves a few things:
    # - Separates punctuation from other characters to be treated as individual
    #   words
    # - Removes timestamps
    # - 
    # - Insert End Of Message word (EOM)
    # - Replace "Speaker:" with "BeginSpeaker" word
    # Preserve inserted linebreaks as LINEBREAK
    sed --regexp-extended --file="$SPLIT_WORDS"
}

function generate_vocab() {
    # Extract unique words from split words and give them a unique token
    # maintain capitalisation of words only appearing capitalised
    # but discard them if there is a lowercase equivalent
    tr ' ' '\n' | # put each word on a separate line
    python "$GENERATE_VOCAB"
}

function tokenize_data() {
    # Using the vocabulary file (first argument), tokenize the split words file
    # (second argument)
    awk -f "$TOKENIZE_DATA" "$1" "$2"
}

split_words < "$INPUT_FILE" > $TEMP_FILE

generate_vocab < "$TEMP_FILE" > "$VOCAB_FILE" 

tokenize_data "$VOCAB_FILE" "$TEMP_FILE" > "$TOKEN_FILE"
