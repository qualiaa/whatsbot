#!/usr/bin/env awk -f

# Store vocabulary from first input file
FNR == NR {
    vocab[$2] = $1
    vocab_size=max($1,vocab_size)
    next
}

# Once finished processing vocabulary, create unknown word token
FNR == 1 { UNKNOWN_WORD = vocab_size+1 }

# Convert words in subsequent files to tokens
{
    ORS=" "
    split($0,words)
    for (word in words) {
        print wordToToken(words[word])
    }
}

function wordToToken(word) {
    if (word in vocab) return vocab[word]
    lower = tolower(word) 
    if (lower in vocab) return vocab[lower]
    return UNKNOWN_WORD
}

function max(a,b) {
    if (a > b) return a
    return b
}
