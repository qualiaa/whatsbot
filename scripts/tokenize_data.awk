#!/usr/bin/env awk -f

FNR == NR { vocab[$2] = $1; vocab_size=max($1,vocab_size);  next }
FNR == 1 { UNKNOWN_WORD = vocab_size+1 }

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
