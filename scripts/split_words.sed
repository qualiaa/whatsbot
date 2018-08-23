#!/usr/bin/env sed -Ef

# Remove timestamp, prepend "Begin" to name (e.g. BeginJamie)
s/^[0-9\/]{10}, [0-9:]{5} - ([^:]+):/Begin\1/

# delete a Whatsapp context that isn't a message
/^[0-9\/]{10}, [0-9:]{5} -/d
:next_line

# Separate all punctuation and digits with spaces.
# Remove duplicate spaces
s/[[:punct:][:digit:]]/ & /g
s/[[:space:]]+/ /g

# If next line is new WhatsApp context, print current message and restart on new one
N; /\n[0-9\/]{10}, [0-9:]{5} -/{
    s/\n/ EOM\n/
    P; D
}

# Otherwise join it to current message with LINEBREAK and repeat on next line
s/\n/ LINEBREAK /
b next_line
