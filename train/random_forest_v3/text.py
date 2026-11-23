"""v3's tokenizer. One policy, no knobs, no fitted state.

The four short answers are split the same way:

    half-width (NFKC) -> delete apostrophes (don't -> dont) -> lowercase
    -> every character outside [a-z0-9] becomes a space -> split on spaces
    -> drop tokens shorter than 2 -> drop tokens with no letter

The hyphen is a separator, so `hot-dog` and `hot dog` produce the same tokens. `nba2k`
keeps `nba2k` (it has a letter), `50 cent` keeps `cent` and drops `50`, and `don't`
becomes `dont`.

Why this replaces the v1 tokenizer `\\b[a-zA-Z]{2,}\\b`: on the 606 training rows the two
produce the same number of answers with no tokens at all (14 food, 9 colour, 35 song,
9 feeling) and 2708 against 2711 food tokens, so the swap costs nothing in coverage. It
is kept because it is explicit: the regex silently drops every token that contains a
digit or an accent, which is invisible in a vocabulary listing, while this function can
be read off the page.

There is no standard form (no suffix stripping) in v3. It was measured on the feeling
answer and did not separate from noise, so the pipeline does not carry the code for it.
"""
import re
import unicodedata

APOSTROPHE = re.compile("['\u2019\u02bc\u2032`]")
# Everything that is not a lowercase letter or a digit becomes a space, after NFKC has
# already folded full-width characters onto their half-width forms.
NOT_WORD = re.compile(r'[^a-z0-9]+')
LETTER = re.compile(r'[a-z]')

MIN_LEN = 2


def normalise(text):
    """Return the string that will be split on spaces."""
    folded = unicodedata.normalize('NFKC', str(text))
    folded = APOSTROPHE.sub('', folded)
    folded = folded.lower()
    return NOT_WORD.sub(' ', folded)


def tokens(text):
    """Split one answer into tokens, dropping the two kinds of noise."""
    out = []
    for token in normalise(text).split():
        if len(token) < MIN_LEN:
            continue
        if not LETTER.search(token):
            continue
        out.append(token)
    return out


def tokenize(texts):
    """Tokenise an iterable of answers, keeping the order."""
    return [tokens(text) for text in texts]
