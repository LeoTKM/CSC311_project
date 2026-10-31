"""v2 text utilities: the tokenizer and the standard form. Standalone.

Nothing here is imported by v1 and nothing in v1 imports it.

Tokenisation, as decided in PLAN.md:

    half-width (NFKC) -> delete apostrophes (don't -> dont) -> lowercase
    -> every character outside [a-z0-9] becomes a space -> split on spaces
    -> drop tokens shorter than min_len -> optionally drop tokens with no letter

The hyphen is a separator, so `hot-dog` and `hot dog` produce the same tokens. The
`keep` and `empty` policies remain available because the scripts in `analysis/` compare
them against `space`.

The standard form folds the derivations of one word onto a single key
(`excited`, `excitement`, `exciting` -> `excite`). PLAN.md applies it to the feeling
answer only. It is a pure string function with no fitted state, so scoring a new answer
later needs nothing but this file.
"""
import re
import unicodedata

APOSTROPHE = re.compile("['\u2019\u02bc\u2032`]")
WITH_HYPHEN = re.compile(r'[^a-z0-9\-]+')
WITHOUT_HYPHEN = re.compile(r'[^a-z0-9]+')
POLICIES = ('keep', 'space', 'empty')


def normalise(text, hyphen='space'):
    """Return the cleaned string that will be split on spaces."""
    if hyphen not in POLICIES:
        raise ValueError(f'unknown hyphen policy {hyphen!r}; choose from {POLICIES}')
    s = unicodedata.normalize('NFKC', str(text))     # full-width -> half-width
    s = APOSTROPHE.sub('', s)                       # don't -> dont
    s = s.lower()
    if hyphen == 'empty':
        s = s.replace('-', '')                      # warm-hearted -> warmhearted
        s = WITHOUT_HYPHEN.sub(' ', s)
    elif hyphen == 'space':
        s = WITH_HYPHEN.sub(' ', s).replace('-', ' ')
    else:
        s = WITH_HYPHEN.sub(' ', s)                 # keep the hyphen inside tokens
    return s


def tokens(text, hyphen='space', min_len=1, require_letter=False):
    """Split into tokens, then apply the optional noise filters."""
    out = []
    for token in normalise(text, hyphen).split():
        if hyphen == 'keep' and set(token) <= {'-'}:
            continue                                 # a lone separator is not a word
        if len(token) < min_len:
            continue
        if require_letter and not re.search(r'[a-z]', token):
            continue
        out.append(token)
    return out


def tokenize_all(hyphen='space', **kwargs):
    """Return a callable suitable for a vectorizer's tokenizer argument."""
    def tokenize(text):
        return tokens(text, hyphen=hyphen, **kwargs)
    return tokenize


# --------------------------------------------------------------------------- standard form
#
# Applied to the feeling answer before its vocabulary is counted, so that one emotion
# is one column instead of three thin ones. The rule is a suffix chain followed by a
# silent-e restoration; the short table covers the words the chain cannot line up.

VOWEL = frozenset('aeiou')
MIN_STEM = 3

# longest first, so that `relaxation` loses `ation` rather than `ion`
STANDARD_SUFFIXES = (
    'ations', 'ation', 'ments', 'ment', 'ness', 'dom', 'ions', 'ion',
    'ances', 'ance', 'ences', 'ence', 'ities', 'ity', 'ings', 'ing',
    'edly', 'ed', 'ly', 'es',
)

STANDARD_IRREGULAR = {
    'energetic': 'energetic', 'energized': 'energetic', 'energize': 'energetic',
    'energy': 'energetic', 'energetized': 'energetic', 'energies': 'energetic',
    'anxious': 'anxious', 'anxiety': 'anxious',
    'happy': 'happy', 'happiness': 'happy',
    'sad': 'sad', 'sadness': 'sad',
    'tense': 'tense', 'tension': 'tense',
    'relaxation': 'relaxe', 'relaxing': 'relaxe', 'relax': 'relaxe',
}


def standard_form(token):
    """Fold one derivation onto a standard key. Stateless and idempotent."""
    if token in STANDARD_IRREGULAR:
        return STANDARD_IRREGULAR[token]
    stem = token
    for _ in range(3):
        for suffix in STANDARD_SUFFIXES:
            if stem.endswith(suffix) and len(stem) - len(suffix) >= MIN_STEM:
                stem = stem[:-len(suffix)]
                break
        else:
            break
    if (len(stem) >= 4 and stem[-1] not in VOWEL and stem[-2] in VOWEL
            and stem[-3] not in VOWEL and stem[-1] != 'y'):
        stem = stem + 'e'
    return stem
