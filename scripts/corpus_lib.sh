#!/usr/bin/env bash
# Corpus list for shell runners, from shared/corpus.py. Source it, don't copy it.
#
#   source scripts/corpus_lib.sh
#   mapfile -t SLUGS < <(corpus_slugs)          # respects CUAD=1
#   mapfile -t DOCIDS < <(corpus_docids)        # wiki_v* document ids
#
# Every answer script used to carry its own list of the original ten documents,
# with the other twenty bolted on through a second run_new20.sh. That is why the
# results tables carry three different row counts for one benchmark, and why our
# index reported on ten movies against the baselines' thirty. One list, derived
# from data/ at call time, so a script cannot drift from the corpus it claims.

corpus_slugs() {
  "${PY:-envs/base/bin/python}" -c "
import os, sys
sys.path.insert(0, 'shared')
from corpus import DOCS
print('\n'.join(DOCS['cuad' if os.environ.get('CUAD') else 'nqa'].slugs))"
}

corpus_docids() {
  "${PY:-envs/base/bin/python}" -c "
import os, sys
sys.path.insert(0, 'shared')
from corpus import wiki_docids
print('\n'.join(wiki_docids('cuad' if os.environ.get('CUAD') else 'nqa').values()))"
}
