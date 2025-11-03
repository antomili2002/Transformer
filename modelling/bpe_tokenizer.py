from collections import Counter
from typing import List, Dict, Tuple, Set

# Preprocessing utilities
def normalize_corpus(corpus: List[str]) -> List[str]:
    # lower and remove whitespace padding
    return [s.lower().strip() for s in corpus]

def find_unique_words(corpus: List[str]) -> Set[str]:
    uniq = set()
    for s in corpus:
        uniq.update(s.split())
    return uniq

def find_unique_characters_from_words(unique_words: Set[str]) -> Set[str]:
    chars = set()
    for w in unique_words:
        chars.update(w)
    return chars

def word_frequency_from_corpus(corpus: List[str]) -> Dict[str, int]:
    freqs = Counter()
    for s in corpus:
        freqs.update(s.split())
    return dict(freqs)

# BPE core helpers
def words_to_symbols(unique_words: Set[str]) -> Dict[str, List[str]]:
    # per-word symbolization (no merges across words)
    return {w: list(w) for w in unique_words}

def count_pair_frequencies(
    sym_words: Dict[str, List[str]],
    word_freqs: Dict[str, int]
) -> Counter:
    pair_counts = Counter()
    for w, syms in sym_words.items():
        f = word_freqs.get(w, 0)
        if f <= 0 or len(syms) < 2:
            continue
        for i in range(len(syms) - 1):
            pair_counts[(syms[i], syms[i+1])] += f
    return pair_counts

def merge_everywhere(
    sym_words: Dict[str, List[str]],
    pair: Tuple[str, str]
) -> Tuple[Dict[str, List[str]], str]:
    a, b = pair
    merged = a + b
    new_sym_words = {}
    for w, syms in sym_words.items():
        out, j = [], 0
        L = len(syms)
        while j < L:
            if j + 1 < L and syms[j] == a and syms[j+1] == b:
                out.append(merged)
                j += 2
            else:
                out.append(syms[j])
                j += 1
        new_sym_words[w] = out
    return new_sym_words, merged

def build_vocab_from_symbols(sym_words: Dict[str, List[str]]) -> Set[str]:
    vocab = set()
    for syms in sym_words.values():
        vocab.update(syms)
    return vocab

# Training
def train_bpe(
    corpus: List[str],
    target_vocab_size: int,
) -> Tuple[Set[str], List[Tuple[str, str]], Dict[str, List[str]]]:
    """
    Train a BPE tokenizer on a corpus and stop when the vocabulary size is reached.

    Args:
        corpus: list of sentences (text corpus)
        target_vocab_size: stop once the vocabulary reaches or exceeds this size

    Returns:
        vocab: final set of tokens (base chars + merged tokens)
        merges: list of (a,b) in training order
        sym_words: symbolized forms of unique words after training
    """
    corpus = normalize_corpus(corpus)
    word_freqs = word_frequency_from_corpus(corpus)
    unique_words = set(word_freqs.keys())
    sym_words = words_to_symbols(unique_words)

    # base vocab from words (characters only, no spaces)
    vocab = find_unique_characters_from_words(unique_words).copy()
    merges: List[Tuple[str, str]] = []

    while len(vocab) < target_vocab_size:
        pair_counts = count_pair_frequencies(sym_words, word_freqs)
        if not pair_counts:
            break
        # pick most frequent pair
        best_pair, best_count = max(pair_counts.items(), key=lambda it: (it[1], it[0]))
        if best_count <= 0:
            break
        sym_words, merged_token = merge_everywhere(sym_words, best_pair)
        merges.append(best_pair)
        vocab.add(merged_token)

    # optional: rebuild vocab from actual used symbols
    vocab = build_vocab_from_symbols(sym_words)
    return vocab, merges, sym_words


# encoding new text with trained merges
def build_merge_ranks(merges: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    # earlier merge = lower rank (applied first in standard BPE)
    return {pair: i for i, pair in enumerate(merges)}

def bpe_merge_word(symbols: List[str], ranks: Dict[Tuple[str, str], int]) -> List[str]:
    # Greedy BPE: repeatedly merge the lowest-rank adjacent pair present
    if len(symbols) < 2:
        return symbols[:]
    while True:
        pairs = [( (symbols[i], symbols[i+1]), i ) for i in range(len(symbols) - 1)]
        # find the mergeable pair with the best (lowest) rank
        best = None
        best_rank = None
        best_idx = None
        for (a_b, idx) in pairs:
            r = ranks.get(a_b)
            if r is None:
                continue
            if best_rank is None or r < best_rank:
                best = a_b
                best_rank = r
                best_idx = idx
        if best is None:
            break  # no more merges applicable
        a, b = best
        merged = a + b
        # apply one merge occurrence (leftmost best-ranked), then continue
        symbols = symbols[:best_idx] + [merged] + symbols[best_idx+2:]
    return symbols

def encode(
    text: str,
    merges: List[Tuple[str, str]]
) -> List[str]:
    # whitespace pretokenization; same lowercase normalization as training
    text = text.lower().strip()
    ranks = build_merge_ranks(merges)
    out_tokens: List[str] = []
    for word in text.split():
        syms = list(word)
        merged = bpe_merge_word(syms, ranks)
        out_tokens.extend(merged)  # keep simple space separator
    return out_tokens


if __name__ == "__main__":
    # use bpe tokenizer from huggingface as reference implementation
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.normalizers import Lowercase, Sequence
    
    corpus = [
    "Machine learning helps in understanding complex patterns.",
    "Learning machine languages can be complex yet rewarding.",
    "Natural language processing unlocks valuable insights from data.",
    "Processing language naturally is a valuable skill in machine learning.",
    "Understanding natural language is crucial in machine learning."
    ]
    vocab, merges, sym_words = train_bpe(corpus, target_vocab_size=64)
    #print("MERGES:", merges)
    #print("VOCAB SIZE:", len(vocab))
    print("ENCODE:", encode("Machine learning is a subset of artificial intelligence.", merges))

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = Sequence([Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=64, special_tokens=[])
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    print("HF ENCODE:", tokenizer.encode("Machine learning is a subset of artificial intelligence.").tokens)