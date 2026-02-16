"""Text utilities for cognitive architecture features."""


def compute_trigram_jaccard(text1: str, text2: str) -> float:
    """Compute Jaccard similarity of word-level trigrams.

    Word trigrams (3-word sequences) are more semantically meaningful
    than character trigrams for detecting paraphrased repetition.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Jaccard similarity in [0, 1]. >0.7 indicates repetitive content.
    """

    def get_word_trigrams(text: str) -> set[tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < 3:
            return set()
        return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}

    t1 = get_word_trigrams(text1)
    t2 = get_word_trigrams(text2)

    if not t1 or not t2:
        return 0.0

    intersection = len(t1 & t2)
    union = len(t1 | t2)

    return intersection / union if union > 0 else 0.0
