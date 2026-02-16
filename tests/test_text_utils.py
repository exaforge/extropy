"""Unit tests for text utilities."""

from extropy.simulation.text_utils import compute_trigram_jaccard


class TestComputeTrigramJaccard:
    """Tests for trigram Jaccard similarity."""

    def test_identical_texts_returns_1(self):
        """Identical texts should have similarity of 1.0."""
        text = (
            "I am very worried about my job security and what this means for my family"
        )
        assert compute_trigram_jaccard(text, text) == 1.0

    def test_completely_different_texts_returns_0(self):
        """Completely different texts should have similarity near 0."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "A completely unrelated sentence with no overlap whatsoever here"
        similarity = compute_trigram_jaccard(text1, text2)
        assert similarity < 0.1

    def test_similar_texts_high_similarity(self):
        """Similar/paraphrased texts should have high similarity."""
        text1 = "I am worried about my job and what this means for my family"
        text2 = "I am worried about my job and what this means for our family"
        similarity = compute_trigram_jaccard(text1, text2)
        # One word change still yields ~69% similarity
        assert similarity > 0.6

    def test_short_text_returns_0(self):
        """Texts with fewer than 3 words should return 0."""
        assert compute_trigram_jaccard("hello world", "hello world") == 0.0
        assert compute_trigram_jaccard("one", "two") == 0.0

    def test_empty_text_returns_0(self):
        """Empty texts should return 0."""
        assert compute_trigram_jaccard("", "") == 0.0
        assert compute_trigram_jaccard("hello there friend", "") == 0.0

    def test_case_insensitive(self):
        """Similarity should be case-insensitive."""
        text1 = "I Am Worried About My Job"
        text2 = "i am worried about my job"
        assert compute_trigram_jaccard(text1, text2) == 1.0

    def test_partial_overlap(self):
        """Texts with partial overlap should have intermediate similarity."""
        text1 = "I need to save money and cut expenses immediately"
        text2 = "I need to save money but also invest for the future"
        similarity = compute_trigram_jaccard(text1, text2)
        # Some overlap but not complete
        assert 0.2 < similarity < 0.8

    def test_repetitive_reasoning_detection(self):
        """Should detect when agent reasoning is repetitive."""
        reasoning1 = (
            "I'm terrified about losing my job. Need to cut spending and save money. "
            "Maybe look for backup work. Lisa and I need to talk about the budget."
        )
        reasoning2 = (
            "Still terrified about losing my job. Need to cut spending and save money. "
            "Looking at gig apps for backup work. Lisa and I talked about the budget."
        )
        similarity = compute_trigram_jaccard(reasoning1, reasoning2)
        # These share themes but are paraphrased — ~43% similarity
        # Higher than completely different texts, showing partial overlap
        assert similarity > 0.3

    def test_different_reasoning_low_similarity(self):
        """Different reasoning should have low similarity."""
        reasoning1 = (
            "I'm terrified about losing my job. Need to cut spending and save money. "
            "Maybe look for backup work."
        )
        reasoning2 = (
            "Actually feeling more optimistic now. The retraining program looks promising. "
            "I signed up for the AI course and it's going well."
        )
        similarity = compute_trigram_jaccard(reasoning1, reasoning2)
        assert similarity < 0.3
