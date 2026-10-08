"""
Phrase Ranker & Scorer Module for Phase 4.

Computes a deterministic, explainable Phrase Quality Score combining:
1. Recurrence frequency
2. Statistical association (PMI / Tri-PMI)
3. Content-word ratio
4. Grammatical POS structure
5. Function word penalties
6. Overlapping / parent-child phrase resolution
"""

import math
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional, Set
import nltk

from src.phrase_filter import ENGLISH_STOPWORDS, BOUNDARY_STOPWORDS
from src.collocation_scorer import calculate_pmi_bigram, calculate_pmi_trigram

# Prime POS patterns for high-value entity and noun phrases
PREFERRED_POS_PATTERNS: Set[str] = {
    # Bigrams
    "JJ-NN",
    "JJ-NNS",
    "NN-NN",
    "NN-NNS",
    "NNP-NNP",
    "NNP-NN",
    "VBG-NN",
    "VBN-NN",
    # Trigrams
    "JJ-JJ-NN",
    "JJ-NN-NN",
    "JJ-NN-NNS",
    "NN-NN-NN",
    "NN-NN-NNS",
    "NN-IN-NN",
    "JJ-IN-NN",
    "NNP-NNP-NNP",
}

CONTENT_POS_PREFIXES = ("NN", "JJ", "VB")


def compute_content_word_ratio(tagged_tokens: List[Tuple[str, str]]) -> float:
    """
    Compute ratio of content words (nouns, adjectives, verbs) to total tokens.
    
    Args:
        tagged_tokens: List of (word, pos_tag) tuples.
        
    Returns:
        Content word ratio between 0.0 and 1.0.
    """
    if not tagged_tokens:
        return 0.0
    content_count = sum(
        1 for word, tag in tagged_tokens
        if any(tag.startswith(p) for p in CONTENT_POS_PREFIXES) and word.lower() not in ENGLISH_STOPWORDS
    )
    return round(content_count / len(tagged_tokens), 3)


def assign_quality_category(quality_score: float) -> str:
    """
    Deterministically assign explainable quality category based on score thresholds.
    
    Categories:
    - HIGH: >= 6.5
    - MEDIUM: >= 4.0 and < 6.5
    - LOW: < 4.0
    """
    if quality_score >= 6.5:
        return "HIGH"
    elif quality_score >= 4.0:
        return "MEDIUM"
    else:
        return "LOW"


def score_phrase_candidate(
    phrase_tuple: Tuple[str, ...],
    frequency: int,
    association_score: float,
    tagged_tokens: List[Tuple[str, str]],
) -> Tuple[float, str, float]:
    """
    Calculate the deterministic Phrase Quality Score.
    
    Formula components:
    - Recurrence: 1.5 * log2(frequency + 1)
    - Association: 0.35 * max(0.0, association_score)
    - Content Density: 2.5 * content_word_ratio
    - POS Pattern: +1.5 to +2.0 bonus for preferred noun phrases
    - Penalties: -8.0 for all-stopword, -2.5 for trailing/leading boundary stopwords
    
    Returns:
        Tuple of (quality_score, pos_pattern, content_word_ratio).
    """
    tokens_lower = [t.lower() for t in phrase_tuple]
    ngram_size = len(phrase_tuple)
    pos_tags = [tag for _, tag in tagged_tokens]
    pos_pattern = "-".join(pos_tags)
    content_ratio = compute_content_word_ratio(tagged_tokens)

    # 1. Base recurrence score
    rec_score = 1.5 * math.log2(frequency + 1.0)

    # 2. Association score contribution
    assoc_contrib = 0.35 * max(0.0, association_score)

    # 3. Content-word contribution
    content_contrib = 2.5 * content_ratio

    # 4. POS pattern bonus
    pos_bonus = 0.0
    if pos_pattern in PREFERRED_POS_PATTERNS:
        pos_bonus = 2.0 if ngram_size == 3 else 1.5
    elif any(tag.startswith("NN") for tag in pos_tags):
        pos_bonus = 0.8

    # 5. Penalties
    penalties = 0.0
    # Severe penalty if all words are stopwords
    if all(t in ENGLISH_STOPWORDS for t in tokens_lower):
        penalties += 8.0
    # Penalty if boundary starts or ends with function words without valid content
    if tokens_lower[0] in BOUNDARY_STOPWORDS:
        penalties += 2.0
    if tokens_lower[-1] in BOUNDARY_STOPWORDS:
        penalties += 2.0
    # Penalty for repeated identical words ('the the')
    if len(set(tokens_lower)) == 1:
        penalties += 8.0

    # Small length bonus for trigrams (more specific semantic context)
    length_bonus = 0.5 if ngram_size == 3 else 0.0

    total_score = rec_score + assoc_contrib + content_contrib + pos_bonus + length_bonus - penalties
    return (round(total_score, 2), pos_pattern, content_ratio)


def analyze_and_rank_phrases(
    tokens: List[str],
    min_frequency: int = 2,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform complete Phase 4 phrase extraction, association scoring, and ranking.
    
    Args:
        tokens: Full normalized token stream.
        min_frequency: Minimum recurrence frequency required (default: 2).
        top_k: Optional limit on number of top ranked phrases returned.
        
    Returns:
        Dict containing bigram analysis, trigram analysis, ranked phrases, and statistics.
    """
    from src.ngram_extractor import extract_bigrams, extract_trigrams

    total_unigrams = len(tokens)
    unigram_counts = Counter(tokens)

    bigram_counts = extract_bigrams(tokens)
    total_bigrams = sum(bigram_counts.values())

    trigram_counts = extract_trigrams(tokens)
    total_trigrams = sum(trigram_counts.values())

    # Tag unigrams/bigrams/trigrams for POS analysis
    # We can tag the entire token sequence to get contextual POS tags
    try:
        tagged_corpus = nltk.pos_tag(tokens)
    except Exception:
        tagged_corpus = [(tok, "NN") for tok in tokens]

    # Pre-build POS lookup for token sequences
    pos_lookup: Dict[Tuple[str, ...], List[Tuple[str, str]]] = {}
    for i in range(len(tokens) - 1):
        bg = tuple(tokens[i : i + 2])
        if bg not in pos_lookup:
            pos_lookup[bg] = tagged_corpus[i : i + 2]
    for i in range(len(tokens) - 2):
        tg = tuple(tokens[i : i + 3])
        if tg not in pos_lookup:
            pos_lookup[tg] = tagged_corpus[i : i + 3]

    # 1. Analyze All Bigrams
    bigram_analysis: List[Dict[str, Any]] = []
    for bg_tuple, freq in bigram_counts.items():
        prob = round(freq / total_bigrams, 6) if total_bigrams > 0 else 0.0
        pmi = calculate_pmi_bigram(bg_tuple, bigram_counts, unigram_counts, total_bigrams, total_unigrams)
        tagged = pos_lookup.get(bg_tuple, [(bg_tuple[0], "NN"), (bg_tuple[1], "NN")])
        q_score, pos_pat, content_ratio = score_phrase_candidate(bg_tuple, freq, pmi, tagged)
        category = assign_quality_category(q_score)

        bigram_analysis.append({
            "phrase": " ".join(bg_tuple),
            "ngram_tuple": bg_tuple,
            "ngram_size": 2,
            "frequency": freq,
            "probability": prob,
            "pmi": pmi,
            "pos_pattern": pos_pat,
            "content_word_ratio": content_ratio,
            "quality_score": q_score,
            "quality_category": category,
        })

    # 2. Analyze All Trigrams
    trigram_analysis: List[Dict[str, Any]] = []
    for tg_tuple, freq in trigram_counts.items():
        prob = round(freq / total_trigrams, 6) if total_trigrams > 0 else 0.0
        pmi = calculate_pmi_trigram(tg_tuple, trigram_counts, unigram_counts, total_trigrams, total_unigrams)
        tagged = pos_lookup.get(tg_tuple, [(t, "NN") for t in tg_tuple])
        q_score, pos_pat, content_ratio = score_phrase_candidate(tg_tuple, freq, pmi, tagged)
        category = assign_quality_category(q_score)

        trigram_analysis.append({
            "phrase": " ".join(tg_tuple),
            "ngram_tuple": tg_tuple,
            "ngram_size": 3,
            "frequency": freq,
            "probability": prob,
            "pmi": pmi,
            "pos_pattern": pos_pat,
            "content_word_ratio": content_ratio,
            "quality_score": q_score,
            "quality_category": category,
        })

    # Sort analysis tables deterministically
    bigram_analysis.sort(key=lambda x: (-x["quality_score"], -x["frequency"], x["phrase"]))
    trigram_analysis.sort(key=lambda x: (-x["quality_score"], -x["frequency"], x["phrase"]))

    # 3. Detect Overlapping / Parent Phrases among recurring candidates
    # Create lookup of recurring trigram phrases
    recurring_trigrams = {
        item["phrase"]: item
        for item in trigram_analysis
        if item["frequency"] >= min_frequency and item["quality_score"] >= 4.0
    }

    # 4. Filter and Combine Recurring Candidates
    combined_candidates: List[Dict[str, Any]] = []

    for item in bigram_analysis + trigram_analysis:
        if item["frequency"] >= min_frequency and item["quality_score"] > 2.0:
            # Check if this phrase is a sub-phrase of an active recurring trigram
            phrase_text = item["phrase"]
            parent_phrase = None
            if item["ngram_size"] == 2:
                for tg_text in recurring_trigrams:
                    if phrase_text in tg_text:
                        parent_phrase = tg_text
                        break

            item_copy = dict(item)
            item_copy["parent_phrase"] = parent_phrase or "none"
            combined_candidates.append(item_copy)

    # 5. Deterministic Ranking
    # Primary: quality_score DESC, Secondary: frequency DESC, Tertiary: phrase ASC
    combined_candidates.sort(
        key=lambda x: (-x["quality_score"], -x["frequency"], x["phrase"])
    )

    # Assign rank
    for rank, item in enumerate(combined_candidates, start=1):
        item["rank"] = rank

    ranked_phrases = combined_candidates if top_k is None else combined_candidates[:top_k]

    return {
        "total_unigrams": total_unigrams,
        "total_bigrams": total_bigrams,
        "total_trigrams": total_trigrams,
        "unique_bigrams": len(bigram_counts),
        "unique_trigrams": len(trigram_counts),
        "recurring_bigrams": sum(1 for f in bigram_counts.values() if f >= min_frequency),
        "recurring_trigrams": sum(1 for f in trigram_counts.values() if f >= min_frequency),
        "min_frequency": min_frequency,
        "bigram_analysis": bigram_analysis,
        "trigram_analysis": trigram_analysis,
        "ranked_phrases": ranked_phrases,
    }
