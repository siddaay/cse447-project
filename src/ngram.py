# src/kn_charlm.py
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Tuple, Set, List
import heapq

Char = str
Hist = Tuple[Char, ...]  # length 0..5
Ngram = Tuple[Char, ...] # length 1..6


@dataclass
class KNCharLM:
    n: int = 6
    discounts: Dict[int, float] = field(default_factory=lambda: {k: 0.75 for k in range(2, 7)})

    # counts[k][ngram] = count, where k is order (1..n)
    counts: Dict[int, Dict[Ngram, int]] = field(
        default_factory=lambda: {k: defaultdict(int) for k in range(1, 7)}
    )

    # ctx_totals[m][hist] = sum_w count(hist+w), where m=len(hist) (0..n-1)
    ctx_totals: Dict[int, Dict[Hist, int]] = field(
        default_factory=lambda: {m: defaultdict(int) for m in range(0, 6)}
    )

    # follow_sets[m][hist] = set of next chars seen after hist (m=len(hist), 0..n-1)
    follow_sets: Dict[int, Dict[Hist, Set[Char]]] = field(
        default_factory=lambda: {m: defaultdict(set) for m in range(0, 6)}
    )

    # precede_sets[w] = set of chars that precede w (for continuation unigram)
    precede_sets: Dict[Char, Set[Char]] = field(default_factory=lambda: defaultdict(set))

    vocab: Set[Char] = field(default_factory=set)

    # --- cached derived quantities (invalidated on update) ---
    _precede_denom: int = field(default=0, init=False, repr=False)
    # sorted unigram scores: list of (score, char) descending
    _unigram_cache: List[Tuple[float, Char]] = field(default_factory=list, init=False, repr=False)
    _cache_dirty: bool = field(default=True, init=False, repr=False)
    # topk cache: hist -> list of chars
    _topk_cache: Dict[Hist, List[Char]] = field(default_factory=dict, init=False, repr=False)

    # ------------------------------------------------------------------ #
    #  Training                                                           #
    # ------------------------------------------------------------------ #

    def update_from_text(self, text: str) -> None:
        """Update counts with a full string (treat each character as a token)."""
        history: List[Char] = []
        n1 = self.n - 1
        for ch in text:
            h = tuple(history[-n1:]) if history else ()
            self.update_step(h, ch)
            history.append(ch)

    def update_step(self, history: Hist, ch: Char) -> None:
        """Online update given current history (len<=n-1) and observed next char ch."""
        self.vocab.add(ch)
        self._cache_dirty = True          # invalidate caches
        self._topk_cache.clear()

        max_hist = min(len(history), self.n - 1)
        for m in range(0, max_hist + 1):
            h = history[-m:] if m > 0 else ()
            order = m + 1
            ng = h + (ch,)
            self.counts[order][ng] += 1
            self.ctx_totals[m][h] += 1
            self.follow_sets[m][h].add(ch)

            if order == 2:
                prev = h[-1]
                self.precede_sets[ch].add(prev)

    # ------------------------------------------------------------------ #
    #  Cached unigram                                                     #
    # ------------------------------------------------------------------ #

    def _rebuild_unigram_cache(self) -> None:
        """Recompute unigram continuation scores for all vocab chars."""
        self._precede_denom = sum(len(s) for s in self.precede_sets.values())
        V = max(len(self.vocab), 1)
        denom = self._precede_denom if self._precede_denom > 0 else 1

        scores = []
        for ch in self.vocab:
            num = len(self.precede_sets.get(ch, set()))
            score = num / denom if self._precede_denom > 0 else 1.0 / V
            scores.append((score, ch))
        scores.sort(reverse=True)
        self._unigram_cache = scores
        self._cache_dirty = False

    def _p_cont_unigram(self, ch: Char) -> float:
        if self._cache_dirty:
            self._rebuild_unigram_cache()
        if self._precede_denom == 0:
            return 1.0 / max(len(self.vocab), 1)
        num = len(self.precede_sets.get(ch, set()))
        return num / self._precede_denom

    # ------------------------------------------------------------------ #
    #  Iterative Kneser-Ney (replaces recursive p_kn)                    #
    # ------------------------------------------------------------------ #

    def p_kn(self, history: Hist, ch: Char) -> float:
        """Interpolated Kneser-Ney probability (iterative, no recursion)."""
        return self._p_kn_iterative(history, ch)

    def _p_kn_iterative(self, history: Hist, ch: Char) -> float:
        """
        Compute P_KN(ch | history) iteratively from longest to shortest context.

        At each level m (context length), the formula is:
            P(ch|h) = max(c(h,ch)-D, 0)/c(h)  +  lambda(h) * P(ch|h[1:])
        We accumulate the result bottom-up, starting from the unigram.
        """
        m = min(len(history), self.n - 1)
        h = history[-m:] if m > 0 else ()

        # Walk back to the deepest level that has a non-zero context total.
        while m > 0 and self.ctx_totals[m].get(h, 0) == 0:
            m -= 1
            h = h[1:] if m > 0 else ()

        # Base: unigram
        prob = self._p_cont_unigram(ch)

        # Iterate from order-1 (unigram already done) up to current m
        for level in range(1, m + 1):
            h_level = history[-(level):] if level > 0 else ()
            # trim to n-1
            h_level = h_level[-(self.n - 1):]
            c_h = self.ctx_totals[level].get(h_level, 0)
            if c_h == 0:
                continue
            order = level + 1
            c_hw = self.counts[order].get(h_level + (ch,), 0)
            D = self.discounts.get(order, 0.75)
            first = max(c_hw - D, 0.0) / c_h
            n1plus = len(self.follow_sets[level].get(h_level, set()))
            lam = (D * n1plus) / c_h
            prob = first + lam * prob

        return prob

    # ------------------------------------------------------------------ #
    #  Top-k prediction with candidate pruning                           #
    # ------------------------------------------------------------------ #

    def topk_next(self, context: str, k: int = 3) -> List[Char]:
        """
        Return top-k next chars given context string.

        Optimisations vs. naive:
        1. Candidate pruning: only score chars that appear in follow_sets at
           any backoff level.  All remaining vocab chars share the same
           unigram score ordering, so we pull the top-k from the precomputed
           sorted unigram list to fill gaps.
        2. Cached unigram scores (rebuilt only when model is updated).
        3. Iterative KN (no Python recursion overhead).
        4. Result cache keyed on history tuple (cleared on every update).
        """
        if not self.vocab:
            return [" "] * k

        hist = tuple(context[-(self.n - 1):])

        # --- cache hit ---
        if hist in self._topk_cache:
            return self._topk_cache[hist]

        if self._cache_dirty:
            self._rebuild_unigram_cache()

        # --- collect candidates: chars with non-zero first term at any level ---
        candidates: Set[Char] = set()
        h = hist
        while h:
            m = len(h)
            fs = self.follow_sets[m].get(h)
            if fs:
                candidates.update(fs)
            h = h[1:]
        # unigram follow set (m=0) — all chars ever seen as "next"
        candidates.update(self.follow_sets[0].get((), set()))

        # --- score candidates with full KN ---
        scored = [(self._p_kn_iterative(hist, ch), ch) for ch in candidates]

        # --- use heapq.nlargest instead of full sort (O(n + k log n)) ---
        top = heapq.nlargest(k, scored)
        result = [ch for _, ch in top]

        # --- if fewer than k candidates, fill from cached unigram ranking ---
        if len(result) < k:
            result_set = set(result)
            for _, ch in self._unigram_cache:
                if ch not in result_set:
                    result.append(ch)
                    if len(result) == k:
                        break

        self._topk_cache[hist] = result
        return result

    # ------------------------------------------------------------------ #
    #  Streaming API (unchanged interface)                               #
    # ------------------------------------------------------------------ #

    def step_streaming(self, context: str, true_next: Char, k: int = 3) -> List[Char]:
        """
        Streaming API: given current context + ground-truth next char,
        return top-k predictions, then update model with the ground-truth char.
        """
        preds = self.topk_next(context, k=k)
        hist = tuple(context[-(self.n - 1):])
        self.update_step(hist, true_next)
        return preds