"""Information-weighted consistency: a similarity whose measure comes from the data, not the tree.

Research utility: not part of sted's stable API (see the README); may change or move.

Motivation (see docs/analysis/aggregation_axioms_and_pareto_ceiling.md for the measurements).
Both published metrics are the same functional under two different measures on the leaf set::

    1 - SLOT  =  E_unif[d(leaf)]     uniform measure
    1 - STED  =  E_walk[d(leaf)]     root-to-leaf uniform walk, leaf weight prod 1/|siblings|

Probes A1/A2 show each satisfies exactly one of two invariances and violates the other:

    bracket invariance     re-nest the same leaves without changing which one disagrees
                           SLOT swing 0.000000     STED swing 0.038715
    refinement invariance  spell one wrong concept with 1, 2, then 4 subfields
                           STED swing 0.007990     SLOT swing 0.096032

That is not an accident of implementation. A measure on leaves cannot be both *exchangeable*
(which forces uniform) and *refinement-invariant* (which forces sibling mass to partition the
parent's) unless the tree is a flat list, so any measure determined by tree shape must fail one.

The escape is to stop deriving the measure from shape. Bracketing and refinement are both
*reparametrisations* of the same content, so both invariances collapse into one axiom:

    (R)  two representations of the same content must receive the same score.

The quantities invariant under reparametrisation are information-theoretic: entropy is
unchanged by bijective recoding, and -- crucially for refinement -- it is *not additive over
dependent parts*, so spelling one concept with four perfectly-correlated subfields does not
quadruple its weight. So the measure is estimated from the distribution of outputs, and R is
satisfied by construction rather than by tuning.

The functional. Given a corpus Y of outputs for comparable inputs, positions are grouped into
redundancy classes (positions that induce the same partition of the corpus carry the same
information). Writing H_g for the information of class g and sigma_p in [0,1] for the leaf
agreement of the pair at position p::

    S(x, y) = sum_g H_g * mean_{p in g, present} sigma_p   /   sum_g H_g

WHAT IS PROVEN (each is a theorem about the form, not a tuned behaviour):

  identity        S(x, x) = 1                     every sigma_p = 1
  symmetry        S(x, y) = S(y, x)               sigma is symmetric
  bounds          S in [0, 1]                     H_g >= 0 and sigma_p in [0, 1]
  monotonicity    worsening any leaf cannot raise S; linear in sigma with H_g >= 0
  bracket (A1)    re-nesting does not change any position's value distribution, so every
                  H_g is unchanged and so is S
  refinement (A2) k perfectly-correlated subfields land in ONE redundancy class and share its
                  mass, so the class contributes H_g * (j/k) for j agreeing subfields --
                  identical to the unrefined field's H_g * (1/1) when all agree or all disagree

  This is the first construction here that satisfies A1 and A2 *together*, which no
  shape-determined measure can.

WHERE A2 STOPS HOLDING -- measured, not conjectured (probe A6, 2026-08-22). Read the A2 line
above carefully: it only closes "when all agree or all disagree", i.e. when the class is
*homogeneous*. When a class mixes agreeing and disagreeing members, ``mean sigma`` is a plain
count average, so refinement changes it and the invariance breaks. The A6 sweep found a residual
swing of 0.0836 that does NOT decay with corpus size (0.082783 at N=12, 0.083649 at N=768), so it
is not estimator bias. Two causes, both real:

  (i)  spurious classes. Redundancy is decided by exact equality of induced partitions, which is
       a brittle in-sample test. In the A6 corpus the filler fields and ``name.first`` both cycle
       with period 4, so ``f0`` determines ``first`` in-sample and they merge. A periodic or
       small corpus never outgrows this -- at N=768 the period-12 structure is exactly as
       redundant as at N=12, which is why the swing plateaus.
  (ii) the within-class mean. Once ``f0,f1,f2,name.first`` share a class, its mean sigma is
       3/3, 3/4, 3/5 across the 1-, 2-, and 4-subfield spellings, reproducing the three observed
       scores (0.3595 / 0.4157 / 0.3325) to four decimals. Count dilution, the very thing
       information weighting was meant to remove, comes back inside the class.

``CoverageConsistency`` below removes both by deleting the grouping step: joint entropy is
already redundancy-aware and has no mean. It has its own, different failure mode (saturation),
which is measured rather than hidden.

WHAT IS NOT PROVEN, and was overclaimed in an earlier note:

  A3 / conjunctive semantics does NOT follow. Information weighting replaces count-dilution
  with *information*-dilution: the mass of a tool name is H_name / (H_name + sum H_arg), which
  still shrinks as independent informative arguments are added. It shrinks less when the extra
  arguments are boilerplate (low entropy) and not at all when they are redundant with each
  other, but "a wrong function name must cost more than one wrong argument" is an empirical
  question here, not a theorem. Note also that if the arguments really are determined by the
  name then they are in the name's redundancy class and carry no separate information -- so a
  candidate with the right arguments and the wrong name is off-distribution, and the corpus
  cannot say how to rank it.

Honest limitation: S is *corpus-relative*, not intrinsic to the pair. For a consistency metric
that is arguably correct -- consistency is a distributional notion -- but for ground-truth
comparison it makes the score depend on the reference corpus, the same standing as IDF in
retrieval. It must be disclosed, not hidden.

Array handling is explicit rather than defaulted. SLOT keys array elements by index (group 1 at
every node) and STED sorts them (group S_n at every node); each is right on about half the
benchmark, which is why one wins the aligned stratum and the other the misaligned one. This
class takes ``array_mode`` and reports all three settings rather than picking one.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Sentinel for "this position does not occur in this sample". It has to be a distinct value
# rather than None, because ``None`` is a legal JSON leaf and conflating the two would make an
# explicit null indistinguishable from an absent key.
ABSENT = object()


def _scalar_key(v: Any) -> Any:
    """A hashable, type-aware identity for a leaf value.

    Types are kept apart on purpose: ``1``, ``1.0``, ``"1"`` and ``True`` are different
    observations for the purpose of estimating how much a position varies, even though a leaf
    *similarity* function may well score them as near-identical.
    """
    return (type(v).__name__, v)


def positions(obj: Any, array_mode: str = "positional",
              prefix: str = "") -> Dict[str, Any]:
    """Flatten JSON to {position: leaf value}.

    Empty containers are positions in their own right. SLOT's ``flatten`` drops them, which
    makes ``{"tool_calls": []}`` indistinguishable from ``{}`` and scores that pair 1.0; an
    empty array is a real, observable claim about the output and is kept here.

    array_mode:
        ``positional``    element index is part of the position (group 1; SLOT's convention)
        ``exchangeable``  index is replaced by ``[*]`` (group S_n; STED's convention), so all
                          elements of an array share one position and the position's value
                          distribution is the pooled element distribution
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix + "{}" if prefix else "{}"] = ()
            return out
        for k, v in obj.items():
            child = f"{prefix}.{k}" if prefix else str(k)
            out.update(positions(v, array_mode, child))
    elif isinstance(obj, list):
        if not obj:
            out[prefix + "[]" if prefix else "[]"] = ()
            return out
        for i, v in enumerate(obj):
            tag = "[*]" if array_mode == "exchangeable" else f"[{i}]"
            sub = positions(v, array_mode, f"{prefix}{tag}")
            for p, val in sub.items():
                if array_mode == "exchangeable" and p in out:
                    # Pooling several elements onto one position: keep them all so the
                    # position's empirical distribution is the element distribution, and so
                    # agreement can be scored as a multiset overlap rather than pairwise.
                    prev = out[p]
                    out[p] = (prev if isinstance(prev, _Bag) else _Bag([prev])) + _Bag([val])
                else:
                    out[p] = val
    else:
        out[prefix] = obj
    return out


def involved_positions(px: Dict[str, Any], py: Dict[str, Any]) -> List[str]:
    """The union of the two position sets, minus empty-container sentinels the other side refutes.

    A position absent from BOTH sides is not part of this pair's content and is excluded from
    numerator and denominator alike. A position present in only one side stays in, with sigma = 0:
    omitting a required field must not be cheaper than getting it wrong.

    The one exception is the empty-container sentinel. ``positions()`` emits ``path{}`` / ``path[]``
    deliberately -- SLOT's ``flatten`` drops them and therefore cannot tell ``{"a": []}`` from
    ``{}``. But when one side is empty at ``path`` while the other has children *under* ``path``,
    that sentinel is not independent information: it is entailed by the absence of those children,
    and those children are already in the union carrying sigma = 0. Counting the sentinel as well
    charges one omission twice, and the refund of that double charge is what made hallucination
    free on a damage ladder: emptying ``arguments`` and then adding a spurious key *swaps*
    ``arguments{}`` for ``arguments.X``, both weighted at the smoothing floor, so H(P) is
    bit-identical and no metric built on it can order the two rungs.

    Measured, on the 589 adjacent argument-insertion pairs of
    ``results/rebuttal/cover_tie_probe.json``: the 152 tied pairs are separated from the 437
    correctly-ordered ones by exactly this signature -- mean positions removed 0.724 vs 0.009 and
    ``w_added == w_removed`` in 68.4% vs 0.0%. Pointwise surprisal of an *unseen* position was the
    prior hypothesis and the same measurement refutes it: unseen additions are commoner in the
    correctly-ordered group (29.3%) than in the tied one (19.7%).

    The condition matters. Dropping sentinels unconditionally would re-open the defect they exist
    for -- gold ``{"a": {}}`` against a candidate omitting ``a`` would score 1.0 -- so the sentinel
    is removed only when the other side has children there, which is precisely when the absence is
    counted elsewhere. No parameter is introduced and the result is still a set-valued function of
    the pair, so any modular or submodular H over it keeps monotonicity.
    """
    keep = set(px) | set(py)
    for src, other in ((px, py), (py, px)):
        for p in src:
            if not (p.endswith("{}") or p.endswith("[]")):
                continue
            base = p[:-2]
            # Entailed iff the other side has a STRICT DESCENDANT of base. At the root every
            # position is a descendant except the root sentinels themselves, which is what keeps
            # ``{}`` against ``{}`` a genuine agreement rather than a vacuous empty set.
            if any(_descends(q, base) for q in other):
                keep.discard(p)
    return sorted(keep)


def _descends(q: str, base: str) -> bool:
    if base == "":
        return q not in ("{}", "[]")
    return q.startswith(base + ".") or q.startswith(base + "[")


class _Bag(tuple):
    """A multiset of leaf values pooled onto one exchangeable position."""

    def __add__(self, other):
        return _Bag(tuple(self) + tuple(other))

    def key(self):
        return ("bag", tuple(sorted(_scalar_key(v) for v in self)))


def _obs_key(v: Any) -> Any:
    if v is ABSENT:
        return ("absent",)
    if isinstance(v, _Bag):
        return v.key()
    if isinstance(v, tuple) and v == ():
        return ("empty_container",)
    return _scalar_key(v)


def _entropy_nats(counts: Iterable[int], smoothing: float) -> float:
    """Smoothed plug-in entropy with the Miller-Madow bias correction.

    Smoothing matters for a reason beyond numerical taste: a position that happens to be
    constant across the sampled corpus would otherwise get mass exactly 0, and any pair
    disagreeing there would be scored as identical. One extra pseudo-cell represents "a value
    not yet observed", so no position is ever entirely free to disagree.
    """
    c = [float(x) for x in counts if x > 0]
    if not c:
        return 0.0
    k = len(c) + 1                      # observed support plus one unseen cell
    c = c + [0.0]
    n = sum(c) + smoothing * k
    if n <= 0:
        return 0.0
    h = 0.0
    for x in c:
        p = (x + smoothing) / n
        if p > 0:
            h -= p * math.log(p)
    # Miller-Madow: the plug-in estimator is biased low by (K-1)/(2N).
    h += (k - 1) / (2.0 * max(n, 1.0))
    return max(h, 0.0)


def default_leaf_sim(a: Any, b: Any) -> float:
    """Exact-match leaf kernel: 1.0 iff the observations are identical, else 0.0.

    Deliberately embedding-free so that the invariance probes measure aggregation only. Pass a
    graded kernel via ``leaf_sim`` to recover semantic tolerance; every theorem above is linear
    in sigma and so holds for any kernel with values in [0, 1] and sim(v, v) = 1.
    """
    if a is ABSENT or b is ABSENT:
        return 0.0
    return 1.0 if _obs_key(a) == _obs_key(b) else 0.0


class InformationConsistency:
    """S(x, y) = sum_g H_g * mean sigma / sum_g H_g, with H estimated from a corpus.

    Usage::

        ic = InformationConsistency(array_mode='positional').fit(corpus)
        ic.score(x, y)
    """

    def __init__(self,
                 array_mode: str = "positional",
                 smoothing: float = 0.5,
                 leaf_sim: Optional[Callable[[Any, Any], float]] = None,
                 group_redundant: bool = True):
        if array_mode not in ("positional", "exchangeable"):
            raise ValueError("array_mode must be 'positional' or 'exchangeable'")
        self.array_mode = array_mode
        self.smoothing = smoothing
        self.leaf_sim = leaf_sim or default_leaf_sim
        self.group_redundant = group_redundant
        self.pos_entropy: Dict[str, float] = {}
        self.group_of: Dict[str, int] = {}
        self.group_entropy: Dict[int, float] = {}
        self.group_members: Dict[int, List[str]] = {}
        self._unseen_entropy = 0.0
        self.n_corpus = 0

    # ------------------------------------------------------------------ estimation
    def fit(self, corpus: List[Any]) -> "InformationConsistency":
        """Estimate per-position information and redundancy classes from a corpus."""
        obs: List[Dict[str, Any]] = [positions(o, self.array_mode) for o in corpus]
        self.n_corpus = len(obs)
        all_pos = sorted({p for o in obs for p in o})

        # Signature of a position = the PARTITION of the corpus it induces, canonicalised by
        # replacing each distinct value with the index of its first occurrence. Two positions are
        # redundant when they cut the corpus the same way, which is not the same as holding the
        # same values: ``name.first`` and ``name.last`` never share a value, yet if no two people
        # share a first name then knowing one determines the other and they carry ONE piece of
        # information, not two. Grouping on raw values misses exactly that case, which is the
        # whole mechanism behind refinement invariance (A2).
        sigs: Dict[str, Tuple] = {}
        for p in all_pos:
            raw = [_obs_key(o.get(p, ABSENT)) for o in obs]
            canon: Dict[Any, int] = {}
            sigs[p] = tuple(canon.setdefault(v, len(canon)) for v in raw)
            self.pos_entropy[p] = _entropy_nats(
                Counter(sigs[p]).values(), self.smoothing)

        buckets: Dict[Tuple, List[str]] = defaultdict(list)
        for p in all_pos:
            # Grouping by the signature itself when enabled; otherwise every position is its
            # own class, which recovers a plain information-weighted average and is kept as an
            # ablation so the effect of the redundancy correction is measurable.
            buckets[sigs[p] if self.group_redundant else (p,)].append(p)

        for gid, (sig, members) in enumerate(sorted(buckets.items(), key=lambda kv: kv[1])):
            self.group_members[gid] = members
            # All members share one signature, so they share one entropy -- taking it once is
            # the whole point. Without this the four subfields of a refined name would
            # contribute 4x the mass of the single unrefined field.
            self.group_entropy[gid] = self.pos_entropy[members[0]]
            for p in members:
                self.group_of[p] = gid

        # A position present in a scored pair but never seen in the corpus still has to cost
        # something, or novel keys would be free. Price it at the median observed information,
        # which is a statement about the corpus rather than a tuned constant.
        vals = sorted(self.group_entropy.values())
        self._unseen_entropy = vals[len(vals) // 2] if vals else 1.0
        return self

    # ---------------------------------------------------------------------- scoring
    def score(self, x: Any, y: Any) -> float:
        return self.explain(x, y)["score"]

    def explain(self, x: Any, y: Any) -> Dict[str, Any]:
        """Score plus the per-class breakdown, so a number can always be traced to a position."""
        px = positions(x, self.array_mode)
        py = positions(y, self.array_mode)

        involved = involved_positions(px, py)
        by_group: Dict[Any, List[str]] = defaultdict(list)
        for p in involved:
            by_group[self.group_of.get(p, ("unseen", p))].append(p)

        num = den = 0.0
        detail = []
        for gid, members in sorted(by_group.items(), key=lambda kv: str(kv[0])):
            h = (self.group_entropy[gid] if gid in self.group_entropy
                 else self._unseen_entropy)
            if h <= 0.0:
                continue
            sims = [self.leaf_sim(px.get(p, ABSENT), py.get(p, ABSENT)) for p in members]
            mean_sim = sum(sims) / len(sims)
            num += h * mean_sim
            den += h
            detail.append({"group": str(gid), "members": members,
                           "H": round(h, 6), "mean_sigma": round(mean_sim, 6)})

        # Two outputs with no positions at all (both ``{}``) agree vacuously.
        score = 1.0 if den == 0.0 else num / den
        return {"score": score, "num": num, "den": den,
                "n_groups_involved": len(detail), "detail": detail}


class CoverageConsistency:
    """S(x, y) = H(A) / H(P): the share of the content's information the pair agrees on.

    ``P`` is the set of positions involved in the pair and ``A`` the subset where they agree, and
    ``H`` is the *joint* entropy of the corpus restricted to a position set. This is the same
    "fraction of information preserved" idea as ``InformationConsistency`` with both of its
    approximations removed:

      no grouping step   joint entropy is redundancy-aware by construction. A position perfectly
                         determined by one already in the set adds exactly 0, so ``name.first``
                         and ``name.last`` in a bijective corpus contribute once between them
                         without anyone having to detect that they are redundant. This kills
                         failure (i) -- a spuriously-correlated filler no longer *steals* the
                         name's mass, it just adds nothing of its own.
      no within-class mean   there is no averaging anywhere, so there is no count dilution to
                         reintroduce. This kills failure (ii), and A2 becomes exact rather than
                         conditional on the class being homogeneous: refinement replaces one
                         position by k that jointly determine it, and joint entropy is invariant
                         under bijective recoding of the value alphabet.

    Why this is the right shape and not just a different formula: ``A -> H(A)`` is monotone and
    submodular, so "how much of the content did we get" is a coverage problem, and the axioms come
    from properties of H rather than from choices. A1 holds because positions are labels and H
    does not see nesting. Identity holds because A = P. Bounds hold because H is monotone and
    non-negative. Monotonicity holds because worsening a leaf removes it from A.

    Graded leaf kernels are handled by the **Lovasz extension** of H, which is the canonical
    continuous extension of a submodular set function and is computed by sorting: with positions
    ordered by decreasing sigma,

        H_hat = sum_i (sigma_i - sigma_{i+1}) * H({p_1, ..., p_i}),    sigma_{|P|+1} := 0

    At binary sigma this collapses to exactly H(A), so the theorems above are not weakened by
    admitting a semantic kernel.

    THE FAILURE MODE, stated up front because it is severe and measured (probe A7). Plug-in joint
    entropy saturates: with N corpus samples H(A) <= log N, so once a position set separates all N
    samples every further position appears free and any A that separates the corpus scores 1.0. A
    candidate agreeing only on a per-request unique ``id`` and getting all three content fields
    wrong scores exactly 1.000000, at N = 12 through 1536 -- more data does not help, because the
    id separates the corpus at every N by construction. ``explain`` returns ``saturation`` =
    H(P) / log N for this reason; near 1 the score is not trustworthy and the corpus is too small
    for the number of positions being scored.

    The same probe refuted the assumption that ``InformationConsistency``'s marginal form escapes
    this: it also ranks the id-only candidate above the content-only one from N=96 on (0.706 vs
    0.294 at N=1536). So this is not an artifact of the joint estimator. Marginal entropy scores
    *unpredictability*, an identifier is maximally unpredictable and semantically inert, and no
    purely distributional weight can separate the two.

    ``fit_conditional`` is the repair, and it follows from that diagnosis rather than patching
    around it: condition H on the input. Measured on 24 prompts x 8 resamples, marginal weighting
    ranks the wrong candidate above the right one (id_only 0.6026 > content_only 0.5950) and
    conditioning reverses it decisively (0.1183 vs 1.0000) while saturation drops 1.0245 -> 0.4342.
    Every invariance above survives the change, because the A1/A2 arguments use only that H is a
    monotone submodular set function on positions, and an average of such functions is one.

    The residual 0.1183 is the smoothing floor, not real information, and it is the honest price of
    refusing to let any position be entirely free. Note also what conditioning implies: a field
    that never varies across resamples carries no conditional information, so getting it wrong is
    free. That is correct for *consistency*, which is a statement about variation under resampling,
    and wrong for ground-truth accuracy -- the two uses need different corpora, and mixing them
    would be an error.
    """

    def __init__(self,
                 array_mode: str = "positional",
                 smoothing: float = 0.5,
                 leaf_sim: Optional[Callable[[Any, Any], float]] = None,
                 entropy_model: str = "plugin",
                 collapse_classes: bool = True,
                 normaliser: str = "union",
                 drop_entailed_empty: bool = True):
        if array_mode not in ("positional", "exchangeable"):
            raise ValueError("array_mode must be 'positional' or 'exchangeable'")
        if entropy_model not in ("plugin", "tree"):
            raise ValueError("entropy_model must be 'plugin' or 'tree'")
        if normaliser not in ("union", "harmonic"):
            raise ValueError("normaliser must be 'union' or 'harmonic'")
        # ``union``    S = H(A) / H(P_x u P_y). One symmetric ratio.
        # ``harmonic`` S = harmonic mean of H(A)/H(P_x) and H(A)/H(P_y): information precision and
        #              information recall, combined as SLOT combines simP and simR.
        #
        # The union form conflates two different quantities in one denominator -- the information
        # *at stake* and the information *asserted* -- and the ladder shows the cost of that. Under
        # ``union``, hallucinating a position raises H(P) (good) and emptying a container lowers it
        # (bad), so the same denominator has to serve two opposite jobs and a repair to one is a
        # regression in the other: dropping the entailed empty-container sentinel cut
        # `add_spurious_arg` ties 24.03% -> 15.11% while raising `drop_arg` ties 28.91% -> 35.35%,
        # a net loss (see known_order_validation.md 11.4).
        #
        # Splitting the denominators makes monotonicity a theorem instead of an observation, for
        # any MODULAR H (entropy_model='tree', where H(A) = sum of class weights >= floor > 0):
        #
        #   insertion  A fixed, H(P_y) fixed, H(P_x) strictly grows  => precision strictly falls,
        #              recall unchanged, harmonic mean strictly falls whenever recall > 0.
        #   deletion   of a position the pair agreed on: H(A) and H(P_x) both fall by the same
        #              w(p) > 0 while H(P_y) is fixed  => recall strictly falls; and since
        #              H(A) <= H(P_x), (H(A) - w) / (H(P_x) - w) <= H(A) / H(P_x), so precision
        #              cannot rise. Harmonic mean strictly falls whenever recall > 0.
        #
        # Both directions hold without reference to the corpus, the weights, or the tree, and the
        # only ties left are at recall = 0, where nothing of the gold has been recovered and no
        # bounded metric can separate the rungs.
        #
        # This was also offered as the term SLOT is missing in the other direction -- its simR being
        # a mean whose denominator moves with the candidate, priced at a 12.10% `drop_arg` inversion
        # rate. That is RETRACTED: the rate was an artifact of a ladder generator that damaged one
        # site twice, simR's denominator is the gold's leaf count and does not move, and SLOT
        # inverts 0.00% of clean deletion pairs. The theorem above stands on its own; it repairs
        # nothing in SLOT. And on the corrected criterion the harmonic form is score-identical to
        # the union form on every column, so it buys resolution nowhere yet measured -- kept because
        # the proof is unconditional while the union form's monotonicity is not.
        self.normaliser = normaliser
        # Kept as a switch because it is only defensible in combination: on its own the
        # canonicalisation is killed by its own pre-registered condition (defect 1.894% -> 2.084%),
        # and the harmonic form is what makes it pay. The 2x2 is measured, not assumed.
        self.drop_entailed_empty = drop_entailed_empty
        self.array_mode = array_mode
        self.smoothing = smoothing
        self.leaf_sim = leaf_sim or default_leaf_sim
        self.entropy_model = entropy_model
        # Only consulted by entropy_model='tree'. Merging positions that induce the same partition
        # of the corpus is exact when they really are redundant, and WRONG when they merely
        # co-occur rarely: a position present in 3 of 853 samples induces almost the same partition
        # as any other position present in the same 3, so the two merge on co-ABSENCE rather than on
        # shared information. Once merged, dropping one of them leaves the class covered by the
        # other and costs exactly nothing -- which is failure (i) of the module docstring, arriving
        # in a new place. Setting this False lets the Chow-Liu conditional entropy do the
        # discounting instead, continuously (H(c|parent) floors at the smoothing pseudo-cell rather
        # than at 0), so genuine redundancy is still nearly free but never exactly free.
        self.collapse_classes = collapse_classes
        self._obs: List[Dict[str, Any]] = []
        self._groups: Optional[List[List[Dict[str, Any]]]] = None
        self._cache: Dict[frozenset, float] = {}
        self._tree: Optional[Dict[str, Any]] = None
        self.n_corpus = 0

    def fit(self, corpus: List[Any]) -> "CoverageConsistency":
        """No estimation happens here; joint entropies are computed lazily per position set.

        Which is the point -- there is nothing to fit, hence no fitted constant. The corpus is
        retained because H is a functional of it.
        """
        self._obs = [positions(o, self.array_mode) for o in corpus]
        self._groups = None
        self.n_corpus = len(self._obs)
        self._cache = {}
        self._tree = None
        return self

    def fit_conditional(self, groups: List[List[Any]]) -> "CoverageConsistency":
        """Fit on repeated outputs per input: H becomes H(position | input).

        This is the fix for the failure probe A7 records, and it comes from inside the framework
        rather than from a heuristic. Marginal entropy measures *unpredictability*, so a field
        carrying a per-request unique identifier is maximally informative and semantically inert --
        agreeing on it alone scored a perfect 1.0. Conditioned on the input that field is
        deterministic, so H(id | input) = 0 and it receives zero mass, while a field the model
        genuinely wavers on across resamples keeps its weight.

        It is also the more faithful reading of what this metric is for. Consistency is variation
        under *resampling a fixed prompt*, not variation across different prompts, and the
        generation results already come in that shape (many samples per prompt per temperature).

        ``groups[i]`` is the list of outputs sampled for input ``i``. Entropy is the average
        within-group entropy, which is the plug-in estimate of H(. | input) under a uniform prior
        over the inputs present.
        """
        self._groups = [[positions(o, self.array_mode) for o in g] for g in groups if g]
        self._obs = [o for g in self._groups for o in g]
        self.n_corpus = len(self._obs)
        self._cache = {}
        self._tree = None
        return self

    def joint_entropy(self, pos_set: Iterable[str]) -> float:
        """Plug-in joint entropy (nats) of the corpus restricted to ``pos_set``.

        After ``fit_conditional`` this is the conditional entropy H(pos_set | input), estimated as
        the mean of the within-group entropies.
        """
        key = frozenset(pos_set)
        if key in self._cache:
            return self._cache[key]
        if not key or not self._obs:
            self._cache[key] = 0.0
            return 0.0
        if self.entropy_model == "tree":
            h = self._tree_entropy(key)
            self._cache[key] = h
            return h
        ordered = sorted(key)

        def h_of(obs: List[Dict[str, Any]]) -> float:
            counts = Counter(
                tuple(_obs_key(o.get(p, ABSENT)) for p in ordered) for o in obs)
            return _entropy_nats(counts.values(), self.smoothing)

        if self._groups is None:
            h = h_of(self._obs)
        else:
            h = sum(h_of(g) for g in self._groups) / len(self._groups)
        self._cache[key] = h
        return h

    # ---------------------------------------------------------- tree-factorised entropy model

    def _tree_entropy(self, key: frozenset) -> float:
        """Conditional-information measure: each redundancy class weighted by its INCREMENTAL
        information given its single most-informative neighbour.

        Why a second estimator is needed at all. The plug-in joint entropy above is bounded by
        ``log N`` however many positions it ranges over, because a corpus of N samples admits at
        most N distinct value-tuples. On the Toucan golds a single position -- the tool name --
        already carries **77.7%** of the whole corpus's entropy
        (``results/rebuttal/entropy_resolution_probe.json``), so once a prefix contains it the
        corpus partition is nearly exhausted and every remaining position has *exactly* zero
        conditional entropy. That is why ``cover`` has the cleanest inversion record of any metric
        measured (26 of 8,923 ordered pairs, a fifth of SLOT's) and the worst tie record (1,225):
        it is never wrong about an order it can see, and it cannot see a dropped or hallucinated
        argument at all, because removing or adding one leaves H(A) and H(P) bit-identical. The
        mirror image of STED's ``min(cost, 1.0)`` clip, arrived at from the opposite direction.

        The escape is to stop estimating an |A|-way joint. Fit a Chow-Liu maximum-mutual-information
        spanning tree over the redundancy classes and weight each class by ``H(c | parent_T(c))``,
        the root by ``H(root)``. Three properties, and the first is the one that matters:

          uncapped        the measure is a SUM of per-class conditional entropies, so it grows with
                          the number of classes instead of stopping at ``log N``. No class is ever
                          free, so no damaging edit is ever free.
          modular         hence trivially monotone and submodular, so every A1/A2 theorem above
                          carries over verbatim -- those arguments use only that H is a monotone
                          submodular set function on positions -- and the Lovasz extension code is
                          shared unchanged.
          redundancy-aware  perfectly-correlated positions still collapse into one class exactly,
                          and *partially* correlated ones are discounted by their measured MI. That
                          discount is real on this corpus, not hypothetical: latitude/longitude
                          score 0.9903 normalised MI, tool names across successive calls 0.76-0.85,
                          and 31.5% of the top-40 class pairs exceed 0.1.

        What is given up, stated plainly. A modular measure captures pairwise redundancy only, so
        three positions that are jointly but not pairwise redundant are over-weighted, where the
        plug-in joint handles them exactly. That is the price of resolution, and it is the right
        trade only because the ties above are a total loss of signal while this is a bounded
        over-weighting. The obvious alternative -- charging ``H(c | parent)`` only when the parent
        is present in A -- was rejected because it is NOT monotone: adding a class determined by its
        own parent but informative about its children lowers the total.
        """
        t = self._tree
        if t is None:
            t = self._tree = self._build_tree()
        class_of, w = t["class_of"], t["weight"]
        # An unseen position is constant (always ABSENT) over the corpus, so the plug-in joint gives
        # it exactly 0 and a hallucinated key costs nothing at all. Here it gets the smoothing
        # floor -- the information of a position whose only observed value is "not present" -- which
        # is the same pseudo-cell ``_entropy_nats`` already uses to stop constant positions being
        # free, applied to the one case the joint estimator cannot express.
        classes = set()
        unseen = 0
        for pos in key:
            c = class_of.get(pos)
            if c is None:
                unseen += 1
            else:
                classes.add(c)
        return float(t["floor"] * unseen + sum(w[c] for c in classes))

    def _build_tree(self) -> Dict[str, Any]:
        """Estimate the Chow-Liu tree and the per-class conditional entropies. Fitted once."""
        import numpy as np

        obs = self._obs
        n = len(obs)
        all_pos = sorted({p for o in obs for p in o})

        # Integer-code each position's column once. Everything downstream is a bincount or a sort
        # over these, so the O(|classes|^2) pass below stays in numpy rather than in tuple hashing.
        coded = {}
        for pos in all_pos:
            local, codes = {}, []
            for o in obs:
                k = _obs_key(o.get(pos, ABSENT))
                if k not in local:
                    local[k] = len(local)
                codes.append(local[k])
            coded[pos] = (np.asarray(codes, dtype=np.int64), len(local))

        # Redundancy classes: positions inducing the same partition of the corpus. Collapsing them
        # is what makes exact refinement free, exactly as in the plug-in path (identical columns add
        # nothing to a joint), and it is done explicitly here because the measure is a sum.
        class_of: Dict[str, int] = {}
        by_sig: Dict[bytes, int] = {}
        cols: List[Tuple[Any, int]] = []
        for pos in all_pos:
            arr, card = coded[pos]
            sig = arr.tobytes() if self.collapse_classes else pos.encode()
            if sig not in by_sig:
                by_sig[sig] = len(cols)
                cols.append((arr, card))
            class_of[pos] = by_sig[sig]
        m = len(cols)

        sm = self.smoothing

        def h_np(cnt) -> float:
            """``_entropy_nats`` in numpy, for the O(|classes|^2) pass only.

            Bit-for-bit the same estimator -- observed support plus one unseen cell, additive
            smoothing, Miller-Madow correction -- just not looping in Python 5*10^5 times over
            arrays of length N. Verified equal to ``_entropy_nats`` to 1e-12 in
            ``scripts/experiments/sted_v3_fixes/cover_tree_smoke.py``.
            """
            c = cnt[cnt > 0].astype(np.float64)
            if c.size == 0:
                return 0.0
            k = c.size + 1
            tot = c.sum() + sm * k
            if tot <= 0:
                return 0.0
            pr = (np.append(c, 0.0) + sm) / tot
            h = float(-(pr * np.log(pr)).sum()) + (k - 1) / (2.0 * max(tot, 1.0))
            return max(h, 0.0)

        h_marg = np.array([h_np(np.bincount(a)) for a, _ in cols])
        floor = _entropy_nats([n], sm)

        def pair_entropy(i: int, j: int) -> float:
            ai, _ = cols[i]
            aj, kj = cols[j]
            _, cnt = np.unique(ai * kj + aj, return_counts=True)
            return h_np(cnt)

        # Pairwise MI, then Prim for a MAXIMUM-weight spanning tree. Chow-Liu's result is that this
        # tree is the maximum-likelihood tree-structured approximation to the joint, so which
        # dependencies get kept is determined rather than chosen.
        mi = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                v = max(h_marg[i] + h_marg[j] - pair_entropy(i, j), 0.0)
                mi[i, j] = mi[j, i] = v

        root = int(np.argmax(h_marg)) if m else 0
        parent = [-1] * m
        in_tree = np.zeros(m, dtype=bool)
        in_tree[root] = True
        best = mi[root].copy()
        best_from = np.full(m, root)
        for _ in range(m - 1):
            cand = np.where(in_tree, -1.0, best)
            c = int(np.argmax(cand))
            if in_tree[c]:
                break
            in_tree[c] = True
            parent[c] = int(best_from[c])
            upd = (~in_tree) & (mi[c] > best)
            best[upd] = mi[c][upd]
            best_from[upd] = c

        # weight[c] = H(c | parent_T(c)), floored so that no class is ever entirely free -- the same
        # principle as the smoothing pseudo-cell. The root is unconditional by construction.
        weight = h_marg.copy()
        for c in range(m):
            pa = parent[c]
            if pa < 0:
                continue
            weight[c] = max(pair_entropy(c, pa) - h_marg[pa], floor)

        sizes = np.bincount(list(class_of.values()), minlength=m)
        present = np.array([int((cols[c][0] != cols[c][0][0]).sum()) if cols[c][1] > 1 else 0
                            for c in range(m)])
        return {"class_of": class_of, "parent": parent, "weight": weight,
                "h_marg": h_marg, "floor": floor, "root": root, "n_classes": m,
                "sum_weight": float(weight.sum()), "sum_h_marg": float(h_marg.sum()),
                "n_positions": len(all_pos),
                # Diagnostics for the co-absence merge above: how many positions share a class, and
                # how few corpus members those classes are actually observed in.
                "max_class_size": int(sizes.max()) if m else 0,
                "n_multi_position_classes": int((sizes > 1).sum()),
                "n_positions_in_multi_classes": int(sizes[sizes > 1].sum()),
                "median_varying_samples_per_class": float(np.median(present)) if m else 0.0}

    def score(self, x: Any, y: Any) -> float:
        return self.explain(x, y)["score"]

    def explain(self, x: Any, y: Any) -> Dict[str, Any]:
        px = positions(x, self.array_mode)
        py = positions(y, self.array_mode)
        involved = (involved_positions(px, py) if self.drop_entailed_empty
                    else sorted(set(px) | set(py)))
        if not involved:
            return {"score": 1.0, "H_A": 0.0, "H_P": 0.0, "saturation": 0.0,
                    "n_positions": 0, "n_unseen_positions": 0, "sigma": {}}

        sigma = {p: self.leaf_sim(px.get(p, ABSENT), py.get(p, ABSENT)) for p in involved}
        h_p = self.joint_entropy(involved)

        # Lovasz extension. Ties in sigma contribute zero-width terms, so the sort order among
        # equal sigmas cannot affect the result -- the value is well defined without a tie-break.
        order = sorted(involved, key=lambda p: (-sigma[p], p))
        h_a = 0.0
        prefix: List[str] = []
        for i, p in enumerate(order):
            prefix.append(p)
            nxt = sigma[order[i + 1]] if i + 1 < len(order) else 0.0
            width = sigma[p] - nxt
            if width > 0.0:
                h_a += width * self.joint_entropy(prefix)

        # A corpus that never varies over these positions carries no information, so there is
        # nothing to have got right or wrong; call that vacuous agreement rather than dividing by
        # zero and silently returning a number.
        prec = rec = None
        if self.normaliser == "union":
            score = 1.0 if h_p <= 0.0 else min(h_a / h_p, 1.0)
        else:
            # Information precision and recall. Each denominator is that side's OWN information, so
            # H(P_y) is fixed by the gold and cannot be moved by editing the candidate.
            h_x = self.joint_entropy([p for p in involved if p in px])
            h_y = self.joint_entropy([p for p in involved if p in py])
            prec = 1.0 if h_x <= 0.0 else min(h_a / h_x, 1.0)
            rec = 1.0 if h_y <= 0.0 else min(h_a / h_y, 1.0)
            score = 0.0 if prec + rec <= 0.0 else 2.0 * prec * rec / (prec + rec)

        ceiling = math.log(self.n_corpus) if self.n_corpus > 1 else 0.0
        seen = {p for o in self._obs for p in o}
        return {"score": score, "H_A": h_a, "H_P": h_p,
                "precision": prec, "recall": rec,
                "saturation": (h_p / ceiling) if ceiling > 0 else None,
                "n_positions": len(involved),
                "n_unseen_positions": len([p for p in involved if p not in seen]),
                "sigma": sigma}
