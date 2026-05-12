"""High-level agent consistency evaluator.

A user-friendly wrapper around the STED metric for evaluating consistency of
LLM agent outputs. Designed for production use cases:
  - Tool-call stability monitoring
  - Multi-step trajectory stability
  - Pre-deployment consistency audit

Example:
    from sted import AgentConsistencyEvaluator

    evaluator = AgentConsistencyEvaluator()

    def my_agent(prompt: str) -> dict:
        return llm_call(prompt)  # returns a JSON-serializable dict

    report = evaluator.evaluate(
        agent_fn=my_agent,
        prompts=["Get the weather in Seattle", "Book a flight to NYC"],
        n_runs=10,
    )

    print(report.summary())
    # Consistency: 0.842 (high)
    # Validity:    0.95
    # Worst prompt: 'Book a flight to NYC' (c_mean=0.42)

    # See per-prompt detail
    for r in report.per_prompt:
        print(f"  {r.prompt[:50]:<50} c={r.c_mean:.3f}  r_v={r.r_v:.2f}")
"""
from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import statistics
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union

from ._logging import get_logger

logger = get_logger("agent_evaluator")

JsonLike = Union[dict, list, str, int, float, bool, None]
AgentFn = Callable[[str], JsonLike]

# Module-level executor for STED timeout enforcement. Lazily created.
_TIMEOUT_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None


def _get_timeout_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazily create a shared executor for timeout-bounded STED calls."""
    global _TIMEOUT_EXECUTOR
    if _TIMEOUT_EXECUTOR is None:
        _TIMEOUT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="sted-timeout"
        )
    return _TIMEOUT_EXECUTOR


@dataclasses.dataclass
class PromptResult:
    """Consistency results for a single prompt across n_runs invocations."""

    prompt: str
    n_runs: int                   # total runs requested
    n_valid: int                  # parseable runs
    r_v: float                    # validity rate = n_valid / n_runs
    c_mean: float                 # mean pairwise consistency on valid runs
    c_std: float                  # std of pairwise consistencies
    c_adj: float                  # r_v * c_mean (deployment-aware composite)
    raw_outputs: list             # the n_runs raw outputs
    pairwise_similarities: list   # all C(n_valid, 2) pairwise STED similarities
    error: Optional[str] = None   # set if evaluation failed for this prompt


@dataclasses.dataclass
class ConsistencyReport:
    """Aggregated consistency report across all prompts."""

    per_prompt: list[PromptResult]
    n_prompts: int
    n_runs: int

    # Overall statistics (averaged across prompts)
    mean_consistency: float       # mean c_mean across prompts
    mean_validity: float          # mean r_v across prompts
    mean_c_adj: float             # mean c_adj across prompts

    # Identify problematic prompts
    worst_prompts: list[PromptResult]    # bottom-k by c_mean
    most_invalid: list[PromptResult]     # bottom-k by r_v

    metric_config: dict           # STED config used (embedding model, w, etc.)

    def summary(self) -> str:
        """Human-readable one-page summary."""
        lines = []
        lines.append(f"Agent Consistency Report")
        lines.append(f"=" * 60)
        lines.append(f"Prompts evaluated:    {self.n_prompts}")
        lines.append(f"Runs per prompt:      {self.n_runs}")
        lines.append(f"")
        lines.append(f"Mean consistency:     {self.mean_consistency:.3f}  "
                     f"({_consistency_label(self.mean_consistency)})")
        lines.append(f"Mean validity:        {self.mean_validity:.3f}")
        lines.append(f"Mean c_adj:           {self.mean_c_adj:.3f}  "
                     f"(deployment-aware)")
        lines.append(f"")
        if self.worst_prompts:
            lines.append(f"Least consistent prompts:")
            for r in self.worst_prompts[:5]:
                short = r.prompt[:60].replace("\n", " ")
                lines.append(f"  c_mean={r.c_mean:.3f}  r_v={r.r_v:.2f}  | {short}")
        if self.most_invalid:
            invalid = [r for r in self.most_invalid if r.r_v < 1.0][:5]
            if invalid:
                lines.append(f"")
                lines.append(f"Highest invalidity prompts (parseability issues):")
                for r in invalid:
                    short = r.prompt[:60].replace("\n", " ")
                    lines.append(f"  r_v={r.r_v:.2f}  c_mean={r.c_mean:.3f}  | {short}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializable summary (no raw outputs to keep it light)."""
        return {
            "n_prompts": self.n_prompts,
            "n_runs": self.n_runs,
            "mean_consistency": self.mean_consistency,
            "mean_validity": self.mean_validity,
            "mean_c_adj": self.mean_c_adj,
            "metric_config": self.metric_config,
            "per_prompt": [
                {
                    "prompt": r.prompt,
                    "n_valid": r.n_valid,
                    "r_v": r.r_v,
                    "c_mean": r.c_mean,
                    "c_std": r.c_std,
                    "c_adj": r.c_adj,
                    "error": r.error,
                }
                for r in self.per_prompt
            ],
        }


def _consistency_label(c: float) -> str:
    if c >= 0.95:
        return "very high"
    if c >= 0.85:
        return "high"
    if c >= 0.70:
        return "moderate"
    if c >= 0.50:
        return "low"
    return "very low"


class AgentConsistencyEvaluator:
    """High-level evaluator for LLM agent consistency.

    Wraps the STED metric with a workflow that:
      1. Invokes the agent n_runs times per prompt (in parallel).
      2. Computes pairwise STED similarity on parseable outputs.
      3. Aggregates per-prompt and overall consistency statistics.
      4. Identifies the least-consistent prompts.

    Typical usage:
        evaluator = AgentConsistencyEvaluator()  # default STED config
        report = evaluator.evaluate(my_agent, prompts, n_runs=10)
        print(report.summary())

    For tool-call evaluation, agent_fn should return either a dict (single
    tool call), a list of dicts (multi-tool), or a JSON string. Anything
    else is treated as invalid (counted in r_v).
    """

    def __init__(
        self,
        evaluator: Optional[Any] = None,
        structural_weight: float = 0.5,
        embedding_model: Optional[str] = None,
        max_parallel_runs: int = 8,
        n_workers: int = 1,
        timeout_seconds: Optional[float] = None,
    ):
        """
        Args:
            evaluator: Optional pre-built SemanticJsonTreeConsistencyEvaluator.
                If None, a default evaluator is created lazily.
            structural_weight: w in [0,1]; default 0.5 (hybrid).
            embedding_model: Sentence-Transformers model name (default
                'all-MiniLM-L6-v2'). Ignored if evaluator is provided.
            max_parallel_runs: Max concurrent agent invocations per prompt.
            n_workers: Number of threads to use for prompt-level
                parallelism in `evaluate` and `evaluate_outputs`. Default 1
                (sequential). Set to a value > 1 (e.g. 4-8) to evaluate
                multiple prompts concurrently. Threads share the same
                evaluator instance — its caches benefit from cross-prompt
                hits, and batch-encoding is also shared. STED is thread-safe
                under this access pattern.
            timeout_seconds: Optional per-pair timeout for STED similarity
                calculation. If a single calculate_tree_edit_distance_fast call
                exceeds this many seconds, the pair's similarity is marked None
                and the prompt's PromptResult.error is set to
                "timeout after Xs". None (default) disables the timeout.
        """
        from .structural_consistency_analyzer import StructuralConsistencyAnalyzer

        if evaluator is None:
            from .semantic_json_tree_consistency import SemanticJsonTreeConsistencyEvaluator
            kwargs = {"structural_weight": structural_weight}
            if embedding_model is not None:
                kwargs["embedding_model_name"] = embedding_model
            evaluator = SemanticJsonTreeConsistencyEvaluator(**kwargs)

        self._evaluator = evaluator
        self._analyzer = StructuralConsistencyAnalyzer(evaluator)
        self.max_parallel_runs = max_parallel_runs
        self.n_workers = max(1, int(n_workers))
        self.timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else None
        )
        self._metric_config = {
            "structural_weight": structural_weight,
            "embedding_model": embedding_model or "all-MiniLM-L6-v2",
            "n_workers": self.n_workers,
            "timeout_seconds": self.timeout_seconds,
        }

    # ---------- Public API ----------

    def evaluate(
        self,
        agent_fn: AgentFn,
        prompts: Iterable[str],
        n_runs: int = 10,
        bottom_k: int = 5,
        progress: bool = False,
    ) -> ConsistencyReport:
        """Evaluate agent consistency across `prompts`, with `n_runs` repeats each.

        Args:
            agent_fn: Callable taking a prompt string and returning a
                JSON-like value (dict, list, str, etc.). Should be
                deterministic-input/stochastic-output (LLM call).
            prompts: Iterable of prompt strings.
            n_runs: Repeats per prompt.
            bottom_k: How many least-consistent prompts to surface.
            progress: If True, print a one-line progress indicator per prompt.

        Returns:
            ConsistencyReport with per-prompt and aggregate statistics.
        """
        prompts = list(prompts)
        per_prompt: list[PromptResult] = [None] * len(prompts)  # type: ignore

        if self.n_workers <= 1 or len(prompts) <= 1:
            for i, prompt in enumerate(prompts):
                if progress:
                    print(f"[{i+1}/{len(prompts)}] {prompt[:60]}...", flush=True)
                per_prompt[i] = self._evaluate_one_prompt(agent_fn, prompt, n_runs)
        else:
            # Run prompts concurrently. STED is thread-safe under this access
            # pattern; cross-prompt embedding cache hits are an additional win.
            def _eval(idx_prompt):
                idx, prompt = idx_prompt
                if progress:
                    print(f"[{idx+1}/{len(prompts)}] {prompt[:60]}...", flush=True)
                return idx, self._evaluate_one_prompt(agent_fn, prompt, n_runs)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                for idx, result in pool.map(_eval, enumerate(prompts)):
                    per_prompt[idx] = result

        return self._aggregate(per_prompt, n_runs, bottom_k)

    def evaluate_outputs(
        self,
        outputs_per_prompt: dict[str, list[JsonLike]],
        bottom_k: int = 5,
        precompute_embeddings: bool = True,
        progress: Union[bool, str] = False,
    ) -> ConsistencyReport:
        """Evaluate from pre-collected outputs (no agent invocation).

        Useful when you have logs from production and want to retroactively
        score consistency without re-running the agent.

        Args:
            outputs_per_prompt: Dict mapping prompt string to list of outputs
                (one per run). Lengths can vary across prompts.
            bottom_k: How many least-consistent prompts to surface.
            precompute_embeddings: If True (default), batch-encode all unique
                strings across all prompts up front. This is much faster than
                lazy per-pair encoding for large batches and is essential when
                using multiple workers (avoids contention on the
                SentenceTransformer model).
            progress: If True, print "[i/N] processing..." every 10 prompts.
                If "tqdm", use tqdm if available (falls back to print otherwise).
        """
        items = list(outputs_per_prompt.items())

        # === Optimization 1: cross-prompt batch encoding =====================
        # Encode all unique strings across all outputs in one model call,
        # before any per-pair STED computation. This both reduces overhead and
        # warms the embedding cache for cross-prompt hits.
        if precompute_embeddings:
            self._precompute_for_outputs(items)

        # Build a progress reporter usable in both serial and threaded paths.
        reporter = _make_progress_reporter(progress, total=len(items))

        # === Optimization 2: prompt-level threading ==========================
        per_prompt: list[Optional[PromptResult]] = [None] * len(items)
        if self.n_workers <= 1 or len(items) <= 1:
            for i, (prompt, outputs) in enumerate(items):
                per_prompt[i] = self._score_outputs(prompt, outputs)
                reporter.update(1)
        else:
            def _score(idx_pair):
                idx, (prompt, outputs) = idx_pair
                return idx, self._score_outputs(prompt, outputs)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                for idx, result in pool.map(_score, enumerate(items)):
                    per_prompt[idx] = result
                    reporter.update(1)

        reporter.close()

        n_runs_max = max((len(o) for _, o in items), default=0)
        return self._aggregate([r for r in per_prompt if r is not None],
                               n_runs_max, bottom_k)

    def evaluate_outputs_streaming(
        self,
        outputs_iterable: Iterable[Tuple[str, "list[JsonLike]"]],
        chunk_size: int = 100,
        progress: Union[bool, str] = False,
    ) -> Iterator[PromptResult]:
        """Streaming counterpart of `evaluate_outputs`.

        Yields PromptResult objects one at a time so callers can process
        massive batches without holding all results (or all inputs) in RAM.

        Args:
            outputs_iterable: Iterable yielding (prompt, outputs) tuples. Can
                be a generator — items are consumed lazily.
            chunk_size: When n_workers > 1, prompts are scored in chunks of
                this size to bound memory. Default 100.
            progress: True / "tqdm" — see `evaluate_outputs`.

        Notes:
            Cross-prompt batch precompute is skipped (we don't see all inputs
            up front); the per-call auto-precompute inside STED still applies.
        """
        reporter = _make_progress_reporter(progress, total=None)
        n_done = 0

        if self.n_workers <= 1:
            for prompt, outputs in outputs_iterable:
                result = self._score_outputs(prompt, outputs)
                n_done += 1
                reporter.update(1)
                yield result
            reporter.close()
            return

        # Threaded path: consume in chunks to bound memory.
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as pool:
            it = iter(outputs_iterable)
            while True:
                chunk: list[Tuple[str, list]] = []
                for _ in range(chunk_size):
                    try:
                        chunk.append(next(it))
                    except StopIteration:
                        break
                if not chunk:
                    break

                def _score(pair):
                    prompt, outputs = pair
                    return self._score_outputs(prompt, outputs)

                for result in pool.map(_score, chunk):
                    n_done += 1
                    reporter.update(1)
                    yield result
        reporter.close()

    # ---------- Cache monitoring ----------

    def cache_stats(self) -> dict:
        """Return cache size + hit-rate stats from the underlying STED evaluator.

        Keys:
            - embedding_cache_size: number of pre-computed string embeddings.
            - subtree_cache_size: current entries in the subtree LRU cache.
            - subtree_cache_hit_rate: hits / (hits + misses) since last clear.

        To reset the subtree cache:
            >>> evaluator._evaluator.clear_subtree_cache()
        """
        try:
            stats = self._evaluator.get_cache_stats()
        except AttributeError:
            return {
                "embedding_cache_size": 0,
                "subtree_cache_size": 0,
                "subtree_cache_hit_rate": 0.0,
            }
        return {
            "embedding_cache_size": stats.get("embedding_cache_size", 0),
            "subtree_cache_size": stats.get("subtree_cache_size", 0),
            "subtree_cache_hit_rate": stats.get("subtree_cache_hit_rate", 0.0),
        }

    def _precompute_for_outputs(
        self,
        items: list,
    ) -> int:
        """Batch-encode all unique strings appearing in `items`.

        Returns the number of newly-encoded strings.
        Silently no-ops when the underlying evaluator does not have a
        SentenceTransformer (e.g. Bedrock-only mode).
        """
        ev = self._evaluator
        if not hasattr(ev, "embedding_model") or ev.embedding_model is None:
            return 0

        try:
            # Skip the per-call auto-precompute inside calculate_tree_edit_distance_fast
            # since we're doing it once for the whole batch.
            ev._skip_auto_precompute = True

            all_strings = set()
            for _, outputs in items:
                for output in outputs:
                    if output is not None and not (
                        isinstance(output, (dict, list)) and len(output) == 0
                    ):
                        try:
                            ev.collect_strings_from_json(output, all_strings)
                        except Exception:
                            pass

            new_strings = [s for s in all_strings
                           if s not in ev._embedding_dict]
            if new_strings:
                ev._batch_encode_sentence_transformer(
                    new_strings, batch_size=64, show_progress=False
                )
            return len(new_strings)
        except Exception:
            return 0
        finally:
            ev._skip_auto_precompute = False

    def evaluate_pair(self, output1: JsonLike, output2: JsonLike) -> float:
        """Score similarity between exactly two outputs (returns STED in [0,1])."""
        if not _is_valid(output1) or not _is_valid(output2):
            return 0.0
        return self._evaluator.calculate_tree_edit_distance_fast(output1, output2)

    # ---------- Internals ----------

    def _evaluate_one_prompt(
        self,
        agent_fn: AgentFn,
        prompt: str,
        n_runs: int,
    ) -> PromptResult:
        """Invoke agent_fn n_runs times in parallel and score the outputs."""
        outputs: list[JsonLike] = [None] * n_runs
        errors: list[Optional[str]] = [None] * n_runs

        def _run(idx: int) -> None:
            try:
                outputs[idx] = agent_fn(prompt)
            except Exception as e:  # capture but continue
                errors[idx] = f"{type(e).__name__}: {e}"

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_parallel_runs
        ) as pool:
            list(pool.map(_run, range(n_runs)))

        result = self._score_outputs(prompt, outputs)
        # Surface a representative agent error if any runs raised. Agent
        # errors are more informative than the generic "fewer than 2 valid"
        # diagnostic, so they take precedence.
        agent_errors = [e for e in errors if e is not None]
        if agent_errors:
            unique_errors = list(dict.fromkeys(agent_errors))  # preserve order
            n = len(agent_errors)
            sample = unique_errors[0]
            if len(unique_errors) > 1:
                result.error = (f"{n} agent invocations raised; "
                                f"first error: {sample}")
            else:
                result.error = f"{n} agent invocations raised: {sample}"
        return result

    def _compute_pair_similarity(
        self,
        output1: JsonLike,
        output2: JsonLike,
    ) -> Tuple[Optional[float], Optional[str]]:
        """Compute STED similarity between two outputs with timeout + fallback.

        Returns:
            (similarity, error_label):
              - similarity is a float in [0,1] if computed (possibly via fallback);
                None only if the pair timed out.
              - error_label is a short string describing any error/timeout, else None.
        """
        timeout = self.timeout_seconds

        def _compute():
            return self._evaluator.calculate_tree_edit_distance_fast(output1, output2)

        try:
            if timeout is None:
                s = _compute()
            else:
                executor = _get_timeout_executor()
                future = executor.submit(_compute)
                try:
                    s = future.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    # Best-effort cancel; if already running, the worker will
                    # finish but its result will be discarded.
                    future.cancel()
                    logger.warning(
                        "STED timed out for pair after %.3fs", timeout,
                    )
                    return None, f"timeout after {timeout}s"
            return float(s), None
        except Exception:
            logger.warning(
                "STED failed for pair; using exact-match fallback",
                exc_info=True,
            )
            fallback = 1.0 if output1 == output2 else 0.0
            return fallback, "sted_error_fallback"

    def _score_outputs(self, prompt: str, outputs: list[JsonLike]) -> PromptResult:
        """Score a list of outputs for a single prompt."""
        n_runs = len(outputs)
        if n_runs == 0:
            return PromptResult(
                prompt=prompt, n_runs=0, n_valid=0, r_v=0.0,
                c_mean=0.0, c_std=0.0, c_adj=0.0,
                raw_outputs=[], pairwise_similarities=[],
                error="no outputs",
            )

        valid = [o for o in outputs if _is_valid(o)]
        n_valid = len(valid)
        r_v = n_valid / n_runs

        if n_valid < 2:
            return PromptResult(
                prompt=prompt, n_runs=n_runs, n_valid=n_valid, r_v=r_v,
                c_mean=0.0, c_std=0.0, c_adj=0.0,
                raw_outputs=outputs, pairwise_similarities=[],
                error="fewer than 2 valid outputs" if n_valid < 2 else None,
            )

        # Compute all C(n_valid, 2) pairwise similarities.
        sims: list = []  # may include None on timeout
        timeout_err: Optional[str] = None
        n_fallback = 0
        for i in range(n_valid):
            for j in range(i + 1, n_valid):
                s, err = self._compute_pair_similarity(valid[i], valid[j])
                sims.append(s)
                if err and err.startswith("timeout"):
                    timeout_err = err
                elif err == "sted_error_fallback":
                    n_fallback += 1

        # Filter out None (timed-out pairs) for stats.
        finite_sims = [s for s in sims if s is not None]
        if not finite_sims:
            return PromptResult(
                prompt=prompt, n_runs=n_runs, n_valid=n_valid, r_v=r_v,
                c_mean=0.0, c_std=0.0, c_adj=0.0,
                raw_outputs=outputs, pairwise_similarities=sims,
                error=timeout_err or "all pairs failed",
            )

        c_mean = statistics.mean(finite_sims)
        c_std = statistics.stdev(finite_sims) if len(finite_sims) > 1 else 0.0
        c_adj = r_v * c_mean

        error_msg: Optional[str] = None
        if timeout_err:
            error_msg = timeout_err
        elif n_fallback:
            error_msg = f"sted_error_fallback used for {n_fallback} pair(s)"

        return PromptResult(
            prompt=prompt,
            n_runs=n_runs,
            n_valid=n_valid,
            r_v=r_v,
            c_mean=c_mean,
            c_std=c_std,
            c_adj=c_adj,
            raw_outputs=outputs,
            pairwise_similarities=sims,
            error=error_msg,
        )

    def _aggregate(
        self,
        per_prompt: list[PromptResult],
        n_runs: int,
        bottom_k: int,
    ) -> ConsistencyReport:
        valid_results = [r for r in per_prompt if r.n_valid >= 2]
        if valid_results:
            mean_c = statistics.mean(r.c_mean for r in valid_results)
            mean_rv = statistics.mean(r.r_v for r in per_prompt)
            mean_cadj = statistics.mean(r.c_adj for r in per_prompt)
        else:
            mean_c = 0.0
            mean_rv = statistics.mean(r.r_v for r in per_prompt) if per_prompt else 0.0
            mean_cadj = 0.0

        worst = sorted(valid_results, key=lambda r: r.c_mean)[:bottom_k]
        most_invalid = sorted(per_prompt, key=lambda r: r.r_v)[:bottom_k]

        return ConsistencyReport(
            per_prompt=per_prompt,
            n_prompts=len(per_prompt),
            n_runs=n_runs,
            mean_consistency=mean_c,
            mean_validity=mean_rv,
            mean_c_adj=mean_cadj,
            worst_prompts=worst,
            most_invalid=most_invalid,
            metric_config=dict(self._metric_config),
        )


class _PrintProgress:
    """Lightweight progress reporter that prints every N completions."""

    def __init__(self, total: Optional[int], every: int = 10):
        self.total = total
        self.every = max(1, every)
        self.n = 0

    def update(self, k: int = 1) -> None:
        self.n += k
        if self.n % self.every == 0 or (self.total and self.n == self.total):
            if self.total:
                print(f"[{self.n}/{self.total}] processing...", flush=True)
            else:
                print(f"[{self.n}] processing...", flush=True)

    def close(self) -> None:
        pass


class _NoopProgress:
    def update(self, k: int = 1) -> None:  # noqa: D401
        return

    def close(self) -> None:
        return


def _make_progress_reporter(
    progress: Union[bool, str], total: Optional[int]
):
    """Build a progress reporter for evaluate_outputs / streaming."""
    if not progress:
        return _NoopProgress()
    if progress == "tqdm":
        try:
            from tqdm import tqdm  # type: ignore

            class _TqdmAdapter:
                def __init__(self, total):
                    self._bar = tqdm(total=total) if total else tqdm()

                def update(self, k: int = 1) -> None:
                    self._bar.update(k)

                def close(self) -> None:
                    self._bar.close()

            return _TqdmAdapter(total)
        except ImportError:
            # Fall back to simple print progress
            return _PrintProgress(total)
    return _PrintProgress(total)


def _is_valid(output: JsonLike, strict_json: bool = False) -> bool:
    """An output is valid if it's a non-empty parseable JSON-like value.

    Args:
        output: The output value.
        strict_json: If True, plain (non-JSON) strings are rejected as invalid.
            If False (default), any non-empty string is considered valid (it
            will be compared as a leaf node).
    """
    if output is None:
        return False
    if isinstance(output, str):
        s = output.strip()
        if not s:
            return False
        # Try to parse as JSON; if successful, recurse on the parsed value.
        try:
            parsed = json.loads(s)
            return _is_valid(parsed, strict_json=strict_json)
        except (json.JSONDecodeError, ValueError):
            if strict_json:
                return False
            # Plain non-empty string is valid (will be compared as leaf).
            return True
    if isinstance(output, dict):
        return len(output) > 0
    if isinstance(output, list):
        # A list is valid if it has at least one valid element.
        return any(_is_valid(item, strict_json=strict_json) for item in output)
    if isinstance(output, (int, float, bool)):
        return True
    return False
