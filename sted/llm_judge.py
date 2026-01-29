"""
LLM-as-Judge Baseline for JSON Consistency Evaluation

This module implements an LLM-based evaluation approach where a language model
(Claude or GPT-4) is used to assess the structural and semantic consistency
between two JSON outputs.

This serves as a baseline comparison for STED, representing how well
state-of-the-art LLMs can evaluate JSON consistency without specialized metrics.
"""

import json
import re
from typing import Dict, Any, Optional, List, Tuple
import warnings


class LLMJudge:
    """
    LLM-as-Judge for evaluating JSON consistency.

    Uses Claude (via Bedrock) or OpenAI GPT-4 to rate similarity between JSON objects.
    """

    # Prompt template for consistency evaluation
    # Purpose: Evaluate JSON similarity for LLM output consistency measurement
    # Expected behavior: More semantic differences → LOWER similarity score
    CONSISTENCY_PROMPT = """You are a JSON similarity evaluator. Your task is to measure how SIMILAR two JSON objects are.

TASK: Compare JSON 2 against JSON 1 (the reference) and output a similarity score.

CRITICAL RULES:
1. JSON 1 is the REFERENCE (ground truth)
2. Score measures how much of JSON 1's content is PRESERVED in JSON 2
3. MORE DIFFERENCES = LOWER SCORE (this is essential!)
4. IDENTICAL JSONs = 1.0, COMPLETELY DIFFERENT = 0.0

SCORING FORMULA:
similarity = 1.0 - (number_of_different_fields / total_fields)

SCORE INTERPRETATION:
- 1.0: Identical content
- 0.9: ~10% of values changed
- 0.8: ~20% of values changed
- 0.7: ~30% of values changed
- 0.6: ~40% of values changed
- 0.5: ~50% of values changed (half the content differs)
- 0.4: ~60% of values changed
- 0.3: ~70% of values changed
- 0.2: ~80% of values changed
- 0.1: ~90% of values changed
- 0.0: Completely different or incompatible structure

WHAT COUNTS AS A DIFFERENCE:
- Different string values (even if semantically similar, e.g., "active" vs "enabled")
- Different numeric values
- Different boolean values
- Missing or added fields
- Different array contents or ordering

REFERENCE JSON (JSON 1):
```json
{json1}
```

COMPARISON JSON (JSON 2):
```json
{json2}
```

STEP-BY-STEP ANALYSIS:
1. List all fields in JSON 1
2. For each field, check if JSON 2 has the EXACT same value
3. Count: DIFFERENT_FIELDS = fields with different or missing values
4. Count: TOTAL_FIELDS = all comparable fields
5. Calculate: SEMANTIC_SCORE = 1.0 - (DIFFERENT_FIELDS / TOTAL_FIELDS)
6. STRUCTURAL_SCORE = 1.0 if same keys exist, else penalize missing/extra keys
7. OVERALL_SCORE = 0.3 * STRUCTURAL_SCORE + 0.7 * SEMANTIC_SCORE

OUTPUT (use exact format):
DIFFERENT_FIELDS: [integer]
TOTAL_FIELDS: [integer]
STRUCTURAL_SCORE: [0.0-1.0]
SEMANTIC_SCORE: [0.0-1.0]
OVERALL_SCORE: [0.0-1.0]
REASONING: [one sentence]"""

    PAIRWISE_PROMPT = """You are evaluating the consistency of multiple JSON outputs that should represent the same data.

Rate the overall consistency of this set of JSON outputs on a scale from 0.0 to 1.0:
- 1.0 = All outputs are identical or semantically equivalent
- 0.8-0.99 = Very consistent with minor variations
- 0.5-0.79 = Moderately consistent
- 0.2-0.49 = Low consistency with significant variations
- 0.0-0.19 = Highly inconsistent or incompatible outputs

JSON Outputs:
{json_outputs}

Provide your evaluation:
CONSISTENCY_SCORE: [0.0-1.0]
REASONING: [Brief explanation of consistency patterns observed]"""

    def __init__(
        self,
        provider: str = "bedrock",
        model_id: str = "global.anthropic.claude-opus-4-5-20251101-v1:0",
        region_name: str = "us-west-2",
        openai_api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 8000
    ):
        """
        Initialize LLM Judge.

        Args:
            provider: "bedrock" for AWS Bedrock (Claude), "openai" for OpenAI
            model_id: Model identifier
            region_name: AWS region for Bedrock
            openai_api_key: OpenAI API key (required if provider="openai")
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.model_id = model_id
        self.temperature = temperature
        self.max_tokens = max_tokens

        if provider == "bedrock":
            self._init_bedrock(region_name)
        elif provider == "openai":
            self._init_openai(openai_api_key)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'bedrock' or 'openai'")

    def _init_bedrock(self, region_name: str):
        """Initialize AWS Bedrock client."""
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                read_timeout=60
            )
            self.client = boto3.client(
                'bedrock-runtime',
                region_name=region_name,
                config=config
            )
        except ImportError:
            raise ImportError("boto3 is required for Bedrock. Install with: pip install boto3")

    def _init_openai(self, api_key: Optional[str]):
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai is required. Install with: pip install openai")

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return response text."""
        if self.provider == "bedrock":
            return self._call_bedrock(prompt)
        else:
            return self._call_openai(prompt)

    def _call_bedrock(self, prompt: str) -> str:
        """Call Claude via Bedrock."""
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body)
        )

        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI GPT model."""
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content

    def _parse_scores(self, response: str) -> Dict[str, float]:
        """Parse scores from LLM response."""
        scores = {}

        # Extract structural score
        structural_match = re.search(r'STRUCTURAL_SCORE:\s*([\d.]+)', response)
        if structural_match:
            scores['structural'] = float(structural_match.group(1))

        # Extract semantic score
        semantic_match = re.search(r'SEMANTIC_SCORE:\s*([\d.]+)', response)
        if semantic_match:
            scores['semantic'] = float(semantic_match.group(1))

        # Extract overall score
        overall_match = re.search(r'OVERALL_SCORE:\s*([\d.]+)', response)
        if overall_match:
            scores['overall'] = float(overall_match.group(1))

        # Extract consistency score (for pairwise evaluation)
        consistency_match = re.search(r'CONSISTENCY_SCORE:\s*([\d.]+)', response)
        if consistency_match:
            scores['consistency'] = float(consistency_match.group(1))

        # Clamp all scores to [0, 1]
        for key in scores:
            scores[key] = max(0.0, min(1.0, scores[key]))

        return scores

    def calculate_similarity(
        self,
        json1: Dict[str, Any],
        json2: Dict[str, Any],
        **kwargs
    ) -> float:
        """
        Calculate similarity between two JSON objects using LLM judgment.

        Args:
            json1: First JSON object
            json2: Second JSON object

        Returns:
            Similarity score between 0 and 1
        """
        # Format JSON objects
        json1_str = json.dumps(json1, indent=2, default=str)
        json2_str = json.dumps(json2, indent=2, default=str)

        # Truncate if too long (reduced for faster processing)
        max_chars = 1500
        if len(json1_str) > max_chars:
            json1_str = json1_str[:max_chars] + "\n... (truncated)"
        if len(json2_str) > max_chars:
            json2_str = json2_str[:max_chars] + "\n... (truncated)"

        # Build prompt
        prompt = self.CONSISTENCY_PROMPT.format(
            json1=json1_str,
            json2=json2_str
        )

        try:
            response = self._call_llm(prompt)
            scores = self._parse_scores(response)

            # Return overall score, or compute from components
            if 'overall' in scores:
                return scores['overall']
            elif 'structural' in scores and 'semantic' in scores:
                # Weight structural slightly higher (0.6/0.4)
                return 0.6 * scores['structural'] + 0.4 * scores['semantic']
            else:
                warnings.warn(f"Could not parse LLM response: {response[:200]}")
                return 0.5  # Default to neutral

        except Exception as e:
            warnings.warn(f"LLM call failed: {e}")
            return 0.5  # Default to neutral on error

    def calculate_batch_consistency(
        self,
        json_outputs: List[Dict[str, Any]],
        method: str = "pairwise_avg"
    ) -> Dict[str, float]:
        """
        Calculate consistency score for a batch of JSON outputs.

        Args:
            json_outputs: List of JSON outputs to evaluate
            method: "pairwise_avg" (average of all pairs) or "holistic" (single LLM call)

        Returns:
            Dict with consistency metrics
        """
        if len(json_outputs) < 2:
            return {'consistency_score': 1.0, 'valid_count': len(json_outputs)}

        if method == "pairwise_avg":
            return self._pairwise_consistency(json_outputs)
        else:
            return self._holistic_consistency(json_outputs)

    def _pairwise_consistency(
        self,
        json_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate consistency using pairwise comparisons."""
        from itertools import combinations
        import numpy as np

        similarities = []
        for j1, j2 in combinations(json_outputs, 2):
            sim = self.calculate_similarity(j1, j2)
            similarities.append(sim)

        if not similarities:
            return {'consistency_score': 1.0, 'valid_count': len(json_outputs)}

        return {
            'consistency_score': float(np.mean(similarities)),
            'std_similarity': float(np.std(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'num_pairs': len(similarities),
            'valid_count': len(json_outputs)
        }

    def _holistic_consistency(
        self,
        json_outputs: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate consistency using a single holistic LLM judgment."""
        # Format all outputs
        outputs_str = ""
        for i, output in enumerate(json_outputs[:5], 1):  # Limit to 5 for context
            json_str = json.dumps(output, indent=2, default=str)
            if len(json_str) > 1000:
                json_str = json_str[:1000] + "\n... (truncated)"
            outputs_str += f"\nOutput {i}:\n```json\n{json_str}\n```\n"

        prompt = self.PAIRWISE_PROMPT.format(json_outputs=outputs_str)

        try:
            response = self._call_llm(prompt)
            scores = self._parse_scores(response)

            return {
                'consistency_score': scores.get('consistency', 0.5),
                'valid_count': len(json_outputs)
            }
        except Exception as e:
            warnings.warn(f"LLM call failed: {e}")
            return {'consistency_score': 0.5, 'valid_count': len(json_outputs)}


def create_llm_judge(
    provider: str = "bedrock",
    model_id: Optional[str] = None,
    **kwargs
) -> LLMJudge:
    """
    Factory function to create an LLM Judge.

    Args:
        provider: "bedrock" or "openai"
        model_id: Model to use (defaults based on provider)
        **kwargs: Additional arguments for LLMJudge

    Returns:
        LLMJudge instance
    """
    if model_id is None:
        if provider == "bedrock":
            model_id = "global.anthropic.claude-opus-4-5-20251101-v1:0"
        else:
            model_id = "gpt-4o"

    return LLMJudge(provider=provider, model_id=model_id, **kwargs)


if __name__ == "__main__":
    # Example usage
    judge = create_llm_judge(provider="bedrock")

    json1 = {
        "user": {"name": "John Doe", "age": 30},
        "status": "active",
        "roles": ["admin", "user"]
    }

    json2 = {
        "user": {"name": "John Doe", "age": 30},
        "status": "enabled",  # Different value, same meaning
        "roles": ["user", "admin"]  # Same items, different order
    }

    json3 = {
        "person": {"full_name": "John Doe"},  # Structural change
        "active": True,  # Type change
        "permissions": ["admin"]  # Renamed field
    }

    print("LLM-as-Judge Baseline Demo")
    print("=" * 50)

    # Pairwise comparison
    sim_1_2 = judge.calculate_similarity(json1, json2)
    print(f"JSON1 vs JSON2 (minor differences): {sim_1_2:.3f}")

    sim_1_3 = judge.calculate_similarity(json1, json3)
    print(f"JSON1 vs JSON3 (structural changes): {sim_1_3:.3f}")

    # Batch consistency
    batch_result = judge.calculate_batch_consistency([json1, json2, json3])
    print(f"\nBatch consistency: {batch_result}")
