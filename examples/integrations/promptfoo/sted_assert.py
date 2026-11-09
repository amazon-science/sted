"""promptfoo Python assertion: the output is the same as `vars.expected` under STED.

Use in promptfooconfig.yaml:

    assert:
      - type: python
        value: file://sted_assert.py

Field rules come from the test's `vars.sted_rules` (JSON), e.g.
{"ignore": ["request_id"], "unordered": ["tags"], "tolerance": {"price": 0.01}}.
"""
import json

import sted


def get_assert(output, context):
    vars_ = context.get("vars", {})
    expected = vars_["expected"]
    rules = vars_.get("sted_rules") or {}
    if isinstance(rules, str):
        rules = json.loads(rules)
    try:
        r = sted.compare(expected, output, **rules)
    except ValueError as e:                       # output is not JSON
        return {"pass": False, "score": 0.0, "reason": str(e)}
    return {"pass": r.same, "score": r.min_readout, "reason": str(r)}
