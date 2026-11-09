"""A stand-in for your agent, so every example runs offline.

`book_flight(run)` returns a tool call; run 1 drifts to another date, the others agree.
Replace it with your own agent or model call."""
REFERENCE = {"name": "book_flight", "arguments": {"to": "JFK", "date": "2026-12-01",
                                                  "passengers": 2, "request_id": "ref"}}


def book_flight(run: int) -> dict:
    return {"name": "book_flight",
            "arguments": {"to": "JFK", "date": "2026-12-02" if run == 1 else "2026-12-01",
                          "passengers": "2", "request_id": f"r-{run}"}}
