"""Populate the idempotent ThreatNet V2 local demonstration workflow."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "backend"))

from api.services import demo_state  # noqa: E402


if __name__ == "__main__":
    state = demo_state()
    print(
        "ThreatNet V2 demo seeded: "
        f"{len(state['cases'])} cases, extracted source text, persisted frame vectors, "
        "a synthetic match indicator, and a spatiotemporal review alert."
    )
