"""Enable `python -m trading.publish` (delegates to publish.main)."""
from trading.publish.publish import main

if __name__ == "__main__":
    raise SystemExit(main())
