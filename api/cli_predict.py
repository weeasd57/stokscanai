import argparse
import json
import os

from dotenv import load_dotenv

from stock_ai import run_pipeline


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--from-date", default="2020-01-01")
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--out", default=None, help="Write output JSON to a file")
    args = parser.parse_args()

    api_key = os.getenv("EODHD_API_KEY")
    if not api_key:
        raise SystemExit("EODHD_API_KEY is not configured. Set it in api/.env")

    cache_dir = os.getenv("CACHE_DIR", "data_cache")

    payload = run_pipeline(
        api_key=api_key,
        ticker=args.ticker.strip().upper(),
        from_date=args.from_date,
        cache_dir=cache_dir,
        include_fundamentals=not args.no_fundamentals,
    )

    if args.out:
        out_path = os.path.abspath(args.out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
