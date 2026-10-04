"""Start the DermaAI web app.

    python app.py                 # http://localhost:8000
    python app.py --port 9000

The app only loads and uses the trained model in models/trained/ (the version named in
models/trained/CURRENT). It never trains. After you retrain with `python training/train.py`,
the app switches to the new model automatically — no code change and no restart needed.
"""

import argparse

import uvicorn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to allow other devices on your network")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true", help="restart on code changes (development)")
    a = p.parse_args()
    uvicorn.run("dermaai.api.main:app", host=a.host, port=a.port, reload=a.reload)


if __name__ == "__main__":
    main()
