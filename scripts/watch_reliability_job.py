"""Bounded read-only watch for this task's already authorized Kaggle job."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--interval', type=int, default=45)
    p.add_argument('--max-hours', type=float, default=5.75)
    args = p.parse_args()
    credential = json.loads((ROOT.parent/'kaggle_api_key/kaggle.json').read_text())
    os.environ['KAGGLE_USERNAME'], os.environ['KAGGLE_KEY'] = credential['username'], credential['key']
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    started, last, errors = time.monotonic(), None, 0
    while time.monotonic()-started < args.max_hours*3600:
        state = json.loads(args.state.read_text())
        owner, slug = state['kernel'].split('/')
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = api.process_response(api.kernel_status_with_http_info(owner, slug, _request_timeout=30))
            status = result.get('status', 'unknown')
            state['status'] = status
            state['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
            temporary = args.state.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(state, indent=2)+'\n')
            temporary.replace(args.state)
            errors = 0
            if status != last:
                print(json.dumps({'status': status, 'checked_at_utc': state['checked_at_utc'],
                                  'url': state.get('url')}), flush=True)
                last = status
            if status == 'complete':
                return
            if status in ['error', 'failed', 'cancelled', 'canceled']:
                raise SystemExit(2)
        except Exception as error:
            errors += 1
            print(json.dumps({'watch_error_type': type(error).__name__, 'consecutive_errors': errors}), flush=True)
            if errors >= 3:
                raise SystemExit(3) from None
        time.sleep(min(args.interval, 60))
    print(json.dumps({'watch_timeout': True, 'job_cancelled': False}), flush=True)
    raise SystemExit(4)


if __name__ == '__main__':
    main()
