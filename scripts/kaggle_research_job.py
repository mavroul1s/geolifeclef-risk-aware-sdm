"""Private, bounded Kaggle research jobs; credentials never enter upload folders."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2)+'\n')


def canonical_kernel_reference(url, requested, owner):
    """Kaggle can derive a new slug from the title instead of the requested ID."""
    if url is None:
        return requested
    parsed = urlsplit(url)
    match = re.fullmatch(r'/code/([^/]+/[^/]+)/?', parsed.path)
    if parsed.scheme != 'https' or parsed.netloc not in ['www.kaggle.com', 'kaggle.com'] or not match:
        raise ValueError('Unexpected pushed kernel URL')
    actual = match.group(1)
    if actual.split('/')[0] != owner:
        raise ValueError('Pushed kernel URL belongs to an unexpected owner')
    return actual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['push', 'status', 'collect', 'logs'])
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--credential', type=Path, default=ROOT.parent/'kaggle_api_key/kaggle.json')
    parser.add_argument('--notebook', type=Path)
    parser.add_argument('--slug')
    parser.add_argument('--title')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--internet', action='store_true', help='Enable public artifact downloads')
    parser.add_argument('--public-inputs-only', action='store_true', help='Do not attach competition data')
    parser.add_argument('--kernel-source', action='append', default=[],
                        help='Completed private output owned by this credential account')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow', nargs='+', default=[])
    parser.add_argument('--max-bytes', type=int, default=1_500_000_000)
    args = parser.parse_args()
    credential = json.loads(args.credential.read_text())
    os.environ['KAGGLE_USERNAME'] = credential['username']
    os.environ['KAGGLE_KEY'] = credential['key']
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    try:
        if args.action == 'push':
            if not args.notebook or not args.notebook.is_file() or not re.fullmatch(r'[a-z0-9-]{6,50}', args.slug or ''):
                raise ValueError('A notebook and a valid new job slug are required')
            if not args.title or len(args.title) > 50:
                raise ValueError('A title of at most 50 characters is required')
            if state.get('status') in ['running', 'queued', 'push_requested']:
                raise ValueError('Refusing to overwrite an unfinished job')
            for source in args.kernel_source:
                if not re.fullmatch(re.escape(credential['username'])+r'/[a-z0-9-]{6,70}', source):
                    raise ValueError('Kernel inputs must belong to the credential account')
                owner, slug = source.split('/')
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    source_status = api.process_response(api.kernel_status_with_http_info(owner, slug, _request_timeout=30))
                if source_status.get('status') != 'complete':
                    raise ValueError('Kernel source is not complete')
            stage = ROOT/'artifacts/kaggle_research_jobs'/args.slug
            stage.mkdir(parents=True, exist_ok=True)
            # Only the explicitly named notebook and metadata are uploaded.
            name = 'experiment.ipynb'
            shutil.copy2(args.notebook, stage/name)
            ref = credential['username']+'/'+args.slug
            meta = {'id': ref, 'title': args.title, 'code_file': name, 'language': 'python',
                'kernel_type': 'notebook', 'is_private': True, 'enable_gpu': args.gpu,
                'enable_tpu': False, 'enable_internet': args.internet, 'dataset_sources': [],
                'competition_sources': [] if args.public_inputs_only else ['geolifeclef-2025'],
                'kernel_sources': args.kernel_source}
            save(stage/'kernel-metadata.json', meta)
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                response = api.kernels_push(str(stage))
            url = getattr(response, 'url', None)
            actual_ref = canonical_kernel_reference(url, ref, credential['username'])
            state = {'status': 'push_requested', 'kernel': actual_ref, 'requested_kernel': ref, 'url': url,
                'private': True, 'GPU_requested': args.gpu, 'competition_submission_made': False,
                'internet_enabled': args.internet, 'public_inputs_only': args.public_inputs_only,
                'kernel_sources': args.kernel_source,
                'requested_at_utc': datetime.now(timezone.utc).isoformat(),
                'notebook_sha256': hashlib.sha256(args.notebook.read_bytes()).hexdigest()}
        elif args.action == 'status':
            owner, slug = state['kernel'].split('/')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = api.process_response(api.kernel_status_with_http_info(owner, slug, _request_timeout=30))
            state['status'] = result.get('status', 'unknown')
            state['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
        elif args.action == 'logs':
            owner, slug = state['kernel'].split('/')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = api.process_response(api.kernel_output_with_http_info(owner, slug, _request_timeout=30))
            log = str(result.get('log', ''))[-8000:]
            for secret in [credential['key']]:
                log = log.replace(secret, '[redacted]')
            log = re.sub(r'https?://[^\s"<>]+', '[URL redacted]', log)
            print(json.dumps({'status': state.get('status'), 'log_tail': log,
                              'available_files': [i['fileName'] for i in result.get('files', [])]}))
            return
        else:
            if not args.output or not args.allow or state.get('status') != 'complete':
                raise ValueError('Completed status, an output directory and exact allowlist are required')
            owner, slug = state['kernel'].split('/')
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                response = api.process_response(api.kernel_output_with_http_info(owner, slug, _request_timeout=30))
            import requests
            allowed = set(args.allow)
            if any(PurePosixPath(name).is_absolute() or '\\' in name or
                   any(part in ['..', '.'] for part in name.split('/')) for name in allowed):
                raise ValueError('Output allowlist must use safe relative filenames')
            args.output.mkdir(parents=True, exist_ok=True)
            files, total = [], 0
            for item in response['files']:
                name = item['fileName']
                if name not in allowed:
                    continue
                destination = args.output/name
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_symlink():
                    raise ValueError('Symlink output destination rejected')
                temporary = destination.with_suffix(destination.suffix+'.partial')
                digest, size = hashlib.sha256(), 0
                # Signed storage requests do not receive Kaggle authorization.
                with requests.get(item['url'], stream=True, timeout=(30, 180)) as download:
                    download.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in download.iter_content(1024*1024):
                            size += len(chunk)
                            if total+size > args.max_bytes:
                                raise ValueError('Research output size cap exceeded')
                            digest.update(chunk)
                            stream.write(chunk)
                temporary.replace(destination)
                total += size
                files.append({'name': name, 'bytes': size, 'sha256': digest.hexdigest()})
            if {f['name'] for f in files} != allowed:
                raise ValueError('Expected outputs missing from completed job')
            state['output_files'], state['output_bytes'] = files, total
        save(args.state, state)
        print(json.dumps({k: v for k, v in state.items() if k not in ['output_files']}))
    except Exception as error:
        diagnostic = str(getattr(error, 'body', ''))[:1500]
        for secret in [credential['key'], credential['username']]:
            diagnostic = diagnostic.replace(secret, '[redacted]')
        diagnostic = re.sub(r'https?://[^\s"<>]+', '[URL redacted]', diagnostic)
        failure = {'action': args.action, 'error_type': type(error).__name__,
            'http_status': getattr(error, 'status', None), 'diagnostic': diagnostic}
        state.setdefault('attempt_errors', []).append(failure)
        save(args.state, state)
        print(json.dumps(failure))
        raise SystemExit(2) from None


if __name__ == '__main__':
    main()
