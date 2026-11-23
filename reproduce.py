"""Portable entrypoint for Periplus; see README.md."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'methods/llm_index'))
sys.path.insert(0, str(ROOT / 'src/wiki_index'))
sys.path.insert(0, str(ROOT / 'shared'))


def configure():
    import azure_client as ac
    import pipeline
    import cold_start

    def llm(prompt, max_tokens=16384, phase='unknown'):
        bucket = 'answer' if phase.startswith(('query', 'route')) else 'build'
        text, _, _ = ac.chat([{'role': 'user', 'content': prompt}],
                             max_tokens=max_tokens, bucket=bucket,
                             usage_context={'phase': phase})
        if not text.strip():
            raise RuntimeError(f'Empty model response during {phase}')
        records = ac.get_call_records()
        if records and records[-1].get('finish_reason') == 'length':
            raise RuntimeError(f'Truncated model response during {phase}')
        return text.strip()

    pipeline.llm = cold_start.llm = llm
    pipeline.set_router(pipeline.llm_route_children)
    pipeline.set_embedder(lambda texts: ac.embed(list(texts), bucket='build')[0])
    return pipeline, ac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--source', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--id', required=True)
    build.add_argument('--title', required=True)
    for command in ('ask', 'evaluate'):
        p = sub.add_parser(command)
        p.add_argument('--source', type=Path, required=True)
        p.add_argument('--index', type=Path, required=True)
        if command == 'ask':
            p.add_argument('--question', required=True)
        else:
            p.add_argument('--questions', type=Path, required=True)
            p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pipeline, ac = configure()
    started = time.perf_counter()
    if args.command == 'build':
        import balance_window
        from verify_tree import validate_doc
        name = pipeline.safe_filename(args.id)
        raw = args.output / 'raw' / name
        balanced = args.output / 'balanced' / name
        if raw.exists() or balanced.exists():
            raise ValueError('Output already exists; choose a fresh output directory or id')
        result = pipeline.build_wiki(args.source.resolve(), args.output / 'raw', args.id, args.title)
        if result is None:
            raise RuntimeError('Build failed')
        balance_window.SRC_DIGESTS = args.output / 'raw'
        balance_window.OUT_DIGESTS = args.output / 'balanced'
        balance_window.window_doc(name)
        report = validate_doc(balanced, strict_integrity=True)
        (args.output / f'{name}-validation.json').write_text(json.dumps(report, indent=2))
        if report['errors']:
            raise RuntimeError(report['errors'])
        (args.output / f'{name}-build.json').write_text(json.dumps({
            'wall_seconds': time.perf_counter() - started,
            'calls': ac.get_call_records(), 'index': str(balanced)}, indent=2))
        print(balanced)
        return
    if args.command == 'ask':
        answer, source, trace = pipeline.query_with_traversal(args.question, args.index, args.source)
        print(json.dumps({'answer': answer, 'source': source, 'trace': trace,
                          'calls': ac.get_call_records()}, indent=2))
        return
    # Exclusive creation avoids appending duplicates to a prior experiment.
    with args.output.open('x', encoding='utf-8') as output:
        for line in args.questions.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            question = json.loads(line)
            before = len(ac.get_call_records())
            t0 = time.perf_counter()
            answer, source, trace = pipeline.query_with_traversal(question['question'], args.index, args.source)
            output.write(json.dumps({**question, 'prediction': answer, 'answer_source': source,
                'wall_seconds': time.perf_counter() - t0, 'trace': trace,
                'calls': ac.get_call_records()[before:]}, ensure_ascii=False) + '\n')
            output.flush()


if __name__ == '__main__':
    main()
