"""Execute the five notebooks and retain their outputs for GitHub preview."""
from pathlib import Path
import argparse
import os
import time
import subprocess
import sys

# These small dense problems avoid excessive BLAS thread overhead.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

def execute_in_process(notebook):
    """Run with IPython without opening kernel ports (restricted environments)."""
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output
    from matplotlib_inline.backend_inline import flush_figures
    import matplotlib
    matplotlib.use('module://matplotlib_inline.backend_inline')
    from traitlets.config import Config
    config = Config()
    config.HistoryManager.hist_file = ':memory:'
    shell = InteractiveShell.instance(config=config)
    os.chdir(ROOT / 'notebooks')
    shell.run_cell('from matplotlib_inline.backend_inline import set_matplotlib_formats\nset_matplotlib_formats("png")', store_history=False)
    count = 0
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        count += 1
        print(f'  cell {count}', flush=True)
        with capture_output() as captured:
            result = shell.run_cell(cell.source, store_history=True)
            flush_figures()
        result.raise_error()
        outputs = []
        if captured.stdout:
            outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
        if captured.stderr:
            outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
        for output in captured.outputs:
            outputs.append(nbformat.v4.new_output('display_data', data=output.data, metadata=output.metadata))
        cell.outputs = outputs
        cell.execution_count = count
    return notebook

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook', help='Execute only this notebook filename.')
    parser.add_argument('--in-process', action='store_true', help='Use IPython without kernel ports.')
    args = parser.parse_args()
    paths = sorted((ROOT / 'notebooks').glob('*.ipynb'))
    if args.notebook:
        paths = [p for p in paths if p.name == args.notebook]
        if not paths:
            parser.error('Notebook not found.')
    for path in paths:
        if args.in_process and not args.notebook:
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--in-process', '--notebook', path.name], check=True)
            continue
        started = time.monotonic()
        print(f'Running {path.name}', flush=True)
        notebook = nbformat.read(path, as_version=4)
        if args.in_process:
            execute_in_process(notebook)
        else:
            client = NotebookClient(notebook, timeout=3600, kernel_name='python3',
                resources={'metadata': {'path': str(ROOT / 'notebooks')}})
            client.execute()
        # Execution timestamps are machine-specific and add unnecessary diffs.
        for cell in notebook.cells:
            cell.metadata.pop('execution', None)
        nbformat.write(notebook, path)
        print(f'OK {path.name}: {time.monotonic() - started:.1f}s', flush=True)

if __name__ == '__main__':
    main()
