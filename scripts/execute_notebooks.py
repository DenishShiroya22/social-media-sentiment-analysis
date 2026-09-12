"""Execute notebooks with this interpreter; save outputs and a verification record."""
import json
import os
from pathlib import Path
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

root = Path(__file__).resolve().parents[1]
results = {}
with tempfile.TemporaryDirectory() as temporary:
    os.environ['IPYTHONDIR'] = str(Path(temporary) / 'ipython')
    kernel_dir = Path(temporary) / 'data-foundation'
    kernel_dir.mkdir()
    (kernel_dir / 'kernel.json').write_text(json.dumps({'argv':[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name':'Data foundation', 'language':'python'}), encoding='utf-8')
    for path in sorted((root / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name='data-foundation', kernel_spec_manager=KernelSpecManager(kernel_dirs=[temporary]))
        try:
            NotebookClient(notebook, km=manager, timeout=180, resources={'metadata': {'path': str(root)}}).execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        results[path.name] = {'status':'passed', 'executed_code_cells': sum(c.cell_type == 'code' for c in notebook.cells)}
        print(f'Executed {path.name}')
(root / 'reports' / 'notebook_execution.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
