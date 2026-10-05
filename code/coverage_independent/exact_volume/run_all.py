"""Run indep_cover.py on all 13 T9 leaf files, 4 processes at a time; JSON reports in results/.
Usage: python run_all.py [NPOINTS] [DATADIR]   (DATADIR defaults to code/, where data/kw9_data.zip is unpacked)"""
import subprocess, sys, os, glob, json
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__))
data = sys.argv[2] if len(sys.argv) > 2 else os.path.normpath(os.path.join(here, '..', '..'))
files = sorted(glob.glob(os.path.join(data, '*_T9_*_final_leaves.jsonl')), key=os.path.getsize, reverse=True)
os.makedirs(os.path.join(here, 'results'), exist_ok=True)
M = sys.argv[1] if len(sys.argv) > 1 else '40000'


def run(f):
    out = os.path.join(here, 'results', os.path.basename(f).replace('_final_leaves.jsonl', '.json'))
    p = subprocess.run([sys.executable, os.path.join(here, 'indep_cover.py'), f, M, '12345', out], capture_output=True, text=True)
    return p.stdout.strip() + (('\nSTDERR ' + p.stderr[-2000:]) if p.returncode else '')


print(len(files), 'files', flush=True)
with ThreadPoolExecutor(4) as ex:
    for r in ex.map(run, files): print(r, flush=True)
