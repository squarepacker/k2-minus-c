"""Worker for the version-2 verification: runs verify_leaves2.py on every final T=9 leaf file with START=w, STEP=4.
Usage: python run_v2.py W [WAIT_PID]   (if WAIT_PID is given, first wait until that process has exited: keeps <= 4 busy cores)"""
import sys, os, glob, subprocess, time
here = os.path.dirname(os.path.abspath(__file__))
w = int(sys.argv[1])
if len(sys.argv) > 2:
    import ctypes
    pid = int(sys.argv[2])
    k32 = ctypes.windll.kernel32
    hnd = k32.OpenProcess(0x00100000, False, pid)          # SYNCHRONIZE
    if hnd:
        k32.WaitForSingleObject(hnd, 0xFFFFFFFF); k32.CloseHandle(hnd)
files = sorted(glob.glob(os.path.join(here, '*_T9_*_final_leaves.jsonl')), key=lambda f: -os.path.getsize(f))
log = open(os.path.join(here, f'v2_worker{w}.log'), 'a')
for f in files:
    subprocess.run([sys.executable, 'verify_leaves2.py', os.path.basename(f), '9', str(w), '4'], stdout=log, stderr=subprocess.STDOUT, cwd=here)
    log.flush()
log.write('### worker done\n'); log.flush()
