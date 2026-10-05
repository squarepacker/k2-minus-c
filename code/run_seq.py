"""Run several B&B jobs one after another in this process (for the small cases). Each job: script n T task ntask secs.
Environment: BNB_DUMP=1, BNB_TAG=_final are set here for the children.  Usage: python run_seq.py LOGFILE 'script n T task ntask secs' ..."""
import subprocess, sys, os
log = open(sys.argv[1], 'a')
env = dict(os.environ, BNB_DUMP='1', BNB_TAG='_final')
for job in sys.argv[2:]:
    args = [sys.executable] + job.split()
    log.write('### ' + job + '\n'); log.flush()
    r = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=os.path.dirname(os.path.abspath(__file__)))
    log.write(f'### exit {r.returncode}\n'); log.flush()
