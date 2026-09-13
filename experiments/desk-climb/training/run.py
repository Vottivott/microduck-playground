"""Portable entry point for the preserved training/evaluation recipes."""
from pathlib import Path
import os,sys,subprocess
P=Path(__file__).parent;root=P.parent;env=os.environ.copy();env.update(PYTHONPATH=str(P)+os.pathsep+str(root/'source/src'),ENDING_ROLE='above',LADDER_SHIFT='.06',HANDOFF_ARM='official',TRIGGER_MODE='supported_root',SWITCH_MARGIN='.04',SWITCH_MAX_SPIN='999',GETUP_FILE='getup.onnx',RECOVERY_MIN_ROOT_Z='.74')
mode=sys.argv[1];args=sys.argv[2:]
if mode=='climber':script='train_climber.py';defaults=['--source',str(P/'models/climber.pt'),'--floor-probability','.9','--desk-probability','0','--mode','train']
elif mode=='getup':script='train_getup.py';defaults=['--checkpoint',str(P/'models/getup.pt'),'--cost','0','--mode','train']
elif mode=='evaluate':script='evaluate_sequence.py';defaults=['--source',str(P/'models/climber.pt'),'--mode','eval','--floor-probability','0','--desk-probability','0','--transition-audit']
else:raise SystemExit('Choose climber, getup, or evaluate')
subprocess.run([sys.executable,str(P/script),*defaults,*args],env=env,cwd=root/'source',check=True)
