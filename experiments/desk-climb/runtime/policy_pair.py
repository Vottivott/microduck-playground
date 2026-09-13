"""CPU ONNX action contract. The caller supplies a simulator eligibility flag.
This module does not implement a deployable desk detector.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort
class PolicyPair:
 def __init__(self,models):
  p=Path(models);options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1;self.climb=ort.InferenceSession(str(p/'climber.onnx'),providers=['CPUExecutionProvider'],sess_options=options);self.getup=ort.InferenceSession(str(p/'getup.onnx'),providers=['CPUExecutionProvider'],sess_options=options);meta=self.getup.get_modelmeta().custom_metadata_map
  self.home=np.fromstring(meta['default_joint_pos'],sep=',').astype(np.float32);names=meta['joint_names'].split(',');self.alpha=np.array([.5 if n.startswith(('neck','head')) else .7 for n in names],dtype=np.float32);self.reset()
 def reset(self):
  self.recovering=False;self.raw=np.zeros(14,np.float32);self.executed=np.zeros(14,np.float32)
 def step(self,observation,eligible=False):
  x=np.asarray(observation,dtype=np.float32).reshape(1,61).copy();assert np.isfinite(x).all() and not x[:,48:].any();x[:,34:48]=self.raw
  self.recovering|=bool(eligible);session=self.getup if self.recovering else self.climb
  self.raw=session.run(None,{session.get_inputs()[0].name:x})[0].reshape(14);assert np.isfinite(self.raw).all()
  self.executed=self.alpha*self.raw+(1-self.alpha)*self.executed if self.recovering else self.raw.copy()
  return {'raw_action':self.raw.copy(),'executed_offset':self.executed.copy(),'target_radians':self.home+self.executed,'kp_ratio':.8 if self.recovering else 1.}
