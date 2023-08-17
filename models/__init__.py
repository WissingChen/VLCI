"""
model list
"""
from .baseline import Baseline
from .vlci import VLCI
from .vlp import VLP

model_fn = {'baseline': Baseline, 'vlci': VLCI, 'vlp': VLP}
