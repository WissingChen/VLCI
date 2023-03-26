"""
model list
"""
from .baseline import Baseline
from .vlci import VLCI

model_fn = {'baseline': Baseline, 'vlci': VLCI}
