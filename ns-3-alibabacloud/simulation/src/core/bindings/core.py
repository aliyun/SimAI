from __future__ import absolute_import
# "from _core import *" doesn't work here because symbols starting
# with underscore would not be imported.  But they are needed because
# other modules depend on them...
from . import _core
g = globals()
for k,v in _core.__dict__.items():
    g[k] = v
del g, k, v, _core


# Without this, Python programs often crash because Node's are kept
# alive after the Python interpreter is finalized, which leads to
# crashes because some destructors call Python API.
import atexit
atexit.register(Simulator.Destroy)
del atexit
