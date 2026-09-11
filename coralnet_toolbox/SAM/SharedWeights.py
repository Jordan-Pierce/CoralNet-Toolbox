"""One copy of each SAM model in memory, shared by the SAM dialogs.

The Predictor dialog (prompts) and the Generator dialog (segment everything)
each build an ultralytics predictor. Loading the same weights in both used to
build two modules, so two copies of the weights sat in VRAM. A SAM predictor
takes a built module through setup_model(model=...), so whichever dialog builds
second borrows the module the first one built.

Modules are held weakly: once neither dialog holds one it is freed, as it was
before sharing.

A shared module keeps one setting that each predictor changes: its input size
(setup_source calls set_imgsz for every image). Code that reuses features
encoded earlier must put the size back first; see
DeployPredictorDialog._sync_model_imgsz.
"""

import weakref


_modules = weakref.WeakValueDictionary()


def _key(weights, device, quantize):
    # Device and precision are part of the key because setup_model moves and
    # casts the module in place: borrowing it at another device or precision
    # would move or cast it under the dialog that built it.
    return str(weights), str(device).strip().lower(), int(quantize)


def get(weights, device, quantize):
    """Return the module already built for these weights, device and precision, or None."""
    return _modules.get(_key(weights, device, quantize))


def register(weights, device, quantize, module):
    """Offer a built module to the other dialog."""
    if module is not None:
        _modules[_key(weights, device, quantize)] = module
