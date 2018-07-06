from .fcn32s import fcn32s
from .fcn32s_lite import fcn32s_lite
from .fgbg import fgbg
from .dios_early import dios_early
from .dios_early_lite import dios_early_lite
from .dios_late import dios_late
from .dios_late_glob import dios_late_glob
from .cofeat_early import cofeat_early
from .cofeat_late import cofeat_late


models = {
    'fcn32s': fcn32s,
    'fcn32s-lite': fcn32s_lite,
    'fgbg': fgbg,
    'dios-early': dios_early,
    'dios-early-lite': dios_early_lite,
    'dios-late': dios_late,
    'dios-late-glob': dios_late_glob,
    'cofeat-early': cofeat_early,
    'cofeat-late': cofeat_late,
}

def prepare_model(model_name, num_classes):
    model = models[model_name](num_classes)
    return model
