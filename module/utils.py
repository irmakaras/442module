### --------------- BEAUTIFUL PRINTING --------------- Shamelessly stolen from pyRAT

# Shamelessly stolen from blender
class bcolors:
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER    = '\033[0;95m'
    WARNING   = '\033[0;93m'
    FAIL      = '\033[0;91m'
    OKRED     = '\033[0;91m'
    OKGREEN   = '\033[0;92m'
    OKYELLOW  = '\033[0;93m'
    OKBLUE    = '\033[0;94m'
    OKMAGENTA = '\033[0;95m'
    OKCYAN    = '\033[0;96m'
    ORANGE    = '\033[38;5;208m'

def modify_printed_string(att, *args, sep=' '):
    text = sep.join(args)
    return f"{att}{text}{bcolors.ENDC}"

def red(*args, sep=' '):
    return modify_printed_string(bcolors.OKRED, *args, sep=sep)

def green(*args, sep=' '):
    return modify_printed_string(bcolors.OKGREEN, *args, sep=sep)

def yellow(*args, sep=' '):
    return modify_printed_string(bcolors.OKYELLOW, *args, sep=sep)

def blue(*args, sep=' '):
    return modify_printed_string(bcolors.OKBLUE, *args, sep=sep)

def magenta(*args, sep=' '):
    return modify_printed_string(bcolors.OKMAGENTA, *args, sep=sep)

def cyan(*args, sep=' '):
    return modify_printed_string(bcolors.OKCYAN, *args, sep=sep)

def orange(*args, sep=' '):
    return modify_printed_string(bcolors.ORANGE, *args, sep=sep)

def bold(*args, sep=' '):
    return modify_printed_string(bcolors.BOLD, *args, sep=sep)

def warning(*args, sep=' '):
    return modify_printed_string(bcolors.WARNING, *args, sep=sep)

### -----

style = {
    "mathtext.default": "regular",
    "figure.figsize": (10.0, 10.0),
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.unicode_minus": False,
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.handlelength": 1.5,
    "legend.borderpad": 0.5,
    "legend.frameon": False,
    "xtick.direction": "in",
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "xtick.major.pad": 6,
    "xtick.top": True,
    "xtick.major.top": True,
    "xtick.major.bottom": True,
    "xtick.minor.top": True,
    "xtick.minor.bottom": True,
    "xtick.minor.visible": True,
    "ytick.direction": "in",
    "ytick.major.size": 12,
    "ytick.minor.size": 6.0,
    "ytick.right": True,
    "ytick.major.left": True,
    "ytick.major.right": True,
    "ytick.minor.left": True,
    "ytick.minor.right": True,
    "ytick.minor.visible": True,
    "grid.alpha": 0.8,
    "grid.linestyle": ":",
    "axes.linewidth": 2,
    "savefig.transparent": False,
    "xaxis.labellocation": "center",
    "yaxis.labellocation": "center",
}