import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import VPacker, HPacker, TextArea, AnchoredOffsetbox
import numpy as np
from module.fit import FitFunction1D, VARIABLES
from sympy import latex


### --------------- PLOTTING FUNCTIONS ---------------

def plot_data(x, y=None, x_err=None, y_err=None, **kwargs):
    """
    Plots the data on the current matplotlib axis. If only ``x`` values are given it produces a histogram plot. If ``x`` and ``y`` values are given it produces a scatter plot.

    Arguments:
    ----------
    ``x``: list
        x values of the data. If only ``x`` is given it produces a histogram plot
    ``y``: list (default=``None``)
        y values of the data.
    ``x_err``: list (default=``None``)
        Error of the x values of the data.
    ``y_err``: list (default=``None``)
        Error of the y values of the data. For histograms, it's calculated as sqrt(y) for each bin during plotting.
    Other Arguments:
    ----------
    ``range``: [min, max] (default=``None``)
        Interval on the x axis that the fit will be applied to
    ``bins``: int (default=``None``)
        Only used for histogram fitting. Number of bins that the histogram will be generated on.
    ``errorbar_style``: dict
        Styling options for the errorbars.
    ``histogram_style``: dict
        Styling options for the histogram.
    """
    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
    else:
        ax = plt.gca()

    if "errorbar_style" in kwargs:
        errorbar_style = kwargs.pop("errorbar_style")
    else:
        errorbar_style = dict(marker= "",
                    capsize=    3,
                    color=      "black",
                    linestyle=  "",
                    label=      "Error",
                    elinewidth= 2)

    if y is not None:
        ax.errorbar(x, y, xerr=x_err, yerr=y_err, zorder=0, **errorbar_style)
    else:
        if "bins" in kwargs:
            bins = kwargs.pop("bins")
        if "histogram_style" in kwargs:
            histogram_style = kwargs.pop("histogram_style")
        hist, edges = np.histogram(x, bins)
        ax.stairs(hist, edges, **histogram_style)
        ax.errorbar(edges[:-1] + (edges[1]-edges[0])/2, hist, yerr=np.sqrt(hist), zorder=1, **errorbar_style)


def plot_fit(functions, **kwargs):
    if isinstance(functions, FitFunction1D):
        _fs = [functions]
    else:
        _fs = functions

    if "ax" in kwargs:
        ax = kwargs.pop("ax")
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
    else:
        ax = plt.gca()
    
    if "fit_style" in kwargs:
        fit_style = kwargs.pop("fit_style")
        if isinstance(fit_style, dict):
            fit_style = [fit_style]
        if len(fit_style) != len(_fs) and len(fit_style) != 1:
            raise ValueError(f"fit_style should either be the same length as the functions list or a single style for all functions! Expected {len(_fs)} but given {len(fit_style)}.")
        
    for f in _fs:
        if not isinstance(f, FitFunction1D):
            raise ValueError(f"functions must contain only FitFunction1D objects. Given: {type(f)} at index {_fs.index(f)}")
        if len(fit_style) == 1:
            ax.plot(f.x_fit, f.y_fit, zorder=2, **fit_style[0])
        else:
            idx = _fs.index(f)
            ax.plot(f.x_fit, f.y_fit, zorder=2, **fit_style[idx])


def draw_fit_info(functions, loc="upper right", offset=(0,0), fontsize=None, linesep=0, horizontal=False, edges=True, boxstyle=None, ax=None):
    # Make sure we can get the axis properly
    if ax is None:
        ax = plt.gca()
    else:
        if not isinstance(ax, plt.Axes):
            raise ValueError("ax must be a matplotlib Axes object")
    fig = ax.get_figure()

    # Make sure we can iterate over everything
    if isinstance(functions, FitFunction1D):
        _fs = [functions]
    else:
        _fs = functions

    # Fallback to default fontsize
    if fontsize is None:
        fontsize = mpl.rcParams["font.size"]


    fits = []
    longest = 0
    for f in _fs:
        if not isinstance(f, FitFunction1D):
            raise ValueError(f"functions must contain only FitFunction1D objects. Given: {type(f)} at index {_fs.index(f)}")
        head = []
        tail = []

        text_properties_head = dict(fontsize=fontsize, horizontalalignment="right", verticalalignment="baseline", fontweight= "bold")
        text_properties_tail = dict(fontsize=fontsize, horizontalalignment="left", verticalalignment="baseline")

        title = TextArea(f"Fit {_fs.index(f)+1 if len(_fs) > 1 else ''}", textprops=dict(fontsize=fontsize, horizontalalignment="left", fontweight="bold"))

        # Expression
        head.append(TextArea(f"f({(f.vars[0])})", textprops=text_properties_tail))
        tail.append(TextArea(f"$={latex(f.f_exp)}$", textprops=text_properties_tail))

        # Chi2
        head.append(TextArea("$\\chi^2$/$n_\\mathrm{{dof}}$", textprops=text_properties_tail))
        tail.append(TextArea(f"$={f.fit_results['chi2']/f.fit_results['ndof']:.3f}$", textprops=text_properties_tail))
        
        # Probability
        head.append(TextArea("Prob", textprops=text_properties_tail))
        tail.append(TextArea(f"$={f.fit_results['p']:.3f}$", textprops=text_properties_tail))

        # Parameters
        for p, v, e in zip(f.params, f.fit_results["values"], f.fit_results["errors"]):
            head.append(TextArea(f"{p}", textprops=text_properties_head))
            if "e" in v:
                _v = v.replace("e", "\\cdot 10^{")+"}"
                _e = e.replace("e", "\\cdot 10^{")+"}"
                extra = 19
            else:
                _v = v
                _e = e
                extra = 0
            ve = f"$={_v} \\pm {_e}$"
            tail.append(TextArea(ve, textprops=text_properties_tail))
            #tail.append(TextArea(f"={v:.5e} $\\pm$ {e:.5e}", textprops=text_properties_tail))
            if len(ve) - extra > longest:
                longest = len(ve) - extra

        _body = []
        for h, t in zip(head,tail):
            _body.append(HPacker(width=fontsize*10, align="left", mode="equal", children=[h,t]))

        body = VPacker(sep=linesep, align="center", children=_body)
        #fit_info = VPacker(align='left', children=[body])
        fits.append(body)


    if horizontal:
        fit_pack = HPacker(width=fontsize*longest*len(_fs), sep=fontsize, align="top", mode="equal", children=fits)
    else:
        fit_pack = VPacker(width=fontsize*longest, sep=fontsize, align="left", children=fits)

    fit_box = AnchoredOffsetbox(loc=2, child=fit_pack, pad=0.5, frameon=edges, 
                             borderpad=0.0, bbox_to_anchor=(1.0, 1.0), 
                             bbox_transform=ax.transAxes, zorder=10)
    
    if boxstyle is None:
        boxstyle="square,pad=0"
    fit_box.patch.set_boxstyle(boxstyle)
    fit_box.set_clip_on(False)
    fit_box.patch.set(lw=mpl.rcParams["axes.linewidth"])

    ax.add_artist(fit_box)
    fig.draw_artist(fit_box)

    fit_bbox = fit_box.get_window_extent()
    ax_bbox = ax.get_window_extent()
    w = fit_bbox.width/ax_bbox.width
    h = fit_bbox.height/ax_bbox.height
    if isinstance(loc, tuple) or isinstance(loc, list):
        _locx = loc[0]
        _locy = loc[1]
    elif loc == "upper right":
        _locx = (1-w) + offset[0]
        _locy = 1 + offset[1]
    elif loc == "upper left":
        _locx = 0 + offset[0]
        _locy = 1 + offset[1]
    elif loc == "lower left":
        _locx = 0 + offset[0]
        _locy = h + offset[1]
    elif loc == "lower right":
        _locx = (1-w) + offset[0]
        _locy = h + offset[1]
    else:
        raise ValueError(f"loc can either be a tuple (x,y) or a valid string. Given: {loc}")

    fit_box.set_bbox_to_anchor((_locx, _locy), ax.transAxes)
    
    fig.canvas.blit(fig.bbox)


### --------------- HELPER FUNCTIONS ---------------

def num_to_exponent(val, err):
    if abs(err) <= 1e-4 or abs(err) >= 1e4:
        e = f"{err:.3e}"+"}"
    else:
        e = f"{err:.4f}"

    if abs(val) >= 1e3 or abs(val) <= 1e-3:
        v = f"{val:.3e}"+"}"
    else:
        v = f"{val:.3f}"


    s = f"{v} \\pm {e}"
    s = s.replace("e", "\\cdot 10^{")
    return s

def relative_exponent(val, err):
    oom = np.floor(np.log10(abs(err)))
    print(oom)
    v = f"{val/(10**oom):.3f}"
    e = f"{err/(10**oom):.3f}"
    s = f"${v} \\pm {e} \\cdot 10^{{oom}}$"
