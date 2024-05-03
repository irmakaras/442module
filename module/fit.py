from sympy import sympify, lambdify
from sympy.abc import x, y, z, t
import numpy as np
from module.utils import green, blue, bold, orange, red
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2
from scipy.special import factorial, xlogy, gammaln
from inspect import getfullargspec

VARIABLES = [x, y, z, t]
REPLACEMENTS = {'factorial': factorial,
                "xlogy": xlogy,
                "gammaln": gammaln}

SPECIALS = {"gaussian": {"f": "A*exp((x-μ)**2/(-2*σ**2))",
                         "f_exp": "A*exp((x-μ)**2/(-2*σ**2))"},
            "poisson": {"f": "A*exp(xlogy(x, λ) - gammaln(x + 1) - λ)",
                        "f_exp": "A*exp(-λ)*λ**x/factorial(x)"}}

def func_from_func(function):
    V = [str(x) for x in VARIABLES]
    args = getfullargspec(function)[0]
    vars = [x for x in args if x in V]
    params = [x for x in args if x not in V]
    return function, vars, params


def func_from_string(function):
    if function in SPECIALS.keys():
        _fstr = SPECIALS[function]["f"]
    else:
        _fstr = function
    func_expression = sympify(_fstr)
    free = sorted(func_expression.free_symbols, key=lambda x: x.sort_key())
    vars = [x for x in free if x in VARIABLES]
    params = [x for x in free if x not in VARIABLES]

    func = lambdify(vars+params, func_expression, ["numpy", REPLACEMENTS])
    vars = [str(x) for x in vars]
    params = [str(x) for x in params]
    if function in SPECIALS.keys():
        func_expression = sympify(SPECIALS[function]["f_exp"])
    return func, vars, params, func_expression

### --------------- FUNCTION CLASSES ---------------

class FitFunction1D:
    """
    A class that generates a callable python function from a string of a mathematical function.

    Properties:
    ----------
    ``f_exp``: SymPy expression of the function.
    ``f_param``: SymPy expression of the function with the current set parameters.
    ``f_call``: callable expression of the function with the current set parameters.
    ``fit_results``: Dictionary containing the results of the fit.

    Methods:
    ----------
    ``fit``: Fits the function to a given dataset. Stores the results in ``fit_results`` property and prints them.
    ``evaluate`` or ``__call__``: Calls the function with the given values. ``f.evaluate(values)`` and ``f(values)`` are the same.
    """
    
    def __init__(self, function):
        """
        Initializes the instanced based on a string.

        Arguments:
        ----------
        ``function``: str or callable
            String of the function. For exponentiation, use ``x**2`` rather than ``x^2``. Acceptable strings are:

            - Any parsable string of a mathemathical expression.
            - ``"gaussian"`` creates a Gaussian pdf with 3 parameters (A, μ, σ).
            - ``"poisson"`` creates a Poisson pdf with 2 parameters (A, λ).

        ``initial_parameters``: list (default=``None``)
            List of initial values for the parameters of the function. If ``None`` all parameters will be initialized to 1.
        
        Example:
        ----------
            ``FitFunction1D("a*x + b")`` will create a callable function ``a*x + b`` with arguments ``(x, a, b)``
        """
        if isinstance(function, str):
            self.func, self.vars, self.params, self.func_expression = func_from_string(function)
        else:
            self.func, self.vars, self.params = func_from_func(function)
            self.func_expression = "Custom Function"

        if len(self.vars) != 1:
            raise ValueError(f"Expected 1D function but got {len(self.vars)}D function! Variables: {self.vars}, Parameters: {self.params}, Acceptable Variables: {VARIABLES}.")


    def __call__(self, x, params):
        if isinstance(params, list):
            y = self.func(x, *params)
        elif isinstance(params, dict):
            y = self.func(x, **params)
        return y
        

    def fit(self, x, y=None, y_err=None, initial_values=None, range=None, limits=None, histogram={}):
        """
        Fits the function to a given dataset. Stores the results in ``fit_results`` property and prints them. If only ``x`` values are given, a histogram fit is done. If ``x``, ``y`` and, ``y_err`` values are given, a regression fit is done.

        Arguments:
        ----------
        ``x``: array
                x values of the data. If only ``x`` is given, a histogram fit is done.
        ``y``: array
                y values of the data.
        ``y_err``: array
                Error of the y values of the data.
        ``initial_values``: array or dict (default=``None``)
                The initial guess for each parameter. If ``None`` it sets the initial guess for each parameter to 1.
        ``range``: [min, max] (default=``None``)
                Interval on the x axis that the fit will be applied to
        ``bins``: int (default=``None``)
                Only used for histogram fitting. Number of bins that the histogram will be generated on.
        ``limits``: array or dict (default=``None``)
                Limits of the fitted parameters. ``(0, None)`` sets a parameter to be greater than zero.
        """
        self.fit_results = {}
        print(blue(bold(f"Fitting: {self.func_expression}")))

        # Initial value parsing
        if initial_values is None:
            _init_vals = [1] * len(self.params)
            print(orange(bold(f"No initial value were given, defaulting to {_init_vals}. This may cause a bad fit!")))
        else:
            if isinstance(initial_values, list):
                if len(initial_values) != len(self.params):
                    raise Exception(red(f"Expected {len(self.params)} parameters for initial guess ({self.params}) but got {len(initial_values)} parameters! Initial guess: {initial_values}."))
                _init_vals = initial_values
            elif isinstance(initial_values, dict):
                _init_vals = [1] * len(self.params)
                if len(initial_values) != len(self.params):
                    print(orange(bold(f"Not all initial values were given. This may cause a bad fit!")))
                for param, value in initial_values.items():
                    if param not in self.params:
                        raise Exception(red(f"Parameter '{param}' not found in parameter list {self.params}"))
                    _init_vals[self.params.index(param)] = value
        _pv_pairs = {p: v for p, v in zip(self.params, _init_vals)}

        # Data parsing
        if y is not None:
            # Set _x, _y, _yerr
            _x = x
            _y = y
            if y_err is None:
                raise Exception(red(f"y_err must be given for fitting!"))
            elif isinstance(y_err, float) or isinstance(y_err, int):
                _y_err = y_err * np.ones(x.shape)
            else:
                _y_err = y_err
            print(green(f"Found x, y and y_err values. Fitting!"))
        else:
            # If no y is given, create a histogram and set _x, _y, _yerr
            if len(histogram) == 0:
                raise Exception(f"Either y or histogram parameters must be given for fitting!")
            else:
                if isinstance(histogram, dict):
                    bins = histogram["bins"]
                    try:
                        min, max = histogram["min"], histogram["max"]
                    except:
                        print(orange(f"Histogram min-max are not given, defaulting to min-max values of the dataset."))
                        min, max = np.min(x), np.max(x)
                elif isinstance(histogram, list):
                    bins = histogram[0]
                    try:
                        min, max = histogram[1:2]
                    except:
                        print(orange(f"Histogram min-max are not given, defaulting to min-max values of the dataset."))
                        min, max = np.min(x), np.max(x)
                else:
                    raise Exception(f"Histogram options should either be a list or a dictionary!")
                _y, _x = np.histogram(x, bins, [min, max])
                _y_err = np.sqrt(_y)
                _x = _x[:-1] + (_x[1] - _x[0])/2 # Take midpoints of each edge for y values

        if range is not None:
            self.fit_range=range
        else:
            self.fit_range=[np.min(_x), np.max(_x)]

        # Cut the data in the range and remove _y_err = 0 points because they will throw a "zero-division" error
        mask = np.where((_x >= self.fit_range[0]) & (_x <= self.fit_range[1]) & (_y_err != 0))
        _x = _x[mask]
        _y = _y[mask]
        _y_err = _y_err[mask]

        # Fit!
        least_squares = LeastSquares(_x, _y, _y_err, self.func)
        m = Minuit(least_squares, **_pv_pairs, name=self.params)
        self.m = m
        
        print(f"Initial values are:")
        print(_pv_pairs)
        
        if limits is not None:
            if isinstance(limits, list):
                m.limits = limits
            elif isinstance(limits, dict):
                for param, limit in limits.items():
                    m.limits[param] = limit

        m.migrad()
        m.hesse()
        self.values = m.values.to_dict()

        self.fit_results["values"] = []
        self.fit_results["errors"] = []
        self.fit_results["chi2"] = m.fval
        self.fit_results["ndof"] = m.ndof
        self.fit_results["p"] = chi2.sf(m.fval, m.ndof)

        for row in m.params.to_table()[0]:
            self.fit_results["values"].append(row[2])
            self.fit_results["errors"].append(row[3])
        print(green(bold(f"Results:")))
        print(m.params)

        self.x_fit = np.linspace(self.fit_range[0], self.fit_range[1], 200)
        self.y_fit = self(self.x_fit, self.values)



    def set_expression(self, str, type="sympy"):
        match type:
            case "sympy":
                _exp = sympify(str)
            case _:
                _exp = str
        self.func_expression = _exp

