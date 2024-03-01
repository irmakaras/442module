from sympy import sympify, lambdify
from sympy.abc import x, y, z, t
import numpy as np
from module.utils import green, blue, bold, orange
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2

VARIABLES = [x, y, z, t]

def lambdify_string(fstr):
    f_exp = sympify(fstr)
    free = sorted(f_exp.free_symbols, key=lambda x: x.sort_key())
    vars = [x for x in free if x in VARIABLES]
    params = [x for x in free if x not in VARIABLES]
    return f_exp, vars, params

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
    
    def __init__(self, function_string, initial_parameters=None):
        """
        Initializes the instanced based on a string.

        Arguments:
        ----------
        ``function_string``: str
            String of the function. For exponentiation, use ``x**2`` rather than ``x^2``. Acceptable strings are:

            - Any parsable string of a mathemathical expression.
            - ``"gaussian"`` creates a gaussian function with 3 parameters.

        ``initial_parameters``: list (default=``None``)
            List of initial values for the parameters of the function. If ``None`` all parameters will be initialized to 1.
        
        Example:
        ----------
            ``FitFunction1D("a*x + b", [1, 0.5])`` will create a callable function ``x + 0.5``
        """
        match function_string:
            case "gaussian":
                _fstr = "A*exp((x-μ)**2/(-2*σ**2))"
            case _:
                _fstr = function_string
        self.f_exp, self.vars, self.params = lambdify_string(_fstr)
        if len(self.vars) != 1:
            raise ValueError(f"Expected 1D function but got {len(self.vars)}D function! Variables: {self.vars}, Parameters: {self.params}, Acceptable Variables: {VARIABLES}.")
        if initial_parameters is not None:
            _init_param = initial_parameters
        else:
            _init_param = [1 for x in self.params]
        #print(f"Setting initial parameters for {self.f_exp} as {_init_param}")
        self.set_params(_init_param)


    def set_params(self, _params):
        self.param_values = _params
        self.f_param = self.f_exp.subs(dict(zip(self.params, self.param_values)))
        self.f_call = lambdify(self.vars, self.f_param, "numpy")


    def evaluate(self, var, params=None):
        if params is not None:
            self.set_params(params)
        return self.f_call(var)
        

    def fit(self, x, y=None, y_err=None, initial_params=None, range=None, bins=None):
        """
        Fits the function to a given dataset. Stores the results in ``fit_results`` property and prints them. If only ``x`` values are given, a histogram fit is done. If ``x``, ``y`` and, ``y_err`` values are given, a regression fit is done.

        Arguments:
        ----------
        ``x``: list
                x values of the data. If only ``x`` is given, a histogram fit is done.
        ``y``: list (default=``None``)
                y values of the data.
        ``y_err``: list (default=``None``)
                Error of the y values of the data.
        ``initial_params``: list (default=``None``)
                List of initial guesses for each parameter. If ``None`` it sets the initial guess for each parameter to 1.
        ``range``: [min, max] (default=``None``)
                Interval on the x axis that the fit will be applied to
        ``bins``: int (default=``None``)
                Only used for histogram fitting. Number of bins that the histogram will be generated on.

        """
        self.fit_results = {}
        print(blue(bold(f"Fitting: {self.f_exp}")))
        if initial_params is None:
            _init_param = [1] * len(self.params)
            print(orange(bold(f"No initial guess given, defaulting to {_init_param}. This may cause a bad fit!")))
        else:
            if len(initial_params) != len(self.params):
                raise ValueError(f"Expected {len(self.params)} parameters for initial guess ({self.params}) but got {len(initial_params)} parameters! Initial guess: {initial_params}.")
            _init_param = initial_params
        
        if y is not None:
            if y_err is None:
                raise TypeError(f"y_err must be given for regression!")
            elif isinstance(y_err, float) or isinstance(y_err, int):
                _y_err = y_err * np.ones(x.shape)
            else:
                _y_err = y_err
            
            print(green(f"Found x, y and y_err values. Fitting for regression."))
            self.fit_reg(x, y, _y_err, _init_param, range)
        else:
            print(green(f"Found only x values. Fitting for histogram."))
            self.fit_hist(x, bins, _init_param, range)
        self.x_fit = np.linspace(self.fit_range[0], self.fit_range[1], 200)
        self.y_fit = self(self.x_fit)
            

    def fit_reg(self, x_data, y_data, y_err, initial_params, range):
        if range is not None:
            self.fit_range=range
        else:
            self.fit_range=[np.min(x_data), np.max(x_data)]

        # Cut the data in the range
        mask = np.where((x_data >= self.fit_range[0]) & (x_data <= self.fit_range[1]))
        self.x_data = x_data[mask]
        self.y_data = y_data[mask]
        self.y_err = y_err[mask]
        self.curve_fit(self.x_data, self.y_data, self.y_err, initial_params)


    def fit_hist(self, x_data, bins, initial_params, range):
        h_data, x_edges = np.histogram(x_data, bins, range=range)
        h_err = np.sqrt(h_data)

        if type(range) != type(None):
            self.fit_range=range
        else:
            self.fit_range=[np.min(x_edges), np.max(x_edges)]

        # Take the midpoints for fitting
        step = (x_edges[1]-x_edges[0])/2
        x_edges += step

        # Don't put zeros into the fit because their errors cause zero-division errors.
        mask = np.where(h_data != 0)
        self.x_data = x_edges[:-1][mask]
        self.y_data = h_data[mask]
        self.y_err = h_err[mask]

        self.curve_fit(self.x_data, self.y_data, self.y_err, initial_params)


    def curve_fit(self, x_data, y_data, y_err, initial_params):
        least_squares = LeastSquares(x_data, y_data, y_err, self.evaluate)
        m = Minuit(least_squares, initial_params, name=[str(p) for p in self.params])
        m.migrad()
        m.hesse()
        self.m = m
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

    def __call__(self, var, params=None):
        return self.evaluate(var, params)
