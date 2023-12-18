import gudhi as gd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from tqdm import tqdm


def compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an input standardized stock price dataframe, transforms
    the dataframe in its relative-valued counterpart:
    - prices are transformed into their log-return
    - volume is transformed into a percent change
    """
    # Copies the input dataframe and updates the column names
    new_df = df.copy()
    # Computes the log-returns for each price column
    price_columns = ["open", "high", "low", "close", "adj_close"]
    for column in tqdm(price_columns):
        log_price            = np.log(new_df[column])
        log_price_shifted    = np.log(new_df[column].shift(1))
        new_df[column+"_lr"] = log_price - log_price_shifted
    # Computes the volume percent change
    new_df["volume_pct_change"]=new_df["volume"].pct_change()
    return new_df


def compute_norm_difference_persistence_landscapes(data: pd.DataFrame,
                                                   landscapes: list) -> pd.DataFrame:
    """
    Computes the norm of difference between landscapes.
    """
    # Retrieves the dates corresponding to each landscape +1
    index = data.index[len(data)-len(landscapes)+1:]
    # Computes the norms of the differences
    norm_diffs = lambda ls, i: np.linalg.norm(ls[i]-ls[i-1])
    norm_of_differences = [norm_diffs(landscapes, i)
                        for i in range(1, len(landscapes))]
    # Computes the output
    df = pd.DataFrame(norm_of_differences, index=index, columns=["norm"])
    ax = df.plot(figsize = (18, 7),
                lw      = 0.8,
                color   = "green",
                alpha   = 0.5,
                ylabel  = "Norm value",
                title   = "Norm of the difference between consecutive landscapes")
    ax.axvline(x         = np.where([df.index=="2000-01-10"])[1][0],
            color     = 'r',
            linestyle = (0, (3, 5, 1, 5, 1, 5)),
            label     = 'America Online/Time Warner merger')
    ax.axvline(x         = np.where([df.index=="2008-09-15"])[1][0],
            color     = 'r',
            linestyle = '--',
            label     = 'Lehman Brothers bankruptcy')
    ax.legend()
    return df


def compute_persistence_diagram(point_cloud: np.ndarray,
                                rips_complex: bool=True,
                                print_graph: bool=False,
                                memory_saving: tuple=(False, 1)) -> np.ndarray:
    """
    Given an input point cloud data set, computes the corresponding
    persistence diagram (only for 1-d loops as in the paper).
    the method relies on using alpha filtration
    """
    # Computes the Vietoris-Rips complex, its barcode and 1-loop diagram
    if rips_complex:
        simplex   = gd.RipsComplex(points = point_cloud)
        simplex   = simplex.create_simplex_tree(max_dimension = 2)
        bar_codes = simplex.persistence()
        if memory_saving[0]:
            simplex = simplex.persistence_intervals_in_dimension(memory_saving[1])
    # Computes the alpha complex, its varcode and 1-loop diagram
    else:
        simplex   = gd.AlphaComplex(points = point_cloud)
        simplex   = simplex.create_simplex_tree()
        bar_codes = [x for x in simplex.persistence() if x[0]<=1]
        if memory_saving[0]:
            simplex = simplex.persistence_intervals_in_dimension(memory_saving[1])
    # prints the persistence diagram graph if requested
    if print_graph: gd.plot_persistence_diagram(bar_codes)
    # the returned diagram comprises the birth and death of 1-loops
    return simplex


def compute_persistence_diagrams(data: pd.DataFrame,
                                 w: int,
                                 rips_complex: bool=True,
                                 memory_saving: tuple=(False, 1)) -> np.ndarray:
    """
    Given an input time series, computes the corresponding
    persistence diagram given a shifting window of size w.
    """
    data = data.values
    diagrams = []
    for slc in tqdm(range(data.shape[0]-w), desc='Diagram Progress'):
        point_cloud = data[slc:slc+w]
        diagram     = compute_persistence_diagram(point_cloud,
                                                rips_complex,
                                                False,
                                                memory_saving)
        diagrams.append(diagram)
    return diagrams


def compute_persistence_landscape(diagram: np.ndarray,         # diagram range
                                  endpoints: list,               # endpoints
                                  homology_dimension: int=1,    # k dimensions
                                  n_landscapes: int=5,    # m landscapes
                                  resolution: int=1000, # n nodes
                                  memory_saving: bool=False,
                                  pbar=None) -> np.ndarray:
    """
    Given a persistence diagram of 1D loops of a given
    time series, computes the corresponding persistence landscape.
    Inspired from: https://github.com/MathieuCarriere/sklearn-tda/
                blob/master/sklearn_tda/vector_methods.py
    """
    # If the diagram is empty, return an empty landscape
    if endpoints[0] == endpoints[1] == 0:
        return np.zeros((n_landscapes, resolution))
    # Renames the min-max range of the given diagram
    diagram_range = endpoints
    # Extracts the homology class from the diagram in case the
    # computation mode  is not memory-saving. I.e. the dimension
    # class was not pre-fetched at the diagram computation level
    if not memory_saving:
        diagram = diagram.persistence_intervals_in_dimension(homology_dimension)
    # Initializes important variables
    x_range        =  np.linspace(diagram_range[0],
                                diagram_range[1],
                                resolution)
    step           = x_range[1] - x_range[0]
    length_diagram = len(diagram)
    computed_landscapes_at_given_resolution = \
        np.zeros([n_landscapes, resolution])
    computed_y_values = [[] for _ in range(resolution)]
    # Initializes important anonymous functions
    compute_x_subrange = lambda x: int(np.ceil(x/step))
    # Computes the persistence landscape coverage, here
    # the x- and y-axes ranges
    for x, y in diagram:
        # Populates thex-axis range as defined for each
        # persistence diagram point
        min_point = x - diagram_range[0]
        mid_point = 0.5*(x+y) - diagram_range[0]
        max_point = y - diagram_range[0]
        minimum_x = compute_x_subrange(min_point)
        middle_x  = compute_x_subrange(mid_point)
        maximum_x = compute_x_subrange(max_point)
        # Populates the y-axis values given the computed
        # x-axis range for that part of the resulting landscape
        if minimum_x<resolution and maximum_x>0:
            y_value = diagram_range[0] + minimum_x * step - x
            for z in range(minimum_x, middle_x):
                computed_y_values[z].append(y_value) ##################################################################################################
                y_value += step
            y_value = y - diagram_range[0] - middle_x * step
            for z in range(middle_x, maximum_x):
                computed_y_values[z].append(y_value)
                y_value -= step
    # Computes for each resolution the corresponding landscape
    for i in range(resolution):
        computed_y_values[i].sort(reverse=True)
        max_range = min(n_landscapes, len(computed_y_values[i]))
        for j in range(max_range):
            computed_landscapes_at_given_resolution[j,i] = \
                computed_y_values[i][j]

    if pbar is not None:
        pbar.update(1)

    return computed_landscapes_at_given_resolution


def compute_persistence_landscapes(diagrams: np.ndarray,         # diagram D
                                   homology_dimension: int=1,    # k dimensions
                                   n_landscapes: int=5,    # m landscapes
                                   resolution: int=1000, # n nodes
                                   memory_saving: bool=False) -> np.ndarray:
    """
    Given a list of persistence diagrams of 1D loops of a given
    time series, computes the corresponding persistence landscapes.
    """
    print("Call" + str())
    k    = homology_dimension
    # Declares the anonymous functions helping to compute the
    # diagram endpoints different depending on the memory saving mode
    if memory_saving:
        minp = lambda d: np.min(d) if len(d)>0 else 0
        maxp = lambda d: np.max(d) if len(d)>0 else 0
    else:
        def compute_endpoint(d, minmax):
            d = d.persistence_intervals_in_dimension(k)
            if len(d)>0 and minmax=="min":    return np.min(d)
            elif len(d) >0 and minmax=="max": return np.max(d)
            else:                             return 0
        minp = lambda d: compute_endpoint(d, "min")
        maxp = lambda d: compute_endpoint(d, "max")
    # Transforms all diagrams into landscapes
    landscapes = [
        compute_persistence_landscape(
            diag,                     # diagram D
            [minp(diag), maxp(diag)], # endpoints
            homology_dimension,       # k dimensions
            n_landscapes,             # m landscapes
            resolution,               # n nodes
            memory_saving
        ) for diag in tqdm(diagrams, desc='Landscapes Progress')
    ]
    return landscapes


def compute_persistence_landscape_norms(landscapes: list) -> np.ndarray:
    """
    Given a list/time series of persistence landscape, computes
    the corresponding normalized L1 and L2 time series
    """
    norms_1 = [np.linalg.norm(ls, 1) for ls in landscapes]
    norms_2 = [np.linalg.norm(ls, 2) for ls in landscapes]
    norms_1 = norms_1/np.linalg.norm(norms_1)
    norms_2 = norms_2/np.linalg.norm(norms_2)
    return np.array([norms_1, norms_2]).T


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the column names of the dataframe:
    - lower casing
    - space swapped for underscore
    """
    # Declares useful anonymous function
    format_column_name = lambda x: x.lower().replace(" ", "_")
    # Copies the input dataframe and updates the column names
    new_df = df.copy()
    new_df.columns = list(map(format_column_name, new_df.columns))
    return new_df


def plot_price_data(df: pd.DataFrame,
                    legend: list,
                    title: str) -> None:
    """
    Given a list of standardized stock price data, plots the
    Adjusted Close value across the whole available timeline.
    """
    plt.figure()
    # Fixes x-ticks interval to c. a year
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=365))
    assets = list(df.columns)
    for asset in assets:
        plt.plot(list(df.index),
                 df[asset],
                 linewidth=.5)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend(legend)
    plt.show()
