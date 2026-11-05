from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri


def r2pandas(rdf):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().rpy2py(rdf)


def pandas2r(df):
    with (ro.default_converter + pandas2ri.converter).context():
        return ro.conversion.get_conversion().py2rpy(df)