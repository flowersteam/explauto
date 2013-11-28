# The action/perception cycle is organized as follow :
# - a motor order is send as an pandas.Series, indexed by features.
# - sensory datapoint (pandas.Series) out of sensory primitives are returned

# The action/perception cycle is organized as follow :
# - a motor order is send as an pandas.Series, indexed by features.
# - sensory datapoint (pandas.Series) out of sensory primitives are returned

try:
    import pandas
    pandas_available = True
except ImportError:
    pandas_available = False


class Robot(object):
    """Class for abstracting robots with one input and one output"""

    def __init__(self, cfg):
        return
        self.m_feats  = m_feats # Motor features
        self.m_bounds = m_bounds # Motor bounds
        assert len(m_feats) == len(m_bounds)
        self.s_feats  = s_feats # Sensory features

    def execute_order(self, order, **kwargs):
        """Return the effect"""
        x = self._pre_x(order)
        y = self._execute_order(x, **kwargs) # f to be provided
        return self._post_y(y, x)

    def _pre_x(self, x):
        """Perform test on x and transform it into a tuple"""
        if pandas_available and type(x) == pandas.Series:
            assert set(x.index).issuperset(set(self.m_feats)), ("Error :"
                    + "expected x.index = %s to be included in self.m_feats = %s"
                    % (x.index, self.m_feats))
            return tuple(x.reindex(self.m_feats))
        else:
            assert len(x) == len(self.m_feats)
            return tuple(x)

    def _post_y(self, result, input_object):
        """Return the result in the same format as the input."""
        if pandas_available and type(input_object) == pandas.Series:
            return pandas.Series(result, index = self.s_feats)
        else:
            return tuple(result)


    def close(self):
        """Clean-up for hardware tethering"""
        pass

