import networkx as nx


class OurGraph(object):
    def __init__(self, nxgraph):
        if 'gid' not in nxgraph.graph or type(nxgraph.graph['gid']) is not int \
                or nxgraph.graph['gid'] < 0:
            raise ValueError('Graph ID must be non-negative integers {}'.
                             format(nxgraph.graph.get('gid')))
        if not nx.is_connected(nxgraph):
            raise ValueError('Graph {} must be connected'.
                             format(nxgraph.graph['gid']))
        self.nxgraph = nxgraph

    def type(self):
        raise NotImplementedError()

    def gid(self):
        return self.nxgraph.graph['gid']

    def get_nxgraph(self):
        return self.nxgraph

    def get_image(self):
        raise ValueError('Image data does not exist in {}'.
                         format(self.__class__.__name__))

    def get_complete_graph(self):
        raise ValueError('Complete graph data does not exist in {}'.
                         format(self.__class__.__name__))


class RegularGraph(OurGraph):
    def type(self):
        return 'regular_graph'


class ImageGraph(OurGraph):
    def __init__(self, delaunay_nxgraph, complete_nxgraph, image):
        super(ImageGraph, self).__init__(delaunay_nxgraph)
        self.compete_graph = complete_nxgraph
        self.image = image

    def type(self):
        return 'image_graph'

    def get_image(self):
        return self.image

    def get_complete_graph(self):
        return self.compete_graph
