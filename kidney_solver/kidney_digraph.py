"""A Digraph class which can be used for representing donor-patient pairs
(as vertices) and their compatibilities (as weighted edges), along with
some related methods.
Modified by Duncan from https://github.com/jamestrimble/kidney_solver
"""

from collections import deque
import pandas

import copy

class KidneyReadException(Exception):
    pass

def cycle_score(cycle, digraph):
    """Calculate the sum of a cycle's edge scores.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
    """

    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score
                        for i in range(len(cycle)))

def failure_aware_cycle_score(cycle, digraph, edge_success_prob):
    """Calculate a cycle's total score, with edge failures and no backarc recourse.

    Args:
        cycle: A list of Vertex objects in the cycle, with the first Vertex not repeated.
        digraph: The digraph in which this cycle appears.
        edge_success_prob: The problem that any given edge will NOT fail
    """

    return sum(digraph.adj_mat[cycle[i-1].id][cycle[i].id].score
                    for i in range(len(cycle))) * edge_success_prob**len(cycle)

class Vertex:
    """A vertex in a directed graph (see the Digraph class)."""

    def __init__(self, id):
        self.id = id
        self.edges = []
        self.sensitized = False

    def __str__(self):
        return ("V{}".format(self.id))

class Edge:
    """An edge in a directed graph (see the Digraph class)."""

    def __init__(self, id, score, src, tgt):
        self.id = id
        self.score = score # edge weight
        self.src = src   # source vertex
        self.tgt = tgt # target vertex

    def __str__(self):
        return ("V" + str(self.src.id) + "-V" + str(self.tgt.id))

class Digraph:
    """A directed graph, in which each edge has a numeric score.

    Data members:
        n: the number of vertices in the digraph
        vs: an array of Vertex objects, such that vs[i].id == i
        es: an array of Edge objects, such that es[i].id = i
    """

    def __init__(self, n):
        """Create a Digraph with n vertices"""
        self.n = n
        self.vs = [Vertex(i) for i in range(n)]
        self.adj_mat = [[None for x in range(n)] for x in range(n)]
        self.es = []

    def add_edge(self, score, source, tgt):
        """Add an edge to the digraph

        Args:
            score: the edge's score, as a float
            source: the source Vertex
            tgt: the edge's target Vertex
        """

        id = len(self.es)
        e = Edge(id, score, source, tgt)
        self.es.append(e)
        source.edges.append(e)
        self.adj_mat[source.id][tgt.id] = e
    
    def find_cycles(self, max_length):
        """Find cycles of length up to max_length in the digraph.

        Returns:
            a list of cycles. Each cycle is represented as a list of
            vertices, with the first vertex _not_ repeated at the end.
        """
        
        return [cycle for cycle in self.generate_cycles(max_length)]

    def generate_cycles(self, max_length):
        """Generate cycles of length up to max_length in the digraph.

        Each cycle yielded by this generator is represented as a list of
        vertices, with the first vertex _not_ repeated at the end.
        """

        vtx_used = [False] * len(self.vs)  # vtx_used[i]==True iff vertex i is in current path

        def cycle(current_path):
            last_vtx = current_path[-1]
            if self.edge_exists(last_vtx, current_path[0]):
                yield current_path[:]
            if len(current_path) < max_length:
                for e in last_vtx.edges: 
                    v = e.tgt
                    if (len(current_path) + shortest_paths_to_low_vtx[v.id] <= max_length
                                and not vtx_used[v.id]):
                        current_path.append(v)
                        vtx_used[v.id] = True
                        for c in cycle(current_path):
                            yield c
                        vtx_used[v.id] = False
                        del current_path[-1]

        # Adjacency lists for transpose graph
        transp_adj_lists = [[] for v in self.vs]
        for edge in self.es:
            transp_adj_lists[edge.tgt.id].append(edge.src)

        for v in self.vs:
            shortest_paths_to_low_vtx = self.calculate_shortest_path_lengths(
                    v, max_length - 1,
                    lambda u: (w for w in transp_adj_lists[u.id] if w.id > v.id))
            vtx_used[v.id] = True
            for c in cycle([v]):
                yield c
            vtx_used[v.id] = False
    
    def get_shortest_path_from_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path from low_vtx to v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""
        return self.calculate_shortest_path_lengths(self.vs[low_vtx], max_path,
                    adj_list_accessor=lambda v: (e.tgt for e in v.edges if e.tgt.id >= low_vtx))

    def get_shortest_path_to_low_vtx(self, low_vtx, max_path):
        """ Returns an array of path lengths. For each v > low_vtx, if the shortest
            path to low_vtx from v is shorter than max_path, then element v of the array
            will be the length of this shortest path. Otherwise, element v will be
            999999999."""
        def adj_list_accessor(v):
            for i in range(low_vtx, len(self.vs)):
                if self.adj_mat[i][v.id]:
                    yield self.vs[i]
            
        return self.calculate_shortest_path_lengths(self.vs[low_vtx], max_path,
                    adj_list_accessor=adj_list_accessor)

    def calculate_shortest_path_lengths(self, from_v, max_dist,
                adj_list_accessor=lambda v: (e.tgt for e in v.edges)):
        """Calculate the length of the shortest path from vertex from_v to each
        vertex with a greater or equal index, using paths containing
        only vertices indexed greater than or equal to from_v.

        Return value: a list of distances of length equal to the number of vertices.
        If the shortest path to a vertex is greater than max_dist, the list element
        will be 999999999.

        Args:
            from_v: The starting vertex
            max_dist: The maximum distance we're interested in
            adj_list_accessor: A function taking a vertex and returning an
                iterable of out-edge targets
        """
        # Breadth-first search
        q = deque([from_v])
        distances = [999999999] * len(self.vs)
        distances[from_v.id] = 0

        while q:
            v = q.popleft()
            #Note: >= is used instead of == on next line in case max_dist<0
            if distances[v.id] >= max_dist:
                break
            for w in adj_list_accessor(v):
                if distances[w.id] == 999999999:
                    distances[w.id] = distances[v.id] + 1
                    q.append(w)

        return distances

    def edge_exists(self, v1, v2):
        """Returns true if and only if an edge exists from Vertex v1 to Vertex v2."""

        return self.adj_mat[v1.id][v2.id] is not None
                    
    def induced_subgraph(self, vertices):
        """Returns the subgraph indiced by a given list of vertices."""

        subgraph = Digraph(len(vertices))
        for i, v in enumerate(vertices):
            for j, w in enumerate(vertices):
                e = self.adj_mat[v.id][w.id]
                if e is not None:
                    new_src = subgraph.vs[i]
                    new_tgt = subgraph.vs[j]
                    subgraph.add_edge(e.score, new_src, new_tgt)
        return subgraph

    # read the *recipient.csv file to label the sensitized recipients
    def KPD_label_sensitized(self, recipient_filename, vtx_index):
        df = pandas.read_csv(recipient_filename)
        df.columns = map(str.lower, df.columns)
        sensitized_ids = df['kpd_candidate_id'].loc[df['highly_sensitized'] == 'Y']
        for n in set(sensitized_ids).intersection(vtx_index.keys()):
            vi = vtx_index[n]
            self.vs[vi].sensitized = True

    # multiply edge weights ending in highly sensitized patients by a factor of (1+beta)
    def augment_weights(self, beta):
        for e in self.es:
            if self.vs[e.tgt.id].sensitized:
                new_score = (1.0 + beta) * e.score
                e.score = new_score

    # inverse of augment_weights
    def unaugment_weights(self, beta):
        for e in self.es:
            if self.vs[e.tgt.id].sensitized:
                new_score = e.score/(1.0 + beta)
                e.score = new_score

    def get_num_sensitized(self):
        num = 0
        for v in self.vs:
            if v.sensitized:
                num += 1
        return num

    def get_num_pairs(self):
        return len(self.vs)

    def fair_copy(self):
        d = Digraph(self.n)
        d.vs = self.vs
        for e in self.es:
            tgt_v = e.tgt
            new_score = e.score if tgt_v.sensitized else 0
            d.add_edge(new_score, e.src, tgt_v)

        return d

    def __str__(self):
        return "\n".join([str(v) for v in self.vs])
        
def read_digraph(lines):
    """Reads a digraph from an array of strings in the input format."""

    vtx_count, edge_count = [int(x) for x in lines[0].split()]
    digraph = Digraph(vtx_count)
    for line in lines[1:edge_count+1]:
        tokens = [x for x in line.split()]
        src_id = int(tokens[0])
        tgt_id = int(tokens[1])
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException("Self-loop from {0} to {0} not permitted".format(src_id))
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]):
            raise KidneyReadException("Duplicate edge from {} to {}".format(src_id, tgt_id))
        score = float(tokens[2])
            
        digraph.add_edge(score, digraph.vs[src_id], digraph.vs[tgt_id])

    if lines[edge_count+1].split()[0] != "-1" or len(lines) < edge_count+2:
        raise KidneyReadException("Incorrect edge count")

    return digraph

# read a KPD graph from *edgeweights.csv file
def read_from_kpd(edgeweights_filename):
    col_names = ['match_run','patient_id', 'patient_pair_id', 'donor_id', 'donor_pair_id', 'weight']
    df = pandas.read_csv(edgeweights_filename , names = col_names, skiprows=1)
    nonzero_edges = df.loc[df['weight'] > 0 ]  # last column is edge weights -- only take nonzero edges
    kpd_edges = nonzero_edges.loc[ ~nonzero_edges['donor_pair_id'].isnull()]# remove NDD edges
    vtx_id = set(list(kpd_edges['patient_id'].unique()) + list(kpd_edges['donor_pair_id'].unique())) # get unique vertex ids

    vtx_count = len(vtx_id)
    digraph = Digraph(vtx_count)
    vtx_index = dict(zip( vtx_id, range(len(vtx_id) ))) # vtx_index[id] gives the index in the digraph

    warned = False
    for index, row in kpd_edges.iterrows():
        src_id = vtx_index[row['donor_pair_id']]
        tgt_id = vtx_index[row['patient_id']]
        score = row['weight']
        if src_id < 0 or src_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(src_id))
        if tgt_id < 0 or tgt_id >= vtx_count:
            raise KidneyReadException("Vertex index {} out of range.".format(tgt_id))
        if src_id == tgt_id:
            raise KidneyReadException("Self-loop from {0} to {0} not permitted".format(src_id))
        if digraph.edge_exists(digraph.vs[src_id], digraph.vs[tgt_id]) & ~warned:
            print "# WARNING: Duplicate edge in file: {}".format(edgeweights_filename)
            warned = True
            # raise KidneyReadException("Duplicate edge from {} to {}".format(src_id, tgt_id))
        if score == 0:
            raise KidneyReadException("Zero-weight edge from {} to {}".format(src_id, tgt_id))

        digraph.add_edge(score, digraph.vs[src_id], digraph.vs[tgt_id])

    return digraph,vtx_index

