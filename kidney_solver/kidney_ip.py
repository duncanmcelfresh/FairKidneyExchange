"""
Solving the kidney-exchange problem using the Gurobi IP solver.
Modified by Duncan from https://github.com/jamestrimble/kidney_solver
"""

import copy
import sys

from kidney_digraph import *
from kidney_ndds import *
import kidney_utils

from gurobipy import *

import numpy as np # added by Duncan
from collections import Counter # added by Duncan

###################################################################################################
#                                                                                                 #
#                                  Code used by all formulations                                  #
#                                                                                                 #
###################################################################################################

class OptConfig(object):
    """The inputs (problem instance and parameters) for an optimisation run

    Data members:
        digraph
        ndds
        max_cycle
        max_chain
        verbose: True if and only if Gurobi output should be writtent to screen and log file
        timelimit
        edge_success_prob
        eef_alt_constraints: True if and only if alternative EEF constraints should be used
        lp_file: The name of a .lp file to write, or None if the file should not be written
        relax: True if and only if the LP relaxation should be solved also
    """

    def __init__(self, digraph, ndds, max_cycle, max_chain, verbose=False,
                 timelimit=None, edge_success_prob=1, eef_alt_constraints=False,
                 lp_file=None, relax=False, multi=1, gap=0, min_fair_score=0):
        self.digraph = digraph
        self.ndds = ndds
        self.max_cycle = max_cycle
        self.max_chain = max_chain
        self.verbose = verbose
        self.timelimit = timelimit
        self.edge_success_prob = edge_success_prob
        self.eef_alt_constraints = eef_alt_constraints
        self.lp_file = lp_file
        self.relax = relax
        self.multi = multi # added by Duncan
        self.gap = gap # added by Duncan
        self.min_fair_score = min_fair_score # added by Duncan

class OptSolution(object):
    """An optimal solution for a kidney-exchange problem instance.
    
    Data members:
        ip_model: The Gurobi Model object
        cycles: A list of cycles in the optimal solution, each represented
            as a list of vertices
        chains: A list of chains in the optimal solution, each represented
            as a Chain object
        total_score: The total score of the solution
    """

    def __init__(self, ip_model, cycles, chains, digraph, edge_success_prob=1, infeasible=False):
        self.ip_model = ip_model
        self.cycles = cycles
        self.chains = chains
        self.digraph = digraph
        self.infeasible = infeasible
        if self.infeasible:
            self.total_score = 0
        else:
            self.total_score = (sum(c.score for c in chains) +
                    sum(failure_aware_cycle_score(c, digraph, edge_success_prob) for c in cycles))
        self.edge_success_prob = edge_success_prob

    # added by Duncan
    def __eq__(self,other):
        """Equal opt solutions have the same set of cycles and chains."""
        if type(self) != type(other):
            return False
        if self.cycles != other.cycles:
            return False
        if self.chains != other.chains:
            return False
        else:
            return True
        
    def display(self):
        """Print the optimal cycles and chains to standard output."""

        print "cycle_count: {}".format(len(self.cycles))
        print "chain_count: {}".format(len(self.chains))
        print "cycles:"
        # cs is a list of cycles, with each cycle represented as a list of vertex IDs
        cs = [[v.id for v in c] for c in self.cycles]
        # Put the lowest-indexed vertex at the start of each cycle
        for i in range(len(cs)):
            min_index_pos = cs[i].index(min(cs[i]))
            cs[i] = cs[i][min_index_pos:] + cs[i][:min_index_pos]
        # Sort the cycles
        cs.sort()
        for c in cs:
            print "\t".join(str(v_id) for v_id in c)
        print "chains:"
        for c in self.chains:
            print str(c.ndd_index) + "\t" + "\t".join(str(v) for v in c.vtx_indices)

    # added by Duncan
    def vertex_mask(self):
        """Returns a numpy array of length |V| containing 1 if the vertex
        participates in the solution, and zero otherwise."""
        # cs is a list of cycles, with each cycle represented as a list of vertex IDs
        v_list = list()
        for cy in self.cycles:
            v_list.append([v.id for v in cy])
        for ch in self.chains:
            v_list.append([v for v in ch.vtx_indices])
#        cy_verts = [v.id for v in self.cycles if self.cycles]
#        ch_verts = [v.id for v in self.chains if self.chains]
        v_mask = np.zeros(self.digraph.n, dtype=np.int)
        if np.sum(v_list) != 0:
            v_mask[ np.concatenate(v_list) ] = 1
        return v_mask

        # added by Duncan
    def num_matched(self):
        return np.sum(self.vertex_mask())

    # added by Duncan
    # returns percentage of sensitized pairs included in solution. if graph contains no sensitized pairs, returns -1
    def num_sensitized(self):
        sens = [ 1 if v.sensitized else 0 for v in self.digraph.vs ]
        num_sens = sens.count(1)
        if num_sens == 0:
            return -1
        else:
            return np.dot(sens, self.vertex_mask()) # for % instead: float( ) / num_sens


    def relabelled_copy(self, old_to_new_vertices, new_digraph):
        """Create a copy of the solution with vertices relabelled.

        If the solution was found on a relabelled copy of the instance digraph, this
        method can be used to transform the solution back to the original digraph. Each
        Vertex v in the OptSolution on which this method is called is replaced in the
        returned copy by old_to_new_vertices[v.id].
        """

        relabelled_cycles = [[old_to_new_vertices[v.id] for v in c] for c in self.cycles]
        relabelled_chains = [Chain(c.ndd_index,
                                   [old_to_new_vertices[i].id for i in c.vtx_indices],
                                   c.score)
                             for c in self.chains]
        return OptSolution(self.ip_model, relabelled_cycles, relabelled_chains,
                           new_digraph, self.edge_success_prob)

    # get score using a digraph with (possibly) different weights
    def get_score(self, digraph, ndds, edge_success_prob):
        score = (sum(c.get_score(digraph, ndds, edge_success_prob) for c in self.chains) +
         sum(failure_aware_cycle_score(c, digraph, edge_success_prob) for c in self.cycles))
        return score

    # calculate and update the score, using the current ndds and digraph
    def update_score(self, ndds):
        self.total_score = self.get_score(self.digraph, ndds, self.edge_success_prob)

    def get_fair_score(self, ndds):
        d_fair = self.digraph.fair_copy()
        ndds_fair = [n.fair_copy() for n in ndds]
        fair_score = self.get_score(d_fair, ndds_fair, self.edge_success_prob)
        return fair_score

# class added by Duncan
class OptCore(object):
    """The set of optimal matchings in a kidney-exchange problem instance.
    
    Data members:
        ip_model: The Gurobi Model object
        solutions: The list of optimal solutions (OptSolution objects)
        total_score: The total score of the solutions
    """
    def __init__(self, ip_model, digraph, max_size, solutions, infeasible=False):
        self.ip_model = ip_model
        self.digraph = digraph
        self.solutions = solutions
        self.size = len(solutions)
        self.max_size = max_size
        self.infeasible = infeasible
        if self.infeasible:
            self.total_score = 0
        else:
            self.total_score = solutions[0].total_score

    def display(self):
        print "number of optimal exchanges: {}".format(len(self.solutions))
        print "number of optima requested: {}".format(self.max_size)
        print "score: {}".format(self.total_score) 
        print "pairs: {}".format(self.digraph.n)
        print "NDDs: {}".format("???")

    def score_counter(self):
        scores = [ s.total_score for s in self.solutions ]
        return Counter(scores)

    # counter of percentage-sensitized in complete core
    def pct_sensitized_counter(self):
        pcts = [ s.num_sensitized() for s in self.solutions ]
        return Counter(pcts)

    # returns a dict with keys = score, value = list of fairness percentages
    def pct_sensitized_by_score(self):
        sol_by_score = self.solutions_by_score()
        d = dict()
        for score,sols in sol_by_score.iteritems():
            d[score] = [ s.num_sensitized() for s in sols ]
        return d

    # returns a dict with keys = score, value = list of solutions with that score
    def solutions_by_score(self):
        scores = self.score_counter().keys()
        d = dict()
        for s in scores:
            d[s] = []
        for sol in self.solutions:
            d[sol.total_score].append(sol)
        return d

    def vertex_participation(self):
        """Return a numpy array containing the number of optimal solutions that
        each vertex participates in."""
        v_part = np.zeros(self.digraph.n, dtype=np.int)
        for sol in self.solutions:
            v_part += sol.vertex_mask()
        return v_part

    def write_vertex_participation(self, filename, cycle_cap, chain_cap, ndds):
        v_part = self.vertex_participation()
        with open(filename,'w') as f:
            f.write("number of optimal exchanges: {}\n".format(len(self.solutions)))
            f.write("number of optima requested: {}\n".format(self.max_size))
            f.write("score: {}\n".format(self.total_score) )
            f.write("chain cap: {}\n".format(chain_cap))
            f.write("cycle cap: {}\n".format(cycle_cap))
            f.write("pairs: {}\n".format(self.digraph.n))
            f.write("NDDs: {}\n".format(ndds))
            f.write("{0:5s} : {1:7s}\n".format("vertx","participation"))
            for i,vp in enumerate(v_part):
                f.write("{0:5d} : {1:7d}\n".format(i,vp))               

    def write_fairness(self, filename, cycle_cap, chain_cap, ndds ):
        sens = [ 1 if v.sensitized else 0 for v in self.digraph.vs ]
        num_sens = sens.count(1)
        sens_ind = [ str(i) for i in range(self.digraph.n) if self.digraph.vs[i].sensitized ]
        fair_by_score = self.pct_sensitized_by_score()
        with open(filename,'w') as f:
            f.write("core_size: {}\n".format(len(self.solutions)))
            f.write("max_core_size: {}\n".format(self.max_size))
            f.write("optimal_score: {}\n".format(self.total_score))
            f.write("chain_cap: {}\n".format(chain_cap))
            f.write("cycle_cap: {}\n".format(cycle_cap))
            f.write("num_pairs: {}\n".format(self.digraph.n))
            f.write("num_ndds: {}\n".format(ndds))
            f.write("num_sensitized: {}\n".format(num_sens))
            f.write("sensitized_pairs: {}\n".format(' '.join(sens_ind)))
            f.write("{0:11s} {1:7s} {2:8s} {3:8s}\n".format("score","count","min_sens", "max_sens"))
            for score, counts in self.score_counter().most_common():
                max_sens = max(fair_by_score[score])
                min_sens = min(fair_by_score[score])
                f.write("{0:11.5f} {1:7d} {2:8d} {3:8d}\n".format(score,counts,min_sens,max_sens))

    def write_all_scores(self, filename, cycle_cap, chain_cap, ndds ):
        sens = [ 1 if v.sensitized else 0 for v in self.digraph.vs ]
        num_sens = sens.count(1)
        sens_ind = [ str(i) for i in range(self.digraph.n) if self.digraph.vs[i].sensitized ]
        fair_by_score = self.pct_sensitized_by_score()
        with open(filename,'w') as f:
            f.write("core_size: {}\n".format(len(self.solutions)))
            f.write("max_core_size: {}\n".format(self.max_size))
            f.write("optimal_score: {}\n".format(self.total_score))
            f.write("chain_cap: {}\n".format(chain_cap))
            f.write("cycle_cap: {}\n".format(cycle_cap))
            f.write("num_pairs: {}\n".format(self.digraph.n))
            f.write("num_ndds: {}\n".format(ndds))
            f.write("num_sensitized: {}\n".format(num_sens))
            f.write("sensitized_pairs: {}\n".format(' '.join(sens_ind)))
            f.write("{0:7s} {1:8s}\n".format("score","num_sensitized"))
            for sol in self.solutions:
                f.write("{0:7f} {1:8d}\n".format(sol.total_score, sol.num_sensitized()))



def optimise(model, cfg):
    if cfg.lp_file:
        model.update()
        model.write(cfg.lp_file)
        sys.exit(0)
    elif cfg.relax:
        model.update()
        r = model.relax()
        r.optimize()
        print "lp_relax_obj_val:", r.obj_val
        print "lp_relax_solver_status:", r.status
        sys.exit(0)
    else:
        model.optimize()

def optimise_relabelled(formulation_fun, cfg):
    """Optimise on a relabelled graph such that vertices are sorted in descending
        order of (indegree + outdegree)"""

    in_degs = [0] * cfg.digraph.n
    for e in cfg.digraph.es:
        in_degs[e.tgt.id] += 1

    sorted_vertices = sorted(cfg.digraph.vs,
                             key=lambda v: len(v.edges) + in_degs[v.id],
                             reverse=True)
    
    relabelled_digraph = cfg.digraph.induced_subgraph(sorted_vertices)

    # old_to_new_vtx[i] is the vertex in the new graph corresponding to vertex
    # i in the original digraph
    old_to_new_vtx = [None] * cfg.digraph.n
    for i, v in enumerate(sorted_vertices):
        old_to_new_vtx[v.id] = relabelled_digraph.vs[i]

    relabelled_ndds = create_relabelled_ndds(cfg.ndds, old_to_new_vtx)
    relabelled_cfg = copy.copy(cfg)
    relabelled_cfg.digraph = relabelled_digraph
    relabelled_cfg.ndds = relabelled_ndds

    opt_result = formulation_fun(relabelled_cfg)
    return opt_result.relabelled_copy(sorted_vertices, cfg.digraph)

# def create_ip_model(time_limit, verbose): # changed by Duncan
def create_ip_model(time_limit, verbose, multi=1, gap=0):
    """Create a Gurobi Model."""

    m = Model("kidney-mip")
    if not verbose:
        m.params.outputflag = 0
    m.params.mipGap = 0
    if multi > 1:
        m.setParam(GRB.Param.PoolSolutions,multi) # number of solutions to collect
        m.setParam(GRB.Param.PoolGap, gap) # only collect optimal solutions (gap = 0)
        m.setParam(GRB.Param.PoolSearchMode, 2) # exhaustive search
    if time_limit is not None:
        m.params.timelimit = time_limit
    return m

###################################################################################################
#                                                                                                 #
#                                       Uncapped formulation                                      #
#                                                                                                 #
###################################################################################################

def add_unlimited_vars_and_constraints(digraph, ndds, m):
    """Add the IP variables and constraints for chains in the uncapped edge formulation. 

    Args:
        digraph: the instance digraph
        ndds: a list of NDDs in the instance
        m: The Gurobi model
    """

    for v in digraph.vs:
        v.grb_vars_in  = []
        v.grb_vars_out = []

    for ndd in ndds:
        ndd_edge_vars = []
        for e in ndd.edges:
            edge_var = m.addVar(vtype=GRB.BINARY)
            e.edge_var = edge_var
            ndd_edge_vars.append(edge_var)
            e.target_v.grb_vars_in.append(edge_var)
        m.update()
        m.addConstr(quicksum(ndd_edge_vars) <= 1)

    # Add pair->pair edge variables
    for e in digraph.es:
        e.grb_vars = []
        edge_var = m.addVar(vtype=GRB.BINARY)
        e.grb_vars.append(edge_var)
        e.src.grb_vars_out.append(edge_var)
        e.tgt.grb_vars_in.append(edge_var)

    m.update()

    for v in digraph.vs:
        if len(v.grb_vars_in) > 1:
            m.addConstr(quicksum(v.grb_vars_in) <= 1)

    # Sum of edges into a vertex must be >= sum of edges out
    for v in digraph.vs:
        m.addConstr(quicksum(v.grb_vars_in) >= quicksum(v.grb_vars_out))

def optimise_uuef(cfg):
    """Optimise using the uncapped edge formulation.

    Args:
        cfg: an OptConfig object

    Returns:
        an OptSolution object
    """

    if cfg.edge_success_prob != 1:
        raise ValueError("This formulation does not support failure-aware matching.")

    m = create_ip_model(cfg.timelimit, cfg.verbose)

    add_unlimited_vars_and_constraints(cfg.digraph, cfg.ndds, m)

    obj_expr = ( quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                 quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars) )
   
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    optimise(m, cfg)

    # Try all possible cycle start positions
    cycle_start_vv = range(cfg.digraph.n)

    cycle_next_vv = {}
    for e in cfg.digraph.es:
        for var in e.grb_vars:
            if var.x > 0.1:
                cycle_next_vv[e.src.id] = e.tgt.id

    return OptSolution(ip_model=m,
                       cycles=kidney_utils.selected_edges_to_cycles(
                                    cfg.digraph, cycle_start_vv, cycle_next_vv),
                       chains=kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds),
                       digraph=cfg.digraph)
        
###################################################################################################
#                                                                                                 #
#                  Chain vars and constraints (used by HPIEF', HPIEF'' and PICEF)                 #
#                                                                                                 #
###################################################################################################

def add_chain_vars_and_constraints(digraph, ndds, max_chain, m, vtx_to_vars,
                                   store_edge_positions=False):
    """Add the IP variables and constraints for chains in PICEF and HPIEF'.

    Args:
        ndds: a list of NDDs in the instance
        max_chain: the chain cap
        m: The Gurobi model
        vtx_to_vars: A list such that for each Vertex v in the Digraph,
            vtx_to_vars[v.id] will contain the Gurobi variables representing
            edges pointing to v.
        store_edge_positions: if this is True, then an attribute grb_edge_positions
            will be added to edges that have associated Gurobi variables.
            edge.grb_edge_positions[i] will indicate the position of the edge respresented
            by edge.grb_vars[i]. (default: False)
    """

    if max_chain > 0:
        for v in digraph.vs:
            v.grb_vars_in  = [[] for i in range(max_chain-1)]
            v.grb_vars_out = [[] for i in range(max_chain-1)]

        for ndd in ndds:
            ndd_edge_vars = []
            for e in ndd.edges:
                edge_var = m.addVar(vtype=GRB.BINARY)
                e.edge_var = edge_var
                ndd_edge_vars.append(edge_var)
                vtx_to_vars[e.target_v.id].append(edge_var)
                if max_chain>1: e.target_v.grb_vars_in[0].append(edge_var)
            m.update()
            m.addConstr(quicksum(ndd_edge_vars) <= 1)

        dists_from_ndd = kidney_utils.get_dist_from_nearest_ndd(digraph, ndds)

        # Add pair->pair edge variables, indexed by position in chain
        for e in digraph.es:
            e.grb_vars = []
            if store_edge_positions:
                e.grb_var_positions = []
            for i in range(max_chain-1):
                if dists_from_ndd[e.src.id] <= i+1:
                    edge_var = m.addVar(vtype=GRB.BINARY)
                    e.grb_vars.append(edge_var)
                    if store_edge_positions:
                        e.grb_var_positions.append(i+1)
                    vtx_to_vars[e.tgt.id].append(edge_var)
                    e.src.grb_vars_out[i].append(edge_var)
                    if i < max_chain-2:
                        e.tgt.grb_vars_in[i+1].append(edge_var)

        m.update()

        # At each chain position, sum of edges into a vertex must be >= sum of edges out
        for i in range(max_chain-1):
            for v in digraph.vs:
                m.addConstr(quicksum(v.grb_vars_in[i]) >= quicksum(v.grb_vars_out[i]))

###################################################################################################
#                                                                                                 #
#                                Code shared by HPIEF' and HPIEF''                                #
#                                                                                                 #
###################################################################################################

def add_hpief_prime_vars_partial_red(max_cycle, digraph, m, hpief_2_prime=False):
    vars_and_edges = [] # A list of (gurobi_var, position, edge, low_vertex) tuples

    # max_pos is the maximum edge position for which variables may be created
    max_pos = max_cycle-2 if hpief_2_prime else max_cycle-1
    
    # Index i is in the list edge_vars_in[pos][v][low_v] if and only if
    # vars_and_edges[i] corresponds to an edge at position pos, pointing to vertex
    # v, in low_v's graph copy 
    edge_vars_in = [[[[] for __ in range(digraph.n)] for __ in range(digraph.n)] for __ in range(max_pos + 1)]

    # Index i is in the list edge_vars_out[pos][v][low_v] if and only if
    # vars_and_edges[i] corresponds to an edge at position pos, leaving vertex
    # v, in low_v's graph copy 
    edge_vars_out = [[[[] for __ in range(digraph.n)] for __ in range(digraph.n)] for __ in range(max_pos + 1)]

    for low_vtx in range(digraph.n-1):
        # Length of shortest path from low vertex to each vertex with a higher index
        # Default value is 999999999 (which represents infinity)
        shortest_path_from_lv = digraph.get_shortest_path_from_low_vtx(low_vtx, max_cycle-1) 
        shortest_path_to_lv = digraph.get_shortest_path_to_low_vtx(low_vtx, max_cycle-1) 

        for v1 in digraph.vs[low_vtx+1:]:
            for e in v1.edges:
                if e.tgt.id >=low_vtx:
                    for pos in xrange(1, max_pos + 1):
                        if (shortest_path_from_lv[e.src.id] <= pos and
                                    shortest_path_to_lv[e.tgt.id] < max_cycle - pos):
                            new_var = m.addVar(vtype=GRB.BINARY)
                            vars_and_edges.append((new_var, pos, e, low_vtx))
                            idx = len(vars_and_edges) - 1 # Index of tuple just added
                            edge_vars_in[pos][e.tgt.id][low_vtx].append(idx)
                            edge_vars_out[pos][e.src.id][low_vtx].append(idx)
    m.update()
    return vars_and_edges, edge_vars_in, edge_vars_out

def add_hpief_prime_vars_full_red(max_cycle, digraph, m, hpief_2_prime=False):
    vars_and_edges = [] # A list of (gurobi_var, position, edge, low_vertex) tuples

    # max_pos is the maximum edge position for which variables may be created
    max_pos = max_cycle-2 if hpief_2_prime else max_cycle-1
    
    edge_vars_in = [[[[] for __ in range(digraph.n)] for __ in range(digraph.n)] for __ in range(max_pos + 1)]
    edge_vars_out = [[[[] for __ in range(digraph.n)] for __ in range(digraph.n)] for __ in range(max_pos + 1)]

    edges_seen = set()  # (low_v_id, src_v_id, tgt_v_id, pos) tuples
    for cycle in digraph.generate_cycles(max_cycle):
        for i in range(1, len(cycle)-1):
            edges_seen.add((cycle[0].id, cycle[i].id, cycle[i+1].id, i))
        if not hpief_2_prime or len(cycle) < max_cycle:
            edges_seen.add((cycle[0].id, cycle[-1].id, cycle[0].id, len(cycle)-1))
            
    for low_v, src_v, tgt_v, pos in edges_seen:
        new_var = m.addVar(vtype=GRB.BINARY)
        e = digraph.adj_mat[src_v][tgt_v]
        vars_and_edges.append((new_var, pos, e, low_v))
        idx = len(vars_and_edges) - 1 # Index of tuple just added
        edge_vars_in[pos][tgt_v][low_v].append(idx)
        edge_vars_out[pos][src_v][low_v].append(idx)
    m.update()
    return vars_and_edges, edge_vars_in, edge_vars_out

def add_hpief_prime_vars_and_constraints(max_cycle, digraph, vtx_to_in_edges, m, full_red, hpief_2_prime=False):
    max_pos = max_cycle-2 if hpief_2_prime else max_cycle-1

    if full_red:
        vars_and_edges, edge_vars_in, edge_vars_out = add_hpief_prime_vars_full_red(max_cycle, digraph, m, hpief_2_prime)
    else:
        vars_and_edges, edge_vars_in, edge_vars_out = add_hpief_prime_vars_partial_red(max_cycle, digraph, m, hpief_2_prime)
    
    for grb_var, pos, edge, low_vtx in vars_and_edges:
        vtx_to_in_edges[edge.tgt.id].append(grb_var)
        if pos==1:
            vtx_to_in_edges[edge.src.id].append(grb_var)
        if hpief_2_prime and pos == max_cycle - 2 and edge.tgt.id != low_vtx:
            vtx_to_in_edges[low_vtx].append(grb_var)
        
    # Capacity constraint for vertices
    for l in vtx_to_in_edges:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)
    
    # Cycle flow-conservation constraint for vertices
    for pos in range(1, max_pos):
        for v in range(digraph.n):
            for low_v_id in range(v):
                in_vars  = [vars_and_edges[i][0] for i in edge_vars_in[pos][v][low_v_id]]
                out_vars = [vars_and_edges[i][0] for i in edge_vars_out[pos+1][v][low_v_id]]
                if len(in_vars) > 0 or len(out_vars) > 0:
                    m.addConstr(quicksum(in_vars) == quicksum(out_vars))

    return vars_and_edges

def optimise_hpief_prime(cfg, full_red=False, hpief_2_prime=False):
    """Optimise using the HPIEF' or HPIEF'' formulation.

    The HPIEF' model is based on HPIEF, but does not include cycle-edge variables at position zero.
    HPIEF'' also removes variables corresponding to edges at the last possible position of a cycle. 

    Args:
        cfg: an OptConfig object
        full_red: True if cycles should be generated in order to reduce number of variables further
        hpief_2_prime: Use HPIEF''? Default: HPIEF'

    Returns:
        an OptSolution object
    """

    if cfg.edge_success_prob != 1:
        raise ValueError("This formulation does not support failure-aware matching.")

    if cfg.max_cycle < 3:
        hpief_2_prime = False

#    m = create_ip_model(cfg.timelimit, cfg.verbose) # changed by Duncan
    m = create_ip_model(cfg.timelimit, cfg.verbose, multi=cfg.multi, gap=cfg.gap) # changed by Duncan
    m.params.method = 2
    m.params.presolve = 0

    # For each vertex v, a list of variables corresponding to in-edges to v
    vtx_to_in_edges = [[] for __ in cfg.digraph.vs]

    add_chain_vars_and_constraints(cfg.digraph, cfg.ndds, cfg.max_chain, m, vtx_to_in_edges)

    vars_and_edges = add_hpief_prime_vars_and_constraints(
            cfg.max_cycle, cfg.digraph, vtx_to_in_edges, m, full_red, hpief_2_prime)

    obj_terms = []
    for var, pos, edge, low_v_id in vars_and_edges:
        score = edge.score
        if pos==1:
            score += cfg.digraph.adj_mat[low_v_id][edge.src.id].score
        if hpief_2_prime and pos==cfg.max_cycle - 2 and edge.tgt.id != low_v_id:
            score += cfg.digraph.adj_mat[edge.tgt.id][low_v_id].score
        obj_terms.append(score * var)

    obj_expr = quicksum(obj_terms)
   
    if cfg.max_chain > 0:
        obj_expr += quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) 
        obj_expr += quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars)
    
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    optimise(m, cfg)
        
    if cfg.multi > 1:
        nSolutions = m.SolCount
        solutions = [None] * nSolutions
        print 'Number of solutions found: ' + str(nSolutions)
        for n_sol in range(nSolutions):
            m.setParam(GRB.Param.SolutionNumber, n_sol)
            cycle_start_vv = []
            cycle_next_vv = {}
        
            for var, pos, edge, low_v_id in vars_and_edges:
                if var.x > 0.1:
                    cycle_next_vv[edge.src.id] = edge.tgt.id
                    if pos == 1:
                        cycle_start_vv.append(low_v_id)
                        cycle_next_vv[low_v_id] = edge.src.id
                    if hpief_2_prime and pos == cfg.max_cycle - 2 and edge.tgt.id != low_v_id:
                        cycle_next_vv[edge.tgt.id] = low_v_id  

            solutions[n_sol] = OptSolution(ip_model=m,
                           cycles=kidney_utils.selected_edges_to_cycles(
                           cfg.digraph, cycle_start_vv, cycle_next_vv),
                           chains=[] if cfg.max_chain==0 else kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds),
                           digraph=cfg.digraph)

        return OptCore(m, cfg.digraph, cfg.multi,  solutions)
    else:        
        cycle_start_vv = []
        cycle_next_vv = {}
        
        for var, pos, edge, low_v_id in vars_and_edges:
            if var.x > 0.1:
                cycle_next_vv[edge.src.id] = edge.tgt.id
                if pos == 1:
                    cycle_start_vv.append(low_v_id)
                    cycle_next_vv[low_v_id] = edge.src.id
                if hpief_2_prime and pos == cfg.max_cycle - 2 and edge.tgt.id != low_v_id:
                    cycle_next_vv[edge.tgt.id] = low_v_id  
            
        return OptSolution(ip_model=m,
                       cycles=kidney_utils.selected_edges_to_cycles(
                                    cfg.digraph, cycle_start_vv, cycle_next_vv),
                       chains=[] if cfg.max_chain==0 else kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds),
                       digraph=cfg.digraph)

###################################################################################################
#                                                                                                 #
#                                               HPIEF'                                            #
#                                                                                                 #
###################################################################################################

def optimise_hpief_prime_full_red(cfg):
    return optimise_hpief_prime(cfg, True)

###################################################################################################
#                                                                                                 #
#                                             HPIEF''                                             #
#                                                                                                 #
###################################################################################################

def optimise_hpief_2prime(cfg, full_red=False):
    return optimise_hpief_prime(cfg, full_red, hpief_2_prime=True)

def optimise_hpief_2prime_full_red(cfg):
    return optimise_hpief_2prime(cfg, full_red=True)

###################################################################################################
#                                                                                                 #
#                                              PICEF                                              #
#                                                                                                 #
###################################################################################################

def optimise_picef(cfg):
    """Optimise using the PICEF formulation.

    Args:
        cfg: an OptConfig object

    Returns:
        an OptSolution object
    """

    cycles = cfg.digraph.find_cycles(cfg.max_cycle)

    # m = create_ip_model(cfg.timelimit, cfg.verbose)
    m = create_ip_model(cfg.timelimit, cfg.verbose, multi=cfg.multi, gap=cfg.gap) # changed by Duncan
    m.params.method = 2

    cycle_vars = [m.addVar(vtype=GRB.BINARY) for __ in cycles]
    m.update()
    
    vtx_to_vars = [[] for __ in cfg.digraph.vs]
    
    add_chain_vars_and_constraints(cfg.digraph, cfg.ndds, cfg.max_chain, m,
            vtx_to_vars, store_edge_positions=cfg.edge_success_prob!=1)

    for i, c in enumerate(cycles):
        for v in c:
            vtx_to_vars[v.id].append(cycle_vars[i])

    # added by Duncan
    #min_fair_score = sens.count(1)
    #pct = 0
    #min_sensitized = pct * min_fair_score
    # print "len dot: {}".format(len(np.dot(sens,v_used)))
    #print "# sensitized = {}".format(cfg.min_fair_score)
    # m.addConstr(np.dot(sens,v_used) >= cfg.min_fair_score)
    #print "in IP: num sens: {}".format(cfg.min_fair_score)
    if cfg.min_fair_score > 0:
        # sens = [1 if v.sensitized else 0 for v in cfg.digraph.vs]
        # v_used = [ quicksum(l) if len(l)>0 else 0 for l in vtx_to_vars ]
        # if cfg.edge_success_prob == 1:
        #     m.addConstr(quicksum(sens[i]*v_used[i] for i in range(cfg.digraph.n)) >= cfg.min_fair_score)
        # below is for failure-aware... the fair score (from only highly sensitized patients) must be above the threshold
        # WARNING: this adds extra variables to the model... should not do this.
        # else:
        d_fair = cfg.digraph.fair_copy()
        # ndd_fair = [n.fair_copy() for n in cfg.ndds]
        # m_copy = m.copy()
        # vtx_to_vars_fair = [[] for __ in d_fair.vs]
        # add_chain_vars_and_constraints(d_fair, ndd_fair, cfg.max_chain, m,
        #                                vtx_to_vars_fair, store_edge_positions=cfg.edge_success_prob != 1)
        if cfg.max_chain == 0:
            fair_score = quicksum(failure_aware_cycle_score(c, d_fair, cfg.edge_success_prob) * var
                     for c, var in zip(cycles, cycle_vars))
        elif cfg.edge_success_prob == 1:
            fair_score = (quicksum(cycle_score(c, d_fair) * var for c, var in zip(cycles, cycle_vars)) +
                        quicksum(e.score * e.edge_var * e.target_v.sensitized for ndd in cfg.ndds for e in ndd.edges) +
                        quicksum(e.score * var * e.tgt.sensitized for e in cfg.digraph.es for var in e.grb_vars))
        else:
            fair_score = ( quicksum(failure_aware_cycle_score(c, d_fair, cfg.edge_success_prob) * var
                                  for c, var in zip(cycles, cycle_vars)) +
                         quicksum(e.score*cfg.edge_success_prob * e.edge_var * e.target_v.sensitized
                                  for ndd in cfg.ndds for e in ndd.edges) +
                         quicksum(e.score*cfg.edge_success_prob**(pos+1) * var * e.tgt.sensitized
                                for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))
        m.addConstr(fair_score >= cfg.min_fair_score)

    for l in vtx_to_vars:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)

    if cfg.max_chain==0:
        obj_expr = quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                            for c, var in zip(cycles, cycle_vars))
    elif cfg.edge_success_prob == 1:
        obj_expr = ( quicksum(cycle_score(c, cfg.digraph) * var for c, var in zip(cycles, cycle_vars)) +
                     quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) +
                     quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars) )
    else:
        obj_expr = ( quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                              for c, var in zip(cycles, cycle_vars)) +
                     quicksum(e.score*cfg.edge_success_prob * e.edge_var
                              for ndd in cfg.ndds for e in ndd.edges) +
                     quicksum(e.score*cfg.edge_success_prob**(pos+1) * var
                            for e in cfg.digraph.es for var, pos in zip(e.grb_vars, e.grb_var_positions)))

    m.setObjective(obj_expr, GRB.MAXIMIZE)
    optimise(m, cfg)

    if cfg.multi > 1:
        if m.status == GRB.status.INFEASIBLE:
            return OptCore(m, cfg.digraph, cfg.multi, [], True)
        else:
            nSolutions = m.SolCount
            solutions = [None] * nSolutions
            for n_sol in range(nSolutions):
                m.setParam(GRB.Param.SolutionNumber, n_sol)
                solutions[n_sol] = OptSolution(ip_model=m,
                               cycles=[c for c, v in zip(cycles, cycle_vars) if v.Xn > 0.5],
                               chains=[] if cfg.max_chain==0 else kidney_utils.get_optimal_chains(
                                    cfg.digraph, cfg.ndds, cfg.edge_success_prob),
                               digraph=cfg.digraph,
                               edge_success_prob=cfg.edge_success_prob)
            # Print objective values of solutions
    #        for e in range(nSolutions):
    #            m.setParam(GRB.Param.SolutionNumber, e)
    #            print '%g ' % m.PoolObjVal
    ##            if e % 15 == 14:
    ##                print('')
    #        print ''
            return OptCore(m, cfg.digraph, cfg.multi, solutions)
    else:
        if m.status == GRB.status.INFEASIBLE:
            return OptSolution(ip_model=m,
                           cycles=[],
                           chains=[],
                           digraph=cfg.digraph,
                           edge_success_prob=cfg.edge_success_prob,
                           infeasible = True)
        else:
            return OptSolution(ip_model=m,
                               cycles=[c for c, v in zip(cycles, cycle_vars) if v.x > 0.5],
                               chains=[] if cfg.max_chain==0 else kidney_utils.get_optimal_chains(
                                    cfg.digraph, cfg.ndds, cfg.edge_success_prob),
                               digraph=cfg.digraph,
                               edge_success_prob=cfg.edge_success_prob)

###################################################################################################
#                                                                                                 #
#                                        Cycle formulation                                        #
#                                                                                                 #
###################################################################################################

def optimise_ccf(cfg):
    """Optimise using the cycle formulation (with one var per cycle and one var per chain).

    Args:
        cfg: an OptConfig object

    Returns:
        an OptSolution object
    """

    cycles = cfg.digraph.find_cycles(cfg.max_cycle)
    chains = find_chains(cfg.digraph, cfg.ndds, cfg.max_chain, cfg.edge_success_prob)
        
    m = create_ip_model(cfg.timelimit, cfg.verbose)
    m.params.method = 2

    cycle_vars = [m.addVar(vtype=GRB.BINARY) for __ in cycles]
    chain_vars = [m.addVar(vtype=GRB.BINARY) for __ in chains]
    m.update()
    
    ndd_to_vars = [[] for __ in cfg.ndds]
    vtx_to_vars = [[] for __ in cfg.digraph.vs]
    
    for var, c in zip(cycle_vars, cycles):
        for v in c:
            vtx_to_vars[v.id].append(var)

    for var, c in zip(chain_vars, chains):
        ndd_to_vars[c.ndd_index].append(var)
        for v in c.vtx_indices:
            vtx_to_vars[v].append(var)

    # Each donor-patient pair and each each NDD is in at most one chosen cycle or chain
    for l in vtx_to_vars + ndd_to_vars:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)

    obj_expr = (quicksum(failure_aware_cycle_score(c, cfg.digraph, cfg.edge_success_prob) * var
                         for (c, var) in zip(cycles, cycle_vars)) +
                quicksum(c.score * var for (c, var) in zip(chains, chain_vars)))
        
    m.setObjective(obj_expr, GRB.MAXIMIZE)
    optimise(m, cfg)

    return OptSolution(ip_model=m,
                       cycles=[c for c, v in zip(cycles, cycle_vars) if v.x > 0.5],
                       chains=[c for c, v in zip(chains, chain_vars) if v.x > 0.5],
                       digraph=cfg.digraph,
                       edge_success_prob=cfg.edge_success_prob)

###################################################################################################
#                                                                                                 #
#                                    Extended Edge Formulation                                    # 
#                                                                                                 #
###################################################################################################

def add_eef_vars_partial_red(max_cycle, digraph, m):
    vars_and_edges = [] # A list of (gurobi_var, edge, low_vertex) tuples

    # Index i is in the list edge_vars_in[low_v][v] if and only if
    # vars_and_edges[i] corresponds to an edge pointing to vertex v, in low_v's graph copy 
    edge_vars_in = [[[] for __ in range(digraph.n)] for __ in range(digraph.n)]

    # Index i is in the list edge_vars_out[low_v][v] if and only if
    # vars_and_edges[i] corresponds to an edge leaving vertex v, in low_v's graph copy 
    edge_vars_out = [[[] for __ in range(digraph.n)] for __ in range(digraph.n)]

    for low_vtx in range(digraph.n-1):
        # Length of shortest path from low vertex to each vertex with a higher index
        # Default value is 999999999 (which represents infinity)
        shortest_path_from_lv = digraph.get_shortest_path_from_low_vtx(low_vtx, max_cycle-1) 
        shortest_path_to_lv = digraph.get_shortest_path_to_low_vtx(low_vtx, max_cycle-1) 

        for v1 in digraph.vs[low_vtx:]:
            for e in v1.edges:
                if e.tgt.id >=low_vtx:
                    if (shortest_path_from_lv[e.src.id] +
                                shortest_path_to_lv[e.tgt.id] < max_cycle):
                        new_var = m.addVar(vtype=GRB.BINARY)
                        vars_and_edges.append((new_var, e, low_vtx))
                        idx = len(vars_and_edges) - 1 # Index of tuple just added
                        edge_vars_in[low_vtx][e.tgt.id].append(idx)
                        edge_vars_out[low_vtx][e.src.id].append(idx)
    m.update()
    return vars_and_edges, edge_vars_in, edge_vars_out

def add_eef_vars_full_red(max_cycle, digraph, m):
    vars_and_edges = [] # A list of (gurobi_var, edge, low_vertex) tuples

    edge_vars_in = [[[] for __ in range(digraph.n)] for __ in range(digraph.n)]
    edge_vars_out = [[[] for __ in range(digraph.n)] for __ in range(digraph.n)]

    edges_seen = set()  # (low_v_id, src_v_id, tgt_v_id) tuples
    for cycle in digraph.generate_cycles(max_cycle):
        for i in range(len(cycle)):
            edges_seen.add((cycle[0].id, cycle[i-1].id, cycle[i].id))
            
    for low_v, src_v, tgt_v in edges_seen:
        new_var = m.addVar(vtype=GRB.BINARY)
        e = digraph.adj_mat[src_v][tgt_v]
        vars_and_edges.append((new_var, e, low_v))
        idx = len(vars_and_edges) - 1 # Index of tuple just added
        edge_vars_in[low_v][tgt_v].append(idx)
        edge_vars_out[low_v][src_v].append(idx)
    m.update()
    return vars_and_edges, edge_vars_in, edge_vars_out

def add_eef_vars_and_constraints(max_cycle, digraph, m, full_red, eef_alt_constraints, vtx_to_in_edges):
    if full_red:
        vars_and_edges, edge_vars_in, edge_vars_out = add_eef_vars_full_red(max_cycle, digraph, m)
    else:
        vars_and_edges, edge_vars_in, edge_vars_out = add_eef_vars_partial_red(max_cycle, digraph, m)
    
    for grb_var, edge, low_vtx in vars_and_edges:
        vtx_to_in_edges[edge.tgt.id].append(grb_var)
        
    # Capacity constraint for vertices
    for l in vtx_to_in_edges:
        if len(l) > 0:
            m.addConstr(quicksum(l) <= 1)
    
    # Cycle flow-conservation constraint for vertices
    for v in range(digraph.n):
        for low_v_id in range(v):
            in_vars  = [vars_and_edges[i][0] for i in edge_vars_in[low_v_id][v]]
            out_vars = [vars_and_edges[i][0] for i in edge_vars_out[low_v_id][v]]
            if len(in_vars) > 0 or len(out_vars) > 0:
                m.addConstr(quicksum(in_vars) == quicksum(out_vars))

    if eef_alt_constraints:
        for low_v_id in range(v):
            edge_indices_in_graph_copy = [i for indices in edge_vars_in[low_v_id] for i in indices]

            edge_vars_leaving_l = []
            edge_vars_not_involving_l = []    # Edge vars where low_v_id is neither the src nor the tgt
            for i in edge_indices_in_graph_copy:
                var, edge, _ = vars_and_edges[i]
                if edge.src.id==low_v_id:
                    edge_vars_leaving_l.append(var)
                elif edge.tgt.id!=low_v_id:
                    edge_vars_not_involving_l.append(var)
            
            # Number of edges constraint for each graph copy
            # (Note that this is redundant, but removing it seems to slow the program down
            # quite a bit.)
            m.addConstr(quicksum(edge_vars_not_involving_l) <= max_cycle-2) 

            # In each graph copy, if any edge is selected then an edge is selected
            # that leaves the low-numbered vertex in the graph copy
            # Note: this differs from (9e) in Constantino et al.
            m.addConstr(quicksum(edge_vars_not_involving_l) <=
                        (max_cycle-2) * quicksum(edge_vars_leaving_l))

    else:
        for low_v_id in range(v):
            edge_indices_in_graph_copy = [i for indices in edge_vars_in[low_v_id] for i in indices]

            # Number of edges constraint for each graph copy
            m.addConstr(quicksum(vars_and_edges[i][0] for i in edge_indices_in_graph_copy) <= max_cycle)        

            # Constraint (9e) from Constantino et al.
            sum_of_edge_vars_leaving_l = quicksum(
                    vars_and_edges[i][0] for i in edge_vars_out[low_v_id][low_v_id])
            for i in range(low_v_id+1, digraph.n):
                vars_leaving_i = [vars_and_edges[j][0] for j in edge_vars_out[low_v_id][i]]
                if len(vars_leaving_i):
                    m.addConstr(quicksum(vars_leaving_i) <= sum_of_edge_vars_leaving_l)


    return vars_and_edges

def optimise_eef(cfg, full_red=False):
    """Optimise using the reduced extended edge formulation (Constantino et al., EJOR, 2013).

    Note that this implementation does not yet include chains, and throws an exception
    if a chain cap greater than zero is used.

    Args:
        cfg: an OptConfig object
        full_red: True if cycles should be generated in order to reduce number of variables further

    Returns:
        an OptSolution object
    """

    if cfg.edge_success_prob != 1:
        raise ValueError("This formulation does not support failure-aware matching.")

    m = create_ip_model(cfg.timelimit, cfg.verbose)
    m.params.method = 2
    m.params.presolve = 0

    # For each vertex v, a list of variables corresponding to in-edges to v
    vtx_to_in_edges = [[] for __ in cfg.digraph.vs]

    add_chain_vars_and_constraints(cfg.digraph, cfg.ndds, cfg.max_chain, m, vtx_to_in_edges)

    vars_and_edges = add_eef_vars_and_constraints(cfg.max_cycle, cfg.digraph, m, full_red,
                                                  cfg.eef_alt_constraints, vtx_to_in_edges)

    obj_expr = quicksum(edge.score * var for var, edge, low_v_id in vars_and_edges)
    if cfg.max_chain > 0:
        obj_expr += quicksum(e.score * e.edge_var for ndd in cfg.ndds for e in ndd.edges) 
        obj_expr += quicksum(e.score * var for e in cfg.digraph.es for var in e.grb_vars)

    m.setObjective(obj_expr, GRB.MAXIMIZE)
    optimise(m, cfg)

    cycle_start_vv = []
    cycle_next_vv = {}
    
    for var, edge, low_v_id in vars_and_edges:
        if var.x > 0.1:
            cycle_next_vv[edge.src.id] = edge.tgt.id
            cycle_start_vv.append(edge.src.id)
        
    return OptSolution(ip_model=m,
                       cycles=kidney_utils.selected_edges_to_cycles(
                                    cfg.digraph, cycle_start_vv, cycle_next_vv),
                       chains=[] if cfg.max_chain==0 else kidney_utils.get_optimal_chains(cfg.digraph, cfg.ndds),
                       digraph=cfg.digraph)

def optimise_eef_full_red(cfg):
    return optimise_eef(cfg, full_red=True)
