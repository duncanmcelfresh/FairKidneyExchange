"""
IP formulations for kidney exchange, including PICEF
"""

import argparse
from argparse import ArgumentTypeError
import time
import sys

import kidney_digraph
import kidney_ip
import kidney_utils
import kidney_ndds

import random # added by Duncan

def percent(string):
    val = float(string)
    if val < 0 or val > 1:
        raise ArgumentTypeError("'" + string + "' is not a fraction between 0 and 1.)")
    return val

def solve_kep(cfg, formulation, use_relabelled=True):

    formulations = {
        "uef":  ("Uncapped edge formulation", kidney_ip.optimise_uuef),
        "eef": ("EEF", kidney_ip.optimise_eef),
        "eef_full_red": ("EEF with full reduction by cycle generation", kidney_ip.optimise_eef_full_red),
        "hpief_prime": ("HPIEF'", kidney_ip.optimise_hpief_prime),
        "hpief_prime_full_red": ("HPIEF' with full reduction by cycle generation", kidney_ip.optimise_hpief_prime_full_red),
        "hpief_2prime": ("HPIEF''", kidney_ip.optimise_hpief_2prime),
        "hpief_2prime_full_red": ("HPIEF'' with full reduction by cycle generation", kidney_ip.optimise_hpief_2prime_full_red),
        "picef": ("PICEF", kidney_ip.optimise_picef),
        "cf":   ("Cycle formulation",
                  kidney_ip.optimise_ccf)
    }
    
    if formulation in formulations:
        formulation_name, formulation_fun = formulations[formulation]
        if use_relabelled:
            opt_result = kidney_ip.optimise_relabelled(formulation_fun, cfg)
        else:
            opt_result = formulation_fun(cfg)
        if cfg.multi > 1: # added by Duncan
            for sol in opt_result.solutions:
               kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
        else:
            kidney_utils.check_validity(opt_result, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
        opt_result.formulation_name = formulation_name
        return opt_result
    else:
        raise ValueError("Unrecognised IP formulation name")

def start():
    parser = argparse.ArgumentParser("Solve a kidney-exchange instance")
    parser.add_argument("cycle_cap", type=int,
            help="The maximum permitted cycle length")
    parser.add_argument("chain_cap", type=int,
            help="The maximum permitted number of edges in a chain")
    parser.add_argument("formulation",
            help="The IP formulation (uef, eef, eef_full_red, hpief_prime, hpief_2prime, hpief_prime_full_red, hpief_2prime_full_red, picef, cf)")
    parser.add_argument("--use-relabelled", "-r", required=False,
            action="store_true",
            help="Relabel vertices in descending order of in-deg + out-deg")
    parser.add_argument("--eef-alt-constraints", "-e", required=False,
            action="store_true",
            help="Use slightly-modified EEF constraints (ignored for other formulations)")
    parser.add_argument("--timelimit", "-t", required=False, default=None,
            type=float,
            help="IP solver time limit in seconds (default: no time limit)")
    parser.add_argument("--verbose", "-v", required=False,
            action="store_true",
            help="Log Gurobi output to screen and log file")
    parser.add_argument("--edge-success-prob", "-p", required=False,
            type=float, default=1.0,
            help="Edge success probability, for failure-aware matching. " +
                 "This can only be used with PICEF and cycle formulation. (default: 1)")
    parser.add_argument("--lp-file", "-l", required=False, default=None,
            metavar='FILE',
            help="Write the IP model to FILE, then exit.")
    parser.add_argument("--relax", "-x", required=False,
            action='store_true',
            help="Solve the LP relaxation.")
    parser.add_argument("--multi", "-m", required=False, # added by Duncan
            type=int, default=1,
            help="Search for multiple solutions. (Specify the maximum number of solutions to search for, default = 1)")
    parser.add_argument("--experiment", "-z", required=False, # added by Duncan
            action='store_true', default=None,
            help="Run experiment")
    parser.add_argument("--highly_sensitized", "-H", type=percent, default=0, required=False, # added by Duncan
            help="Fraction of highly sensitized patients (on [0-1], default = 0)")
    parser.add_argument("--seed", "-s", type=str, default=1, required=False, # added by Duncan
            help="Random seed for choosing sensitized vertices")
    parser.add_argument("--fairness", "-f", type=str, default=1, required=False, # added by Duncan
            help="Output file of fairness data")
    parser.add_argument("--all_solutions", "-a", type=str, default=1, required=False, # added by Duncan
            help="Output file of all solutionss")
    parser.add_argument("--add_weight_type", "-w", type=int, default=0, required=False, # added by Duncan
            help="Add random edge weights (1 = floating point, >1 = number of weight levels")
    parser.add_argument("--pool_gap", "-g", type=int, default=0, required=False, # added by Duncan
            help="Pool gap (if >0, find suboptimal solutions)")
    parser.add_argument("--pareto", "-P", required=False, # added by Duncan
            action='store_true', default=None,
            help="Calculate Pareto curve")
    args = parser.parse_args()
    args.formulation = args.formulation.lower()

    input_lines = [line for line in sys.stdin if len(line.strip()) > 0]
    n_digraph_edges = int(input_lines[0].split()[1])
    digraph_lines = input_lines[:n_digraph_edges + 2]

    d = kidney_digraph.read_digraph(digraph_lines)

    if len(input_lines) > len(digraph_lines):
        ndd_lines = input_lines[n_digraph_edges + 2:]
        altruists = kidney_ndds.read_ndds(ndd_lines, d)
    else:
        altruists = []

    # add random weights
    # args.add_weight_type = 0 # 0 is default, read weights from file (constant, uniform for each edge)
    if args.add_weight_type == 1:  # random floating point weight between 1 and 2 for each edge
        for edge in d.es:
            edge.score = random.random() + 1.0
    elif args.add_weight_type > 1:  # random integer, either 1 or 2
        for edge in d.es:
            edge.score = random.randint(1, args.add_weight_type)

    if args.highly_sensitized > 0:
        num_sensitized = int( round(d.n * args.highly_sensitized) )
        random.seed(args.seed)
        sensitized_pairs = random.sample(range(d.n),num_sensitized)
        for i in sensitized_pairs:
            d.vs[i].sensitized = True
    else:
        num_sensitized = 0
        sensitized_pairs = []

    if args.pareto:
        print "formulation : {}".format(args.formulation)
        print "using_relabelled : {}".format(args.use_relabelled)
        print "cycle_cap : {}".format(args.cycle_cap)
        print "chain_cap : {}".format(args.chain_cap)
        print "num_pairs : {}".format(d.n)
        print "num_ndds : {}".format(len(altruists))
        print "max_core_size : {}".format(args.multi)
        print "num_sensitized : {}".format(num_sensitized)
        print "sensitized_pairs : {}".format(' '.join([str(i) for i in sensitized_pairs]))
        # 1) calculate optimal core, and all fairness values within core
        min_sens = 0
        gap = 0
        min_sens = 0
        cfg = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                  args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                  args.lp_file, args.relax, args.multi, gap, min_sens)
        core = solve_kep(cfg, args.formulation, args.use_relabelled)

        print "optimal_score : {}".format(core.total_score)

        core_fairness = [s.num_sensitized() for s in core.solutions]

        print "{:10s} {:10s}".format('num_sens','score')
        for s in core.solutions:
            print "{:10d} {:10.3f}".format(s.num_sensitized(),s.total_score)

        # 2) start with a 0 fairness bound, and increase, solving KEP for optimal solution
        # ONLY SEARCH FOR ONE SOLUTION PER FAIRNESS
        for min_sensitized in range(num_sensitized):
            cfg2 = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                       args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                       args.lp_file, args.relax, 1, 0, min_sensitized)
            sol = solve_kep(cfg2, args.formulation, args.use_relabelled)
            if sol.total_score > 0:
                print "{:10d} {:10.3f}".format(sol.num_sensitized(),sol.total_score)

    elif args.experiment:

        experiment_num = 1

        # run experiment:

        # randomly assign a percentage of pairs to be highly sensitized
        # use the filename as a seed for python.random

        if experiment_num == 1:
            #   1) find optimal solution
            min_sens=0
            cfg = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, args.pool_gap, min_sens)
            opt_solution = solve_kep(cfg, args.formulation, args.use_relabelled)

            #   2) find all solutions within 90% of optimal solution
            opt_score = opt_solution.total_score
            print "inital opt finished. OPT score = {} (max={})".format(opt_score,args.multi)
            pct = 90
            low_score = max( opt_score * pct / 100, 1)
            gap = opt_score - low_score

            print "Now finding all solutions within {}% of OPT. low score = {}".format(pct,low_score)

            cfg_2 = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, gap)
            opt_solution_2 = solve_kep(cfg_2, args.formulation, args.use_relabelled)

            opt_solution_2.write_fairness( args.fairness, args.cycle_cap, args.chain_cap, len(altruists) )
            #opt_solution_2.write_all_scores(args.all_solutions, args.cycle_cap, args.chain_cap, len(altruists))

            # print "new core size : {} (max={})".format(opt_solution.size,args.multi)
            # print "score  counts ; min_fairness max_fairness"
            # fair_by_score = opt_solution.pct_sensitized_by_score()
            # for score, counts in opt_solution.score_counter().most_common():
            #     max_fair = max(fair_by_score[score])
            #     min_fair = min(fair_by_score[score])
            #     print"{:5} {:10} ; {:6.2f} {:6.2f}".format(score,counts,min_fair,max_fair)
            #
            # fairness = opt_solution.pct_sensitized_counter()
            # print "--- fairness ---"
            # print "max percent sens: {}".format(max(fairness.keys()))
            # print "min percent sens: {}".format(min(fairness.keys()))

        elif experiment_num == 2:
            #   1) find optimal solution without constraining fairness

            min_sensitized = 0

            cfg = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, args.pool_gap, min_sensitized)
            opt_solution = solve_kep(cfg, args.formulation, args.use_relabelled)

            print "No constraint on fairness. OPT = {}".format(opt_solution.total_score)
            print "# sensitized pairs used (total) = {} ({})".format(opt_solution.num_sensitized(),num_sensitized)

            print "----------- sensitized verts -----------"
            print ' '.join([str(i) for i in sorted(sensitized_pairs)])
            print "----------- solution -----------"
            print opt_solution.display()

            # now force 3 sensitized pairs to be used

            min_sensitized = 1
            cfg2 = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, args.pool_gap, min_sensitized)
            opt_solution2 = solve_kep(cfg2, args.formulation, args.use_relabelled)

            print "Fairness constrined to {}. OPT = {}".format(min_sensitized, opt_solution2.total_score)
            print "# sensitized pairs used (total) = {} ({})".format(opt_solution2.num_sensitized(),num_sensitized)

            print "----------- sensitized verts -----------"
            print ' '.join([str(i) for i in sorted(sensitized_pairs)])
            print "----------- solution -----------"
            print opt_solution2.display()
            # now force 10 sensitized pairs to be used

            min_sensitized = 13
            cfg3 = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, args.pool_gap, min_sensitized)
            opt_solution3 = solve_kep(cfg3, args.formulation, args.use_relabelled)

            print "Fairness constrined to {}. OPT = {}".format(min_sensitized, opt_solution3.total_score)
            print "# sensitized pairs used (total) = {} ({})".format(opt_solution3.num_sensitized(),num_sensitized)

            print "----------- sensitized verts -----------"
            print ' '.join([str(i) for i in sorted(sensitized_pairs)])
            print "----------- solution -----------"
            print opt_solution3.display()

        elif experiment_num == 3:
            # 1) calculate core (optimal), and fairness values within core
            min_sens = 0
            cfg = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                      args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                      args.lp_file, args.relax, args.multi, 0, 0)
            core = solve_kep(cfg, args.formulation, args.use_relabelled)

            core_fairness = [s.num_sensitized() for s in core.solutions]
            print "optimal score: {}".format(core.total_score)
            print "fairness values (num sensitized = {}):".format(num_sensitized)
            print core_fairness

            # 2) start with a 0% fairness bound, and increase as long as KEP is feasible. for each increase in fairness,
            #    solve KEP and record optimal solution
            # ONLY SEARCH FOR ONE SOLUTION
            for min_sensitized in range(num_sensitized):
                cfg2 = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                          args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                          args.lp_file, args.relax, 1, 0, min_sensitized)
                sol = solve_kep(cfg2, args.formulation, args.use_relabelled)
                print "min_sensitized : {} score : {}".format(min_sensitized, sol.total_score)

    else:
        start_time = time.time()
        cfg = kidney_ip.OptConfig(d, altruists, args.cycle_cap, args.chain_cap, args.verbose,
                                  args.timelimit, args.edge_success_prob, args.eef_alt_constraints,
                                  args.lp_file, args.relax, args.multi)  # added multi -- Duncan
        opt_solution = solve_kep(cfg, args.formulation, args.use_relabelled)
        time_taken = time.time() - start_time
        if args.multi > 1: # added by Duncan
            print "formulation: {}".format(args.formulation)
            print "formulation_name: {}".format(opt_solution.formulation_name)
            print "using_relabelled: {}".format(args.use_relabelled)
            print "cycle_cap: {}".format(args.cycle_cap)
            print "chain_cap: {}".format(args.chain_cap)
            print "total_time: {}".format(time_taken)
            print "ip_solve_time: {}".format(opt_solution.ip_model.runtime)
            print "# pairs : {}".format(cfg.digraph.n)
            print "# NDDs : {}".format(len(cfg.ndds))
            print "total_score: {}".format(opt_solution.total_score)
            print "max core size: {}".format(args.multi)
            print "core size : {}".format(opt_solution.size)
            if args.output_core:
                opt_solution.write_vertex_participation(args.output_core, args.cycle_cap, args.chain_cap, len(altruists) )
        else:
            print "formulation: {}".format(args.formulation)
            print "formulation_name: {}".format(opt_solution.formulation_name)
            print "using_relabelled: {}".format(args.use_relabelled)
            print "cycle_cap: {}".format(args.cycle_cap)
            print "chain_cap: {}".format(args.chain_cap)
            print "edge_success_prob: {}".format(args.edge_success_prob)
            print "ip_time_limit: {}".format(args.timelimit)
            print "ip_vars: {}".format(opt_solution.ip_model.numVars)
            print "ip_constrs: {}".format(opt_solution.ip_model.numConstrs)
            print "total_time: {}".format(time_taken)
            print "ip_solve_time: {}".format(opt_solution.ip_model.runtime)
            print "solver_status: {}".format(opt_solution.ip_model.status)
            print "total_score: {}".format(opt_solution.total_score)
            opt_solution.display()

if __name__=="__main__":
    start()
