"""
Fairness experiments, varying 
- edge success probability
- chain cap
- fairness rule
"""

import os

import kidney_digraph
import kidney_ip
import kidney_utils
import kidney_ndds
from read_CMU_format import read_CMU_format

import argparse
import glob
import numpy

def solve_kep(cfg, formulation, use_relabelled=True):
    formulations = {
        "uef": ("Uncapped edge formulation", kidney_ip.optimise_uuef),
        "eef": ("EEF", kidney_ip.optimise_eef),
        "eef_full_red": ("EEF with full reduction by cycle generation", kidney_ip.optimise_eef_full_red),
        "hpief_prime": ("HPIEF'", kidney_ip.optimise_hpief_prime),
        "hpief_prime_full_red": (
        "HPIEF' with full reduction by cycle generation", kidney_ip.optimise_hpief_prime_full_red),
        "hpief_2prime": ("HPIEF''", kidney_ip.optimise_hpief_2prime),
        "hpief_2prime_full_red": (
        "HPIEF'' with full reduction by cycle generation", kidney_ip.optimise_hpief_2prime_full_red),
        "picef": ("PICEF", kidney_ip.optimise_picef),
        "cf": ("Cycle formulation",
               kidney_ip.optimise_ccf)
    }

    if formulation in formulations:
        formulation_name, formulation_fun = formulations[formulation]
        if use_relabelled:
            opt_result = kidney_ip.optimise_relabelled(formulation_fun, cfg)
        else:
            opt_result = formulation_fun(cfg)
        if cfg.multi > 1:  # added by Duncan
            for sol in opt_result.solutions:
                kidney_utils.check_validity(sol, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
        else:
            kidney_utils.check_validity(opt_result, cfg.digraph, cfg.ndds, cfg.max_cycle, cfg.max_chain)
        opt_result.formulation_name = formulation_name
        return opt_result
    else:
        raise ValueError("Unrecognised IP formulation name")


def start():

    parser = argparse.ArgumentParser("Run Kidney Exchange Experiments")
    parser.add_argument("config_file", help="The experiment configuration file containing: input file type, input path, output file path, beta list , weighted or alpha-lex, chain cap list, and edge success prob. list.")
    args = parser.parse_args()

    config = {}
    # execfile(args.config_file, config)
    exec(open(args.config_file).read(), config)

    print("config file: {}".format(args.config_file))
    # print(config)
    for k,v in zip(config.keys(),config.values()):
        if not k.startswith('_'):
            print("{} : {}".format(k,v))

    # experiment parameters

    input_graph_type = config['input_graph_type'] # either 'KPD' or CMU'

    # input directory containing either:
    # - (input_graph_type == KPD): ONLY directories containing unzipped KPD files ('*edgeweights.csv' and '*recipient.csv') or
    # - (input_graph_type == CMU): random graphs in CMU format ('*maxcard.input' and '*details.input')
    input_dir = config['input_dir']
    output_file = config['output_file']

    chain_cap_list = config['chain_cap_list'] # [0,5,10,20]
    chain_cap_list.sort()

    edge_prob_list = config['edge_prob_list'] # [0.2, 0.5, 0.8, 1.0]
    edge_prob_list.sort()
    fairness_rule = config['fairness_rule'] # "weighted" or "alpha_lex"
    if fairness_rule == 'weighted':
        beta_list = numpy.array(config['beta_list'])  # list of beta values (>=0), ex: numpy.arange(0, 30.5, 0.5)
        beta_list.sort()
    if fairness_rule == 'alpha_lex':
        alpha_list = numpy.array(config['alpha_list']) # list of alpha values (0<=alpha<=1), ex: [0.0,0.1,0.5,0.9,1.0]
        alpha_list.sort()

    ## for KPD file format

    #input_dir = '/Users/duncan/Google Drive/research/kpd_zips'
    # output_file = '/Users/duncan/Google Drive/research/fairness_results/failure_aware/lex_fairness_result_2.txt'

    if input_graph_type == 'KPD':

        print_header(output_file)
        kpd_dirs = glob.glob(input_dir+os.sep+'*/')

        for dir in kpd_dirs[:2]:

            dirname = dir.split('/')[-2] # os.path.basename(dir)

            edges_filename = glob.glob(dir+os.sep+'*edgeweights.csv')[0]
            recipient_filename = glob.glob(dir+os.sep+'*recipient.csv')[0]

            # read KPD graph
            d,vtx_index  = kidney_digraph.read_from_kpd(edges_filename)

            # add highly sensitized patients
            d.KPD_label_sensitized(recipient_filename, vtx_index)

            # num_sensitized = d.get_num_sensitized()

            # read altruists
            altruists,ndd_index = kidney_ndds.read_from_kpd(edges_filename, d, vtx_index)

            if fairness_rule == 'weighted':
                weighted_fairness_experiment(beta_list,d, altruists, dirname, output_file, chain_cap_list, edge_prob_list)
            if fairness_rule == 'alpha_lex':
                alpha_lex_experiment(d, altruists, dirname, output_file, alpha_list, chain_cap_list, edge_prob_list)

    ## for random graph format
    if input_graph_type == 'CMU':

        # input_dir = '/Users/duncan/Google Drive/research/graphs/graphs_from_john/graphs_128'
        # output_file = '/Users/duncan/Google Drive/research/fairness_results/failure_aware/lex_fairness_result_128_rev.txt'
        print_header(output_file)

        maxcard_files = glob.glob(input_dir+os.sep+'*maxcard.input')
        for maxcard_filename in maxcard_files[:2]: # ['/Users/duncan/Google Drive/research/graphs/graphs_from_john/graphs_64/unos_bimodal_apd_v64_i3_maxcard.input']

            file_base = '_'.join(maxcard_filename.split('_')[:-1])
            dirname = maxcard_filename.split('/')[-1]

            details_files = glob.glob(file_base + '_*details.input')

            if len(details_files)>0:
                details_filename = details_files[0]

                d, altruists = read_CMU_format( details_filename, maxcard_filename )

                #weighted_fairness_experiment(beta_list,d, altruists, dirname, outfile, chain_cap_list, edge_prob_list)
                # alpha_lex_experiment(d, altruists, dirname, output_file, chain_cap_list, edge_prob_list)

                if fairness_rule == 'weighted':
                    weighted_fairness_experiment(beta_list,d, altruists, dirname, output_file, chain_cap_list, edge_prob_list)
                if fairness_rule == 'alpha_lex':
                    alpha_lex_experiment(d, altruists, dirname, output_file, alpha_list, chain_cap_list, edge_prob_list)
            else:
                print("could not find *details.input file for: {}\n".format(maxcard_filename))


def alpha_lex_experiment(digraph,altruists,dirname, outfile, alpha_list, chain_cap_list, edge_prob_list=[1.0]):

    alpha_list.sort()

    cycle_cap = 3
    # chain_cap = 4
    verbose = False
    timelimit = None
    #edge_success_prob = edge_prob
    eef_alt_constraints = None
    lp_file = None
    relax = None
    formulation = 'picef'
    use_relabelled = False
    multi = 1
    gap = 0
    min_sensitized = 0
    beta = 0

    # find maximum number of sensitized patients possible
    # copy KPD graph with nonzero weights only for highly sensitized patientss
    d_fair = digraph.fair_copy()
    a_fair = [a.fair_copy() for a in altruists]

    # graph parameters
    total_num_sensitized = digraph.get_num_sensitized()
    total_num_pairs = digraph.get_num_pairs()
    num_ndds = len(altruists)

    with open(outfile, 'a') as csvfile:

        for chain_cap in chain_cap_list:
            for edge_success_prob in edge_prob_list:
                fair_cfg = kidney_ip.OptConfig(d_fair, a_fair, cycle_cap, chain_cap, verbose,
                                               timelimit, edge_success_prob, eef_alt_constraints,
                                               lp_file, relax, multi, gap, min_sensitized)
                fair_sol = solve_kep(fair_cfg, formulation, use_relabelled)

                max_fair_score = fair_sol.total_score
                max_sens_matched = fair_sol.num_sensitized()
                # first solve original problem (alpha = 0)
                min_fair_score = 0
                cfg = kidney_ip.OptConfig(digraph, altruists, cycle_cap, chain_cap, verbose,
                                          timelimit, edge_success_prob, eef_alt_constraints,
                                          lp_file, relax, multi, gap, min_fair_score)
                sol = solve_kep(cfg, formulation, use_relabelled)

                fair_score = sol.get_fair_score(altruists)
                num_matched = sol.num_matched()
                num_sensitized = sol.num_sensitized()
                # csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                    # dirname,
                    # cycle_cap,
                    # chain_cap,
                    # total_num_pairs,
                    # num_ndds,
                    # total_num_sensitized,
                    # max_fair_score,
                    # beta,
                    # min_fair_score,
                    # sol.total_score,
                    # fair_score,
                    # num_matched,
                    # num_sensitized,
                    # edge_success_prob))

                for alpha in alpha_list:
                    min_fair_score = alpha * max_fair_score  # numpy.linspace(0,1,11) * max_fair_score

                    # if the previous solution already satisfies this minimum score, just write it again
                    if fair_score >= min_fair_score:
                        csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                                    dirname,
                                    cycle_cap,
                                    chain_cap,
                                    total_num_pairs,
                                    num_ndds,
                                    total_num_sensitized,
                                    max_fair_score,
                                    max_sens_matched,
                                    beta,
                                    alpha,
                                    sol.total_score,
                                    fair_score,
                                    num_matched,
                                    num_sensitized,
                                    edge_success_prob))
                    # if the previous solution doesn't meet the alpha criteria, solve again with the new min fair score
                    else:
                        cfg = kidney_ip.OptConfig(digraph, altruists, cycle_cap, chain_cap, verbose,
                                                   timelimit, edge_success_prob, eef_alt_constraints,
                                                   lp_file, relax, multi, gap, min_fair_score)
                        sol = solve_kep(cfg, formulation, use_relabelled)

                        fair_score = sol.get_fair_score(altruists)
                        num_matched = sol.num_matched()
                        num_sensitized = sol.num_sensitized()
                        csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                            dirname,
                            cycle_cap,
                            chain_cap,
                            total_num_pairs,
                            num_ndds,
                            total_num_sensitized,
                            max_fair_score,
                            max_sens_matched,
                            beta,
                            alpha,
                            sol.total_score,
                            fair_score,
                            num_matched,
                            num_sensitized,
                            edge_success_prob))

def weighted_fairness_experiment(beta_list,digraph,altruists,dirname,outfile,chain_cap_list,edge_prob_list=[1.0]):

    beta_list.sort()

    cycle_cap = 3
    # chain_cap = 3
    verbose = False
    timelimit = None
    #edge_success_prob = edge_prob
    eef_alt_constraints = None
    lp_file = None
    relax = None
    formulation = 'picef'
    use_relabelled = False
    multi = 1
    gap = 0
    alpha = 0
    min_sensitized = 0

    total_score = numpy.zeros(len(beta_list))
    fair_score = numpy.zeros(len(beta_list))
    num_matched = numpy.zeros(len(beta_list))
    sens_matched = numpy.zeros(len(beta_list))

    total_num_sensitized = digraph.get_num_sensitized()
    total_num_pairs = digraph.get_num_pairs()
    num_ndds = len(altruists)

    # find maximum number of sensitized patients possible
    # copy KPD graph with nonzero weights only for highly sensitized patientss
    d_fair = digraph.fair_copy()
    a_fair = [a.fair_copy() for a in altruists]

    # print("maximum possible fairness: {} (total matched = {})".format(max_sens, fair_sol.num_matched()))

    for chain_cap in chain_cap_list:
        for edge_success_prob in edge_prob_list:

            fair_cfg = kidney_ip.OptConfig(d_fair, a_fair, cycle_cap, chain_cap, verbose,
                                           timelimit, edge_success_prob, eef_alt_constraints,
                                           lp_file, relax, multi, gap, min_sensitized)
            fair_sol = solve_kep(fair_cfg, formulation, use_relabelled)

            max_fair_score = fair_sol.total_score
            max_sens_matched = fair_sol.num_sensitized()

            # these values should never be used
            current_total_score = -1
            current_fair_score = -1
            current_num_matched = -1
            current_sens_matched = -1

            for i,beta in enumerate(beta_list):

                # if we've already hit the maximum score...
                if current_fair_score == max_fair_score:
                    total_score[i] = current_total_score
                    fair_score[i] = current_fair_score
                    num_matched[i] = current_num_matched
                    sens_matched[i] = current_sens_matched
                else:

                    digraph.augment_weights(beta)
                    for a in altruists:
                        a.augment_weights(beta)

                    cfg = kidney_ip.OptConfig(digraph, altruists, cycle_cap, chain_cap, verbose,
                                               timelimit, edge_success_prob, eef_alt_constraints,
                                               lp_file, relax, multi, gap, min_sensitized)
                    sol = solve_kep(cfg, formulation, use_relabelled)

                    digraph.unaugment_weights(beta)
                    for a in altruists:
                        a.unaugment_weights(beta)

                    # get the total scores (unaugment the weights first, and recalculate the score...)
                    current_total_score = sol.get_score(digraph, altruists, edge_success_prob)
                    current_fair_score = sol.get_score(d_fair, a_fair, edge_success_prob)
                    current_num_matched = sol.num_matched()
                    current_sens_matched = sol.num_sensitized()

                    total_score[i] = current_total_score
                    fair_score[i] = current_fair_score
                    num_matched[i] = current_num_matched
                    sens_matched[i] = current_sens_matched

            with open(outfile,'a') as csvfile:
                for i,beta in enumerate(beta_list): # enumerate(beta_list[0:max_i+1]):
                    csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        dirname,
                        cycle_cap,
                        chain_cap,
                        total_num_pairs,
                        num_ndds,
                        total_num_sensitized,
                        max_fair_score,
                        max_sens_matched,
                        beta,
                        alpha,
                        total_score[i],
                        fair_score[i],
                        num_matched[i],
                        sens_matched[i],
                        edge_success_prob))

def find_optimal_fairness(digraph,altruists,dirname):
    # copy KPD graph with only weights for sensitized patients
    d_fair = digraph.fair_copy()
    a_fair = [a.fair_copy() for a in altruists]

def print_header(outfile):
    with open(outfile,'w') as csvfile:
        colnames = ['kpd_dirname',
                    'cycle_cap',
                    'chain_cap',
                    'num_pairs',
                    'num_ndds',
                    'num_sens',
                    'max_fair_score',
                    'max_num_sens_matched',
                    'beta',
                    'alpha',
                    'total_score',
                    'fair_score',
                    'num_pairs_matched',
                    'num_sens_matched',
                    'edge_success_prob']
        csvfile.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(*colnames))



if __name__ == "__main__":
    start()
