import pandas
from kidney_digraph import Digraph
from kidney_ndds import Ndd, NddEdge

def read_CMU_format( details_filename, maxcard_filename ):
    # read details.inuput file
    col_names = ['id','abo_patient', 'abo_fonor', 'wife_patient', 'pra', 'in_deg', 'out_deg', 'is_ndd', 'is_marginalized']
    df_details = pandas.read_csv(details_filename , names = col_names, skiprows=1, delim_whitespace=True)

    pair_details = df_details.loc[df_details[ 'is_ndd' ] == 0 ]
    pair_id = list(pair_details['id'].unique())
    vtx_index = dict( zip( pair_id, range(len(pair_id) ) ) ) # vtx_index[id] gives the index in the digraph

    vtx_count = len(vtx_index)
    digraph = Digraph(vtx_count)

    # label sensitized pairs
    for index, row in pair_details.iterrows():
        if row['is_marginalized']:
          digraph.vs[ vtx_index[ row['id'] ] ].sensitized = True

    # read maxcard.inuput file (edges)
    col_names = ['src_id','tgt_id', 'weight', 'c4', 'c5']
    df_edges = pandas.read_csv(maxcard_filename , names = col_names, skiprows=1, delim_whitespace=True)
    df_edges.drop(df_edges.index[-1]) # drop the last column
    nonzero_edges = df_edges.loc[df_edges['weight'] > 0 ] # take only nonzero edges

    # ind ndds if they exist
    ndd_details = df_details.loc[df_details[ 'is_ndd' ] == 1 ]
    ndd_count = len(ndd_details)

    if ndd_count > 0:
        ndds = [Ndd() for _ in range(ndd_count)]
        ndd_id = list(ndd_details['id'].unique())
        ndd_index = dict( zip( ndd_id, range(len(ndd_id) ) ) ) # ndd_index[id] gives the index in the ndd list
    else:
        ndds = []
        ndd_index = []

    use_ndds = ndd_count > 0

    # add edges to pairs and ndds
    for index, row in nonzero_edges.iterrows():
        src = row['src_id']
        tgt_id = vtx_index[ row['tgt_id'] ]
        weight = row['weight']
        if use_ndds and ndd_index.has_key(src): # this is an ndd edge
            src_id = ndd_index[src]
            ndds[src_id].add_edge(NddEdge(digraph.vs[tgt_id], weight))
        else: # this edge is a pair edge
            src_id = vtx_index[src]
            digraph.add_edge(weight, digraph.vs[src_id], digraph.vs[tgt_id])

    return digraph, ndds