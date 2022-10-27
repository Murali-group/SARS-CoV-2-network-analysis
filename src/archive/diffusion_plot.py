import matplotlib.pyplot as plt
import pandas as pd

def plot_path_based_diffusion(path_len_wise_contr_df, l):
    #l = uptill which path length to consider
    targets_with_neighbour= set(path_len_wise_contr_df[path_len_wise_contr_df['neighbour']==1]['target'])
    targets_with_no_neighbour= set(path_len_wise_contr_df['target']).difference(targets_with_neighbour)

    neighbour_pred_df = path_len_wise_contr_df[path_len_wise_contr_df['target'].isin(targets_with_neighbour)]
    no_neighbour_pred_df = path_len_wise_contr_df[path_len_wise_contr_df['target'].isin(targets_with_no_neighbour)]

    print('len neighbour_pred_df: ', len(neighbour_pred_df))
    print('len no_neighbour_pred_df: ', len(no_neighbour_pred_df))

    #consider till pathlen l
    column_names = []
    for i in range(1,l+1):
        column_names.append('frac_contr_via_pathlen_'+str(i))

    #DONOT want to sum it up though. think of another metric to visualize
    grouped_neighbour_pred_df = neighbour_pred_df.groupby('target').sum()[column_names]
    grouped_no_neighbour_pred_df = no_neighbour_pred_df.groupby('target').sum()[column_names]


    plt.boxplot(grouped_neighbour_pred_df.to_numpy(), labels=column_names)
    plt.title('predictions with seed as direct neighbour: ' + str(len(targets_with_neighbour)))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    plt.boxplot(grouped_no_neighbour_pred_df.to_numpy(), labels=column_names)
    plt.title('predictions with no seed as direct neighbour: ' + str(len(targets_with_no_neighbour)))
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

contr_file_name = '/data/tasnina/Provenance-Tracing/SARS-CoV-2-network-analysis/outputs/viz/networks/' \
                  'biogrid-y2h/GO:0062012/diffusion-path-analysis/genemaniaplus/' \
                  'orig-length_wise_contr-k500-nsp500-m20-a0.01.tsv'
path_len_wise_contr_df = pd.read_csv(contr_file_name, sep='\t')

plot_path_based_diffusion(path_len_wise_contr_df, 10)
