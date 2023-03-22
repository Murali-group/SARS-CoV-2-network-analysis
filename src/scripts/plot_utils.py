import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statistics

def get_plot_alg_name(alg_name):
    alg_name_map = {'rwr': 'RWR', 'genemaniaplus':'RL'}
    return alg_name_map[alg_name]


def get_plot_term_name(term_name):
    term_map = {'2020-03-sarscov2-human-ppi-ace2':'Krogan'}
    return term_map[term_name] if term_name in term_map else term_name

def boxplot_dfsn_across_terms(diffusion_all_terms_dict, xlabel, ylabel,title, filename):
    '''
    diffusion_all_terms_dict: dict. A dictionary with (key,value) where key= a term, value= a list, that contains
    some measure (e.g. contribution via pathlength 1 for a target) for each target.
    filename = string. Filename to save the boxplot.
    '''
    # df = pd.DataFrame(diffusion_all_terms_dict)
    # #sort according to median values
    # sorted_index = df.median().sort_values().index
    # df = df[sorted_index]
    # sns.boxplot(data=df)

    plt.boxplot(diffusion_all_terms_dict.values(),\
                labels= list(diffusion_all_terms_dict.keys()))

    plt.ylim([0,1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(filename.replace('.pdf','.png')) #save plot in .png format as well
    # plt.show()
    plt.close()
    print('Save fig to ', filename)