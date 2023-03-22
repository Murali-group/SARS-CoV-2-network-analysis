import matplotlib.pyplot as plt
import numpy as np

def plot_loss_terms(loss_term1_across_param, loss_term2_across_param, xlabel, title, filename):
    loss_term1_across_param = dict(sorted(loss_term1_across_param.items()))
    loss_term2_across_param = dict(sorted(loss_term2_across_param.items()))

    params = list(loss_term1_across_param.keys())
    loss_term1 = list(loss_term1_across_param.values())
    loss_term2 = list(loss_term2_across_param.values())

    x_data = np.arange(0, len(loss_term1), 1)
    plt.plot(x_data, loss_term1, label='term1')
    plt.plot(x_data, loss_term2, label='term2')
    plt.xticks(x_data, params, rotation=90)

    # plt.plot(params, loss_term1, label='term1')
    # plt.plot(params, loss_term2, label='term2')

    plt.xlabel(xlabel)
    plt.ylabel('Quadratic loss terms')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.savefig(filename.replace('.png','.pdf'))

    plt.close()


def plot_min_diff(loss_diff_per_alg, ylabel, title, filename):
    if ylabel == 'Alpha':
        tuple_index = 0
    else:
        tuple_index = 1

    # min_param_dict contains the  alpha/beta value for for which difference in loss terms is minimum
    min_param_dict = {term: loss_diff_per_alg[term][tuple_index] for term in loss_diff_per_alg}
    # min_diff_dict ={term: loss_diff_per_alg[term][2] for term in loss_diff_per_alg}
    # min_param_dict = dict(sorted(min_param_dict.items()))
    min_param_dict = dict(sorted(min_param_dict.items(), key=lambda item: item[1]))
    terms = list(min_param_dict.keys())
    min_params = list(min_param_dict.values())

    x_data = np.arange(0, len(terms), 1)
    plt.bar(x_data, min_params)

    plt.xticks(x_data, terms, rotation=90)
    plt.xlabel('Terms')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
