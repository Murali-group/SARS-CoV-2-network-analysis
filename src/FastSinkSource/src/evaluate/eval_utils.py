import os
import numpy as np
from collections import defaultdict
from scipy import sparse
import fcntl
# needed for evaluation metrics
try:
    from sklearn import metrics
except ImportError:
    print("WARNING: Unable to import sklearn")
    pass


def evaluate_ground_truth(
        run_obj, eval_ann_obj, out_file,
        non_pos_as_neg_eval=False, taxon='-',
        early_prec=None, write_prec_rec=False, 
        append=False, **kwargs):
    """
    *early_prec*: A list of recall values for which to get the precision. 
        Each will get its own column in the output file
    *write_prec_rec*: For every term, write a file containing the 
        precision and recall at every positive and negative example
    """
    term_scores, terms = run_obj.term_scores, run_obj.terms_to_run 
    eval_ann_matrix, prots = eval_ann_obj.ann_matrix, eval_ann_obj.prots
    score_term2idx = run_obj.ann_obj.term2idx
    # only evaluate the terms that are in both the term_scores matrix and the eval matrix
    terms_to_eval = set(terms) & set(eval_ann_obj.terms)
    if len(terms_to_eval) != len(terms) or len(terms_to_eval) != len(eval_ann_obj.terms):
        print("\nINFO: only %d terms both have scores (/%d terms) and are in the eval matrix (/%d terms)" % (
            len(terms_to_eval), len(terms), len(eval_ann_obj.terms)))

    print("Computing fmax from ground truth for %d terms" % (len(terms_to_eval)))
    term_stats = {}
    term_num_pos = {} 
    term_prec_rec = {}

    for term in terms_to_eval:
        eval_idx = eval_ann_obj.term2idx[term]
        # get the row corresponding to the current terms annotations 
        term_ann = eval_ann_matrix[eval_idx,:].toarray().flatten()
        positives = np.where(term_ann > 0)[0]
        # to get the scores, map the current term index to the
        # index of the term in the scores matrix
        scores = term_scores[score_term2idx[term]]
        # this is only needed for aptrank since it does not return a sparse matrix
        if sparse.issparse(scores):
            scores = scores.toarray().flatten()
        term_num_pos[term] = len(positives)
        if len(positives) == 0:
            if kwargs['verbose']:
                print("%s has 0 positives after restricting to nodes in the network. Skipping" % (term))
            continue
        if non_pos_as_neg_eval is True:
            # leave everything not a positive as a negative
            negatives = None
        else:
            # alternatively, use the negatives from that species as the set of negatives
            negatives = np.where(term_ann < 0)[0]
            if len(negatives) == 0:
                print("WARNING: 0 negatives for %s - %s. Skipping" % (term, taxon))
                continue
        prec, recall, fpr, pos_neg_stats = compute_eval_measures(
                scores, positives, negatives=negatives, 
                track_pos=True, track_neg=True)
        # Takes too much RAM to store these values for all terms 
        # so only store them when they will be written to a file
        if write_prec_rec:
            term_prec_rec[term] = (prec, recall, pos_neg_stats)
        fmax = compute_fmax(prec, recall)
        avgp = compute_avgp(prec, recall)
        if len(prec) == 1:
            auprc = 0
            auroc = 0
        else:
            auprc = compute_auprc(prec, recall)
            auroc = compute_auroc([r for r, f in fpr], [f for r, f in fpr])
        eprec_vals = []
        if early_prec is not None:
            # compute the early precision at specified values
            eprec_vals = compute_early_prec(
                prec, recall, pos_neg_stats, early_prec, term_num_pos[term])
        term_stats[term] = (fmax, avgp, auprc, auroc, eprec_vals)
        if kwargs['verbose']:
            print("%s fmax: %0.4f" % (term, fmax))

    # skip writing the output file if there's only one term specified
    if write_prec_rec and len(term_prec_rec) == 1:
        print("skipping writing %s" % (out_file))
    else:
        out_str = ""
        # sort by # ann per term
        for g in sorted(term_stats, key=term_num_pos.get, reverse=True):
            fmax, avgp, auprc, auroc, eprec_vals = term_stats[g]
            # format the values so they'll be ready to be written to the output file
            early_prec_str = '\t'+'\t'.join("%0.4f" % (p) for p in eprec_vals) \
                             if len(eprec_vals) > 0 else ""
            out_str += "%s%s\t%0.4f\t%0.4f\t%0.4f\t%0.4f\t%d%s\n" % (
                "%s\t"%taxon if taxon not in ["-", None] else "",
                g, fmax, avgp, auprc, auroc, term_num_pos[g], early_prec_str)
        # don't re-write the header if this file is being appended to
        if not os.path.isfile(out_file) or not append:
            print("Writing results to %s\n" % (out_file))
            header_line = "#term\tfmax\tavgp\tauprc\tauroc"
            if taxon not in ['-', None]:
                header_line = "#taxon\t%s\t# test ann" % (header_line)
            else:
                header_line += "\t# ann"
            if early_prec is not None:
                header_line += '\t'+'\t'.join(["eprec-rec%s" % (r) for r in early_prec])
            out_str = header_line+"\n" + out_str
        else:
            print("Appending results to %s\n" % (out_file))
        with open(out_file, 'a' if append else 'w') as out:
            # lock it to avoid scripts trying to write at the same time
            fcntl.flock(out, fcntl.LOCK_EX)
            out.write(out_str)
            fcntl.flock(out, fcntl.LOCK_UN)

    if write_prec_rec:
        term = list(term_prec_rec.keys())[0]
        out_file_pr = out_file.replace('.txt', "prec-rec%s%s.txt" % (
            taxon if taxon not in ['-', None] else '',
            '-%s'%(term) if len(term_prec_rec) == 1 else ""))
        print("writing prec/rec to %s" % (out_file_pr))
        with open(out_file_pr, 'w') as out:
            out.write("#term\tprec\trec\tnode\tscore\tidx\tpos/neg\n")
            #for term, (prec, rec, pos_neg_stats) in sorted(term_prec_rec.items(), key=term_num_pos.get, reverse=True):
            for term, (prec, rec, pos_neg_stats) in term_prec_rec.items():
                out.write(''.join(["%s\t%0.4f\t%0.4f\t%s\t%0.4e\t%d\t%d\n" % (
                    term, p, r, prots[n], s, idx, pos_neg) for p,r,(n,s,idx,pos_neg,_) in zip(prec, rec, pos_neg_stats)]))


def compute_eval_measures(scores, positives, negatives=None, 
        track_pos=False, track_neg=False):
    """
    Compute the precision and false-positive rate at each change in recall (true-positive rate)
    *scores*: array containing a score for each node
    *positives*: indices of positive nodes
    *negatives*: if negatives are given, then the FP will only be from the set of negatives given
    *track_pos*: if specified, track the score and rank of the positive nodes,
        and return a tuple of the node ids in order of their score, their score, their idx, and 1/-1 for pos/neg
    *track_neg*: also track the score and rank of the negative nodes
    """
    #f1_score = metrics.f1score(positives, 
    #num_unknowns = len(scores) - len(positives) 
    positives = set(positives)
    check_negatives = False
    if negatives is not None:
        check_negatives = True 
        negatives = set(negatives)
    else:
        print("TODO. Treating all non-positives as negatives not yet implemented.")
    # compute the precision and recall at each change in recall
    # use np.argsort
    nodes_sorted_by_scores = np.argsort(scores)[::-1]
    precision = []
    recall = []
    fpr = []
    pos_neg_stats = []  # tuple containing the node, score, idx, pos/neg assign., and the # positives assigned so far
    # TP is the # of correctly predicted positives so far
    TP = 0
    FP = 0
    rec = 0
    for i, n in enumerate(nodes_sorted_by_scores):
        # TODO this could be slow if there are many positives
        if n in positives:
            TP += 1
            # precisions is the # of true positives / # true positives + # of false positives (or the total # of predictions)
            precision.append(TP / float(TP + FP))
            # recall is the # of recovered positives TP / TP + FN (total # of positives)
            rec = TP / float(len(positives))
            recall.append(rec)
            # fpr is the FP / FP + TN
            fpr.append((rec, FP / float(len(negatives))))
            if track_pos:
                pos_neg_stats.append((n, scores[n], i, 1, TP+FP)) 
        elif check_negatives is False or n in negatives:
            FP += 1
            # store the prec and rec even though recall doesn't change since the AUPRC will be affected
            precision.append(TP / float(TP + FP))
            recall.append(TP / float(len(positives)))
            fpr.append((rec, FP / float(len(negatives))))
            if track_neg:
                pos_neg_stats.append((n, scores[n], i, -1, TP+FP)) 
        #else:
        #    continue

    # TODO shouldn't happen
    if len(precision) == 0:
        precision.append(0)
        recall.append(1)

    #print(precision[0], recall[0], fpr[0])

    if track_pos or track_neg:
        return precision, recall, fpr, pos_neg_stats
    else:
        return precision, recall, fpr


def compute_fmax(prec, rec, fmax_idx=False):
    """
    *fmax_idx*: also return the index at which the harmonic mean is the greatest
    """
    f_measures = []
    for i in range(len(prec)):
        p, r = prec[i], rec[i]
        if p+r == 0:
            harmonic_mean = 0
        else:
            # see https://en.wikipedia.org/wiki/Harmonic_mean#Two_numbers
            harmonic_mean = (2*p*r)/(p+r)
        f_measures.append(harmonic_mean)
    if fmax_idx:
        idx = np.argmax(np.asarray(f_measures))
        return max(f_measures), idx
    else:
        return max(f_measures)


def compute_avgp(prec, rec):
    # average precision score
    # see http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    avgp = 0
    prev_r = 0 
    for p,r in zip(prec, rec):
        recall_change = r - prev_r
        avgp += (recall_change*p)
        prev_r = r
    #avgp = avgp / float(len(alg_prec_rec))
    return avgp


def compute_auprc(prec, rec):
    auprc = metrics.auc(rec, prec)
    return auprc


def compute_auroc(tpr, fpr):
    auroc = metrics.auc(fpr, tpr)
    return auroc


def compute_early_prec(prec, rec, pos_neg_stats, recall_vals, num_pos):
    early_prec_values = []
    for curr_recall in recall_vals:
        # if a k recall is specified, get the precision at the recall which is k * # ann in the left-out species
        if 'k' in curr_recall:
            k_val = float(curr_recall.replace('k',''))
            num_nodes_to_get = k_val * num_pos 
            # the pos_neg_stats tracks the prec for every node.
            # So find the precision value for first node with an index >= (k * # ann)
            for idx, (_, _, _, _, curr_num_pos) in enumerate(pos_neg_stats): 
                if curr_num_pos >= num_nodes_to_get:
                    break
            # get the precision at the corresponding index
            p = prec[idx]
            early_prec_values.append(p)
        else:
            curr_recall = float(curr_recall)
            # find the first precision value where the recall is >= the specified recall
            for p, r in zip(prec, rec):
                if r >= curr_recall:
                    early_prec_values.append(p)
                    break
    return early_prec_values


def sample_neg_examples(ann_obj, sample_neg_examples_factor=10):
    """
    For each term, sample negative examples from all prots (minus the pos examples)
    """
    print("Sampling negative examples using a factor of %d * # pos per term from the universe of %s prots" % (
        sample_neg_examples_factor, len(ann_obj.prots)))
    pos_mat = (ann_obj.ann_matrix > 0).astype(int)
    new_ann_mat = pos_mat.copy()
    # the universe is the indices of the prots 
    prot_universe = set(np.arange(len(ann_obj.prots)))
    for idx, term in enumerate(ann_obj.terms):
        pos_examples = np.where(pos_mat[idx].toarray().flatten() > 0)[0]
        neg_examples = sample_term_neg_examples(set(pos_examples), prot_universe, sample_neg_examples_factor)
        # make sure they are treated as indexes
        neg_examples = neg_examples.astype(int)
        neg_vec = np.zeros(pos_mat.shape[1])
        neg_vec[neg_examples] = -1
        new_ann_mat[idx] += neg_vec
    return new_ann_mat


def sample_term_neg_examples(pos_examples, prot_universe, sample_neg_examples_factor):
    """
    *pos_examples*: indices of prots
    """
    neg_factor = sample_neg_examples_factor * len(pos_examples) 
    non_pos_universe = prot_universe - pos_examples
    #print("Sampling %s (%s*%s) negative examples from the universe of %s non-pos prots" % (
    #    neg_sample, sample_neg_examples_factor, len(pos_examples), len(non_pos_universe)))
    if neg_factor > len(non_pos_universe):
        print("Sampling %s (%s*%s) negative examples from the universe of %s non-pos prots" % (
            neg_factor, sample_neg_examples_factor, len(pos_examples), len(non_pos_universe)))
        print("ERROR: cannot sample more negative examples than specified by non_pos_universe")

    # now perform the sampling
    non_pos_universe = np.asarray(list(non_pos_universe), dtype=str)
    # sample without replacement
    neg_examples = np.random.choice(non_pos_universe, size=int(neg_factor), replace=False)
    return neg_examples
