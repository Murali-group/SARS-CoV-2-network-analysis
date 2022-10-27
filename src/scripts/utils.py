def term_based_pred_file(pred_file, term):
    pred_file = \
        pred_file.split('.')[0] + '-' +term + '.' + pred_file.split('.')[-1]
    #also in GO term we might see ':' which we replaced by '-' while writing the pediction to file. so consider that here.
    pred_file  = pred_file.replace(':','-')
    return pred_file