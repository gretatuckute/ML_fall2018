"""
Everything related to assiciation mining
"""

from apyori import apriori
import numpy as np


class AssociationMining:
    def __init__(self):
        print('Association mining object created')

    def mat2transactions(self, X, labels=[]):
        T = []
        for i in range(X.shape[0]):
            l = np.nonzero(X[i, :])[0].tolist()
            if labels:
                l = [labels[i] for i in l]
            T.append(l)
        return T

    def get_rules(self, t, min_support=0.8, min_confidence=1, print_rules=False):
        rules = apriori(t, min_support=min_support, min_confidence=min_confidence)

        if print_rules:
            frules = []
            for r in rules:
                conf = r.ordered_statistics[0].confidence
                supp = r.support
                x = ', '.join(list(r.ordered_statistics[0].items_base))
                y = ', '.join(list(r.ordered_statistics[0].items_add))
                print('{%s} -> {%s}  (supp: %.3f, conf: %.3f)' % (x, y, supp, conf))
        return rules

