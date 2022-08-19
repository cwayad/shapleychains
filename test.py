from shapleychains import ChainContrib
from shapleychains import get_di_posneg, get_in_posneg, get_di_pos, print_feat_contribs, xor

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

import pandas as pd

df = xor()

var_x = ['X1', 'X2']

var_y = ['AND', 'OR', 'XOR']

train_xor = df.iloc[:16, :]
test_xor = df.iloc[16:, :]

chain_xor_LR = ClassifierChain(LogisticRegression())
chain_xor_LR.fit(train_xor[var_x], train_xor[var_y])

cc_LR = ChainContrib(df, var_x, var_y, chain_xor_LR.estimators_, explainer='Kernel')

#predictions_proba = chain_xor_LR.predict_proba(test_xor[var_x])


dc = cc_LR.get_direct_contrib()
ic = cc_LR.get_indirect_contrib(dc)
di_pos = get_di_pos(dc, len(var_x))
di_posneg = get_di_posneg(dc, len(var_x))
ic_posneg = get_in_posneg(ic)

raw, normalized = print_feat_contribs(var_x, var_y, dc, ic)