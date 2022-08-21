# shapleychains lib

Official code for the paper Shapley chains: Extending Shapley values to Classifier chains

## Installation

Run the following to install

```python
pip install shapleychains

```

## Usage

```python
from shapleychains import ChainContrib
from shapleychains import get_direct_positive, get_direct_unsigned, draw_features_contribs, get_indirect_unsigned, xor

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression


df = xor()

var_x = ['X1', 'X2']

var_y = ['AND', 'OR', 'XOR']

train_xor = df.iloc[:16,:]
test_xor = df.iloc[16:,:]

chain_xor_LR = ClassifierChain(LogisticRegression())
chain_xor_LR.fit(train_xor[var_x], train_xor[var_y])

cc_LR = ChainContrib(df, var_x, var_y, chain_xor_LR.estimators_, explainer='Kernel')


dc = cc_LR.get_direct_contrib()
ic = cc_LR.get_indirect_contrib(dc)
di_pos = get_direct_positive(dc, len(var_x))
di_posneg = get_direct_unsigned(dc, len(var_x))
ic_posneg = get_indirect_unsigned(ic)

raw, normalized = draw_features_contribs(var_x, var_y, dc, ic)

```
