# shapleychains lib

descrpetion

## Installation

Run the following to install

```python
pip install shapleychains

```

## Usage

```python
from shapleychains import ChainContrib
from shapleychains import get_all_direct, get_all_in_direct, get_n_drect, print_shapley, xor

from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression

import pandas as pd

df = xor()

df['X3'] = 0
var_x = ['X1', 'X2']
# var_x = ['X2', 'X1']

var_y = ['AND', 'OR', 'XOR']

train_xor = df.iloc[:16, :]
test_xor = df.iloc[16:, :]

chain_xor_LR = ClassifierChain(LogisticRegression())
chain_xor_LR.fit(train_xor[var_x], train_xor[var_y])

cc_LR = ChainContrib(df, var_x, var_y, chain_xor_LR.estimators_, explainer='Kernel')

predictions_proba = chain_xor_LR.predict_proba(test_xor[var_x])


dc_xor_LR = cc_LR.get_direct_contrib()
ic_xor_LR = cc_LR.get_indirect_contrib(dc_xor_LR)
n_dc_xor_LR = get_n_drect(dc_xor_LR, len(var_x))
all_dc_xor_LR = get_all_direct(dc_xor_LR, len(var_x))
all_ic_xor_LR = get_all_in_direct(ic_xor_LR)

well11, well21 = print_shapley(var_x, var_y, dc_xor_LR, ic_xor_LR)

```
