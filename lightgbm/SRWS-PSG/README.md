# SRWS-PSG by LightGBM

---
## Installation
```
pip install matplotlib numpy pandas tqdm joblib
pip install lightgbm sklearn scipy imblearn torch
pip install texthero transformers
```

---
## Sample
### Hyper Parameter Setup
- experiment directory name
    - e.g. EXP_try1
- dataset genre in remake_datasets function
    - comment out unused datasets
- model_variable_params, n_fold and ratio in Config class

### Execute
```
python 2_train_and_inference.py --exp-name EXP_try1
```

---

