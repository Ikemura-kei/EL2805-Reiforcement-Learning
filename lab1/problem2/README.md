# Problem 2: Mountain Car

## Running
```bash
python problem2.py
```

The outputs will be contained in `./outputs/YYYY-MM-DD HH:MM:SS`, including the trained weights, hyper-parameters, as well as plots. Also, the newest weight will be copied to `./weights.pkl` too, for fast evaluation.

## Evaluation

```bash
python check_solution.py
```

This will check the performance of the weights located at `./weights.pkl`. If you want to evaluate other weights, either copy the weights to this location, or modify the path in the evaluation script. Later I will write argument parser to pass in the weights to be evaluated.

