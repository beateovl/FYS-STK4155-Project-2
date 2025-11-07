from Code.cost import CostOLS, dCostOLS
from Code.activations import identity, derivate
from Code.scheduler import Constant, RMS_prop, Adam
from Code.ffnn2 import NeuralNetwork as FFNN 
import itertools, pandas as pd
from Code.activations import sigmoid, RELU, LRELU


def build_nn(input_dim, layer_sizes, act_name):
    n_hidden = max(0, len(layer_sizes) - 1)
    h = activation_tests[act_name]
    activation_funcs = [h]*n_hidden + [identity]
    activation_ders  = [derivate(f) for f in activation_funcs]
    return FFNN(
        network_input_size=input_dim,
        layer_output_sizes=tuple(layer_sizes),
        activation_funcs=activation_funcs,
        activation_ders=activation_ders,
        cost_fun=CostOLS,
        cost_der=dCostOLS,
        seed=42
    )

# Create scheduler
def make_sched(opt_name, eta):
    opt_class, fixed = optimizers_to_sweep[opt_name]
    return opt_class(eta=float(eta), **fixed)

# Single training and evaluation run
def train_eval_once(opt_name, eta, Xtr, ytr, Xte, yte, input_dim, layer_sizes, act_name,
                    epochs=500, batches=32, l1=0.0, l2=0.0, lam=0.0):
    nn = build_nn(input_dim, layer_sizes, act_name)
    nn.reset_weights()
    sched = make_sched(opt_name, eta)
    _ = nn.fit(X=Xtr, t=ytr, scheduler=sched, epochs=epochs, batches=batches,
               lam=float(lam), l1=float(l1), l2=float(l2))
    ypred = nn.predict(Xte)
    return float(CostOLS(ypred, yte))



def sweep(archs, acts, opts, etas, Xtr, ytr, Xte, yte, input_dim,
          epochs=500, batches=32, l1=0.0, l2=0.0, lam=0.0):
    rows = []
    for (arch_name, layer_sizes), (act_name, _), (opt_name, _) , eta in itertools.product(
            archs.items(), acts.items(), opts.items(), etas):
        # “GD” vs “SGD”: keep same optimizer but control batches
        b = 1 if opt_name == "GD" else batches
        mse = train_eval_once(opt_name, eta, Xtr, ytr, Xte, yte,
                              input_dim=input_dim, layer_sizes=layer_sizes,
                              act_name=act_name, epochs=epochs, batches=b,
                              l1=l1, l2=l2, lam=lam)
        rows.append(dict(architecture=arch_name, activation=act_name,
                         optimizer=opt_name, eta=float(eta), l1=l1, l2=l2,
                         epochs=epochs, batches=b, mse=mse))
    return pd.DataFrame(rows).sort_values("mse").reset_index(drop=True)

def top_k(df, k=10):
    return df.nsmallest(k, "mse").reset_index(drop=True)

def best_per(df, by=("architecture","activation","optimizer")):
    return (df.sort_values("mse")
              .groupby(list(by), as_index=False)
              .first()
              .sort_values("mse")
              .reset_index(drop=True))


activation_tests = {
    'Sigmoid': sigmoid,
    'RELU': RELU,
    'LRELU': LRELU,
}

rho_val, rho2_val = 0.9, 0.999
optimizers_to_sweep = {
    'GD':       (Constant, {}),
    'SGD':      (Constant, {}),      
    'RMS_prop': (RMS_prop, {'rho': rho_val}),
    'Adam':     (Adam,     {'rho': rho_val, 'rho2': rho2_val}),
}