
#Helper function to build architectures for layer/node sweeps


#couldn't get this to give me what I wanted for part d
def build_architectures(depths=(0,1,2,3), widths=(50,100,200), out_dim=1,
                        two_layer_pairs=None):
    """
    Returns dict: name -> layer_output_sizes
    d=0: [out]
    d=1: [w, out] for each w in widths
    d=2: only pairs in two_layer_pairs (if provided), else ALL ordered pairs
    """
    arch = {}

    # 0 layers
    if 0 in depths:
        arch["0_Hidden_Layers"] = [out_dim]

    # 1 layer
    if 1 in depths:
        for w in widths:
            arch[f"1_Hidden_Layer ({w})"] = [w, out_dim]

    # 2 layers
    if 2 in depths:
        if two_layer_pairs is None:
            # Fallback: all ordered pairs (50,50), (50,100), (100,50), (100,100)
            pairs = [(w1, w2) for w1 in widths for w2 in widths]
        else:
            pairs = list(two_layer_pairs)

        for w1, w2 in pairs:
            arch[f"2_Hidden_Layers ({w1}, {w2})"] = [w1, w2, out_dim]

    return arch

# Use only (50,100) for d=2:
architectures_to_sweep = build_architectures(
    depths=(0,1,2), widths=(50,100), out_dim=1,
    two_layer_pairs=[(50,100)]
)


def build_architectures_exact(out_dim=1):
    return {
        "0_Hidden_Layers":                   [out_dim],
        "1_Hidden_Layer (50)":                [50, out_dim],
        "1_Hidden_Layer (100)":               [100, out_dim],
        "2_Hidden_Layers (50, 100)":          [50, 100, out_dim],
        "2_Hidden_Layers (100, 200)":         [100, 200, out_dim],
        "3_Hidden_Layers (50, 100, 200)":     [50, 100, 200, out_dim],
    }

def build_architectures_class(out_dim=1, hidden_lists=(
    (), (50,), (100,), (50, 100), (100, 200), (50, 100, 200)
)):
    arch = {}
    for hs in hidden_lists:
        k = len(hs)
        name = f"{k}_Hidden_Layer" + ("s" if k != 1 else "")
        if hs:
            name += " (" + ", ".join(map(str, hs)) + ")"
        arch[name] = list(hs) + [out_dim]
    return arch

def build_architectures_2(depths=(1, 2, 3), widths=(32, 64, 128), out_dim=10):
    """Returns a dict mapping architecture name -> layer_output_sizes list."""
    arch = {}
    for d in depths:
        for w in widths:
            name = f"{d}L-{w}N"
            sizes = [w]*d + [out_dim]
            arch[name] = sizes
    return arch
