
#Helper function to build architectures for layer/node sweeps



def build_architectures_exact(out_dim=1):
    return {
        "0_Hidden_Layers":                   [out_dim],
        "1_Hidden_Layer (50)":                [50, out_dim],
        "1_Hidden_Layer (100)":               [100, out_dim],
        "2_Hidden_Layers (50, 100)":          [50, 100, out_dim],
        "2_Hidden_Layers (100, 200)":         [100, 200, out_dim],
        "3_Hidden_Layers (50, 100, 200)":     [50, 100, 200, out_dim],
    }


def build_architectures_from_list(hidden_layer_lists, out_dim=1, include_zero=True):
    """
    hidden_layer_lists: list of lists/tuples, e.g.
        [(50,), (100,),
         (50, 100), (100, 200),
         (50, 100, 200)]
    """
    arch = {}

    if include_zero:
        arch["0_Hidden_Layers"] = [out_dim]

    for layers in hidden_layer_lists:
        depth = len(layers)
        if depth == 1:
            name = f"1_Hidden_Layer ({layers[0]})"
        else:
            layers_str = ", ".join(str(w) for w in layers)
            name = f"{depth}_Hidden_Layers ({layers_str})"
        arch[name] = list(layers) + [out_dim]

    return arch
