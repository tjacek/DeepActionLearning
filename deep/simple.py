def build_model(params):
    print(params)
    input_shape=params["input_shape"]
    n_filters=params["num_filters"]
    filter_size2D=params["filter_size"]
    pool_size2D=params["pool_size"]
    p_drop=params["p"]
    n_cats=params['n_cats']
    n_hidden=params.get('n_hidden',32) 