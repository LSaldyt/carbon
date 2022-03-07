try:
    from . import single_objective
    from . import multi_objective
except ImportError:
    import single_objective
    import multi_objective

single_obj_functions = {name : getattr(single_objective, name)
                        for name in dir(single_objective)
                        if '__' not in name and name != 'np'}
multi_obj_functions = {name : getattr(multi_objective, name)
                       for name in dir(multi_objective)
                       if '__' not in name and name != 'np'}
all_functions = single_obj_functions | multi_obj_functions
