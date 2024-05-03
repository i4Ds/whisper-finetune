from torch.nn.utils.parametrize import is_parametrized


def disable_all_but_parametrized_grads(model):
    for _, module in model.named_modules():
        # If the module itself is not parametrized
        if not is_parametrized(module):
            # Iterate over each parameter in the module
            for param_name, param in module.named_parameters(recurse=False):
                # Additional check if the parameter itself is parametrized
                if not is_parametrized(module, param_name):
                    param.requires_grad = False
