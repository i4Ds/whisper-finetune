from torch.nn.utils.parametrize import is_parametrized


def set_all_requires_grad_to_false(model):
    for p in model.parameters():
        p.requires_grad = False


def freeze_except_parametrized(model):
    for _, module in model.named_modules():
        # If the module itself is not parametrized
        if not is_parametrized(module):
            # Iterate over each parameter in the module
            for param_name, param in module.named_parameters(recurse=False):
                # Additional check if the parameter itself is parametrized
                if not is_parametrized(module, param_name):
                    param.requires_grad = False
