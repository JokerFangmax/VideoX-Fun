import inspect

import numpy as np
import torch


def _concat_cfg_outputs(first, second):
    if isinstance(first, torch.Tensor) and isinstance(second, torch.Tensor):
        return torch.cat([first, second], dim=0)
    if isinstance(first, np.ndarray) and isinstance(second, np.ndarray):
        return np.concatenate([first, second], axis=0)
    if isinstance(first, tuple) and isinstance(second, tuple):
        if len(first) != len(second):
            raise ValueError("cfg_skip received tuple outputs with mismatched lengths")
        return tuple(
            _concat_cfg_outputs(first_item, second_item)
            for first_item, second_item in zip(first, second)
        )
    if isinstance(first, list) and isinstance(second, list):
        return first + second
    if first is None and second is None:
        return None
    raise TypeError(
        f"cfg_skip does not know how to concatenate outputs of type "
        f"{type(first).__name__} and {type(second).__name__}"
    )


def cfg_skip():
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if torch.is_grad_enabled():
                return func(self, *args, **kwargs)

            if 'hidden_states' in kwargs and kwargs['hidden_states'] is not None:
                main_input = kwargs['hidden_states']
            elif 'x' in kwargs and kwargs['x'] is not None:
                main_input = kwargs['x']
            elif len(args) > 0:
                main_input = args[0]
            else:
                raise ValueError("No input tensor found in args or kwargs")
            
            bs = len(main_input)
            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                bs_half = int(bs // 2)
                
                new_x = main_input[bs_half:]
                new_args = [
                    arg[bs_half:] if
                    isinstance(arg,
                                (torch.Tensor, list, tuple, np.ndarray)) and
                    len(arg) == bs else arg for arg in args
                ]

                new_kwargs = {
                    k: (v[bs_half:] if
                    isinstance(v,
                        (torch.Tensor, list, tuple,
                        np.ndarray)) and len(v) == bs else v
                    ) for k, v in kwargs.items()
                }
            else:
                new_x = main_input
                new_args = args
                new_kwargs = kwargs

            sig = inspect.signature(func)
            
            new_bs          = len(new_x)
            new_bs_half     = int(new_bs // 2)
            if new_bs >= 2:
                # cond
                args_i = [
                    arg[new_bs_half:] if
                    isinstance(arg,
                                (torch.Tensor, list, tuple, np.ndarray)) and
                    len(arg) == new_bs else arg for arg in new_args
                ]
                kwargs_i = {
                    k: (v[new_bs_half:] if
                    isinstance(v,
                        (torch.Tensor, list, tuple,
                        np.ndarray)) and len(v) == new_bs else v
                    ) for k, v in new_kwargs.items()
                }
                if 'cond_flag' in sig.parameters:
                    kwargs_i["cond_flag"] = True
            
                cond_out = func(self, *args_i, **kwargs_i)
                
                # uncond
                uncond_args_i = [
                    arg[:new_bs_half] if
                    isinstance(arg,
                                (torch.Tensor, list, tuple, np.ndarray)) and
                    len(arg) == new_bs else arg for arg in new_args
                ]
                uncond_kwargs_i = {
                    k: (v[:new_bs_half] if
                        isinstance(v,
                                    (torch.Tensor, list, tuple,
                                    np.ndarray)) and len(v) == new_bs else v
                        ) for k, v in new_kwargs.items()
                }
                if 'cond_flag' in sig.parameters:
                    uncond_kwargs_i["cond_flag"] = False
                uncond_out = func(self, *uncond_args_i,
                                    **uncond_kwargs_i)

                result = _concat_cfg_outputs(uncond_out, cond_out)
            else:
                result = func(self, *new_args, **new_kwargs)

            if bs >= 2 and self.cfg_skip_ratio is not None and self.current_steps >= self.num_inference_steps * (1 - self.cfg_skip_ratio):
                result = _concat_cfg_outputs(result, result)

            return result
        return wrapper
    return decorator
