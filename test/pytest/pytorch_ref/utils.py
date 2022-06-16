import torch


def torch_jacobian(func, inputs, params=None, flatten=True):
    """Calculates jacobian and return value of the given function that uses
    torch tensors.
    Args:
        func (callable): function which jacobian is calculating.
        inputs (tuple of torch tensors): function inputs by which it is
            differentiated.
        params (tuple of torch tensors, optional): function inputs by which it
            is doesn't differentiated. Defaults to None.
        flatten (bool, optional): if True then jacobian will be written in
            1D array row-major. Defaults to True.
    Returns:
        torch tensor, torch tensor: function result and function jacobian.
    """

    def recurse_backwards(output, inputs, J, flatten):
        """Recursively calls .backward on multi-dimensional output."""

        def get_grad(tensor, flatten):
            """Returns tensor gradient flatten representation. Added for
            performing concatenation of scalar tensors gradients."""

            if tensor.dim() > 0:
                if flatten:
                    return tensor.grad.flatten()
                else:
                    return tensor.grad
            else:
                return tensor.grad.view(1)

        if output.dim() > 0:
            for item in output:
                recurse_backwards(item, inputs, J, flatten)
        else:
            for inp in inputs:
                inp.grad = None

            output.backward(retain_graph=True)

            J.append(torch.cat(list(get_grad(inp, flatten) for inp in inputs)))

    if params != None:
        res = func(*inputs, *params)
    else:
        res = func(*inputs)

    J = []
    recurse_backwards(res, inputs, J, flatten)

    J = torch.stack(J)
    if flatten:
        J = J.t().flatten()

    return res, J
