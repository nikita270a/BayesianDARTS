""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import torch
from torch.autograd import Variable


class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        loss = self.net.loss(trn_X, trn_y) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.weights())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
                m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
                vw.copy_(w - xi * (m + g + self.w_weight_decay*w))

            # synchronize alphas
            for a, va in zip(self.net.alphas(), self.v_net.alphas()):
                va.copy_(a)

    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        # do virtual step (calc w`)
        self.virtual_step(trn_X, trn_y, xi, w_optim)

        # calc unrolled loss
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)

        # compute gradient
        v_alphas = tuple(self.v_net.alphas())
        v_weights = tuple(self.v_net.weights())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, trn_X, trn_y)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad = da - xi*h

    def compute_hessian(self, dw, trn_X, trn_y):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_pos = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p -= 2. * eps * d
        loss = self.net.loss(trn_X, trn_y)
        dalpha_neg = torch.autograd.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.weights(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
    
    
    def compute_Hw(self, input_valid, target_valid):
        self.zero_grads(self.net.weights())
        self.zero_grads(self.net.alphas())
        #if unrolled:
        #    self._backward_step_unrolled(input_train, target_train,
        #                                 input_valid, target_valid, eta,
        #                                 network_optimizer, True)
        #else:
        #    self._backward_step(input_valid, target_valid, True)

        #self.grads = [v.grad + self.weight_decay*v for v in self.model.arch_parameters()]
        #loss = self.model._loss(input_valid, target_valid)
        loss = self.net.loss(input_valid, target_valid)

        self.hessian = self._hessian(loss, self.net.alphas())#self.model.arch_parameters())
        return self.hessian
    
    
    def _hessian(self, outputs, inputs, out=None, allow_unused=False,
                 create_graph=False):
        #assert outputs.data.ndimension() == 1

        if torch.is_tensor(inputs):
            inputs = [inputs]
        else:
            inputs = list(inputs)

        n = sum(p.numel() for p in inputs)
        print(f'SIZE OF N: {n}')
        if out is None:
            out = Variable(torch.zeros(n, n)).type_as(outputs)

        ai = 0
        for i, inp in enumerate(inputs):
            [grad] = torch.autograd.grad(outputs, inp, create_graph=True,
                                         allow_unused=allow_unused)
            grad = grad.contiguous().view(-1) + self.w_weight_decay*inp.view(-1)
            #grad = outputs[i].contiguous().view(-1)
            #print(f'{i}: ')
            for j in range(inp.numel()):
                # print('(i, j): ', i, j)
                if grad[j].requires_grad:
                    row = self.gradient(grad[j], inputs[i:], retain_graph=True)[j:]
                else:
                    n = sum(x.numel() for x in inputs[i:]) - j
                    row = Variable(torch.zeros(n)).type_as(grad[j])
                    #row = grad[j].new_zeros(sum(x.numel() for x in inputs[i:]) - j)

                out.data[ai, ai:].add_(row.clone().type_as(out).data)  # ai's row
                if ai + 1 < n:
                    out.data[ai + 1:, ai].add_(row.clone().type_as(out).data[1:])  # ai's column
                del row
                ai += 1
            del grad
        return out
    
   
    def zero_grads(self, parameters):
        for p in parameters:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
                #if p.grad.volatile:
                #    p.grad.data.zero_()
                #else:
                #    data = p.grad.data
                #    p.grad = Variable(data.new().resize_as_(data).zero_())

    def gradient(self, _outputs, _inputs, grad_outputs=None, retain_graph=None,
                create_graph=False):
        if torch.is_tensor(_inputs):
            _inputs = [_inputs]
        else:
            _inputs = list(_inputs)
        grads = torch.autograd.grad(_outputs, _inputs, grad_outputs,
                                    allow_unused=True,
                                    retain_graph=retain_graph,
                                    create_graph=create_graph)
        grads = [x if x is not None else torch.zeros_like(y) for x, y in zip(grads,
                                                                             _inputs)]
        return torch.cat([x.contiguous().view(-1) for x in grads])
