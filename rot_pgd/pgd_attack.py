import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True, attack_rotations=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.attack_rotations = attack_rotations

    def forward(self, model, bx, by, by_prime, curr_batch_size,data_parallel = False):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits, pen = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits[:curr_batch_size], by, reduction='sum')
                if self.attack_rotations:
                    if data_parallel:
                        loss += F.cross_entropy(model.module.rot_pred(pen[curr_batch_size:]), by_prime, reduction='sum')
                    else:
                        loss += F.cross_entropy(model.rot_pred(pen[curr_batch_size:]), by_prime, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx
