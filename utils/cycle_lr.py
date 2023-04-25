import math

class OneCycle():

    def __init__(self, num_iteration, max_lr, momentum_vals=(0.95, 0.85), start_div=5, end_div=20):
        self.num_iteration = num_iteration
        self.start_div = start_div
        self.end_div = end_div
        self.high_lr = max_lr
        self.low_mom = momentum_vals[1]
        self.high_mom = momentum_vals[0]
        self.prcnt = 0
        self.iteration = 0
        self.lrs = []
        self.moms = []
        self.step_len = int(self.num_iteration / 4)

    def calc(self):
        lr = self.calc_lr_cosine()
        mom = self.calc_mom_cosine()
        self.iteration += 1
        return (lr, mom)

    def calc_lr_cosine(self):
        if self.iteration == 0:
            self.lrs.append(self.high_lr/self.start_div)
            return self.high_lr/self.start_div
        elif self.iteration == self.num_iteration:
            self.iteration = 0
            self.lrs.append(self.high_lr/self.end_div)
            return self.high_lr/self.end_div
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/(self.num_iteration - self.step_len)
            lr = (self.high_lr/self.end_div) + 0.5 * (self.high_lr - self.high_lr/self.end_div) * (1 + math.cos(math.pi * ratio))
        else:
            ratio = self.iteration/self.step_len
            lr = self.high_lr - 0.5 * (self.high_lr - self.high_lr/self.start_div) * (1 + math.cos(math.pi * ratio))
        self.lrs.append(lr)
        return lr

    def calc_mom_cosine(self):
        if self.iteration == 0:
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration == self.num_iteration:
            self.iteration = 0
            self.moms.append(self.high_mom)
            return self.high_mom
        elif self.iteration > self.step_len:
            ratio = (self.iteration -self.step_len)/(self.num_iteration - self.step_len)
            mom = self.high_mom - 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))
        else:
            ratio = self.iteration/self.step_len
            mom = self.low_mom + 0.5 * (self.high_mom - self.low_mom) * (1 + math.cos(math.pi * ratio))
        self.moms.append(mom)
        return mom
