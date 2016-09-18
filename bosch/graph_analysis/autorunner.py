import subprocess
import random

loss_flags = ['fraction_loss', 'naive_loss']
depths = [2, 3, 4, 5]
widths = [256, 512, 1024, 2048]
keep_rates = [1, 0.9, 0.8, 0.7]
l2_regs = [0, 0, 0, 0.1, 10]
normalizers = [0, 0, 0, 0.1, 10]
batch_sizes = [400, 1000, 4000, 10000]

loss_flag = random.choice(loss_flags)
depth = random.choice(depths)
width = random.choice(widths)
keep_rate = random.choice(keep_rates)
l2_reg = random.choice(l2_regs)
normalizer = random.choice(normalizers)
batch_size = random.choice(batch_sizes)
inp = random.choice(range(1, 24))

graph = "Auto_%s_d%d_w%d_k%.1f_l%.1f_n%.1f_b%d_i%d" % (
        loss_flag, depth, width, keep_rate, l2_reg, normalizer, batch_size, inp)
network = ("python ../training/network.py " + 
        "--%s -d %d -w %d -k %f --l2 %f -r %f %s") % (
                loss_flag, depth, width, keep_rate, l2_reg, normalizer, graph)
train = ("python ../training/train.py " +
        "%s 600 -b %d -i %d --log_every_seconds 10 --seconds -v 0") % (
                graph, batch_size, inp)

subprocess.call(["bash", "-c", network])
subprocess.call(["bash", "-c", train])
