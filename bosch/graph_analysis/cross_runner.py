import subprocess
import random
import sys

depth = int(sys.argv[1])
width = int(sys.argv[2])
keep_rate = float(sys.argv[3])
l2_reg = float(sys.argv[4])
normalizer = float(sys.argv[5])
batch_size = int(sys.argv[6])

graph = "Auto_naive_d%d_w%d_k%.1f_l%.1f_n%.1f_b%d_iall" % (
        depth, width, keep_rate, l2_reg, normalizer, batch_size)
network = ("python ../training/network.py " + 
        "--naive_loss -d %d -w %d -k %f --l2 %f -r %f %s") % (
                depth, width, keep_rate, l2_reg, normalizer, graph)
subprocess.call(["bash", "-c", network])
order = range(1, 24)
random.shuffle(order)
for i in order:
  train = ("python ../training/train.py " +
           "%s 600 -b %d -i %d --log_every_seconds 10 --seconds -v 0") % (
                   graph, batch_size, i)
  subprocess.call(["bash", "-c", train])
