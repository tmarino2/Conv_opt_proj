import sys
from collections import defaultdict as dd
import matplotlib.pyplot as plt

D = dd(lambda : dd(list))
inner_theta = dd(lambda : dd(list))
inner_W = dd(lambda : dd(list))
inner_H = dd(lambda : dd(list))

good = set([(0.01, 0.01), (0.2, 0.5), (0.5, 0.1), (1.0, 0.5), (1., 1.), (0.01, 1.0), (1.0, 0.01)])

for L in [0.01, 0.1, 0.5, 1.]:
    for eta in [0.01, 0.1, 0.5, 1.]:

        name = "lambda={0},eta={1}.log2".format(*(L, eta)).replace("1.0.log2", "1..log2").replace("lambda=1.0,", "lambda=1.,")

        with open(name, 'rb') as f:
            for line in f:
                line = line.strip()
                try:
                    tmp, val = line.split(":")
                    tmp, val = tmp.strip(), val.strip()
                    
                    if len(tmp.split(" ")) == 2:
                        if "theta" in tmp:
                            k = int(tmp.split(" ")[1])
                            D['obj'][(L, eta)].append(val)
                    else:
                        if "theta" in tmp:
                            i, o = map(int, tmp.split(" ")[1:])
                            inner_theta[(L, eta)][o].append(val)
                        elif "W" in tmp:
                            i, o = map(int, tmp.split(" ")[1:])
                            inner_W[(L, eta)][o].append(val)
                        elif "H" in tmp:
                            i, o = map(int, tmp.split(" ")[1:])
                            inner_H[(L, eta)][o].append(val)
                except:
                    pass
                
# objective
fig, ax = plt.subplots()
for (L, eta), lst in D['obj'].items():
    if (L, eta) not in good:
        continue
    ax.plot(range(len(lst)), lst, label="(lambda={0}, eta={1})".format(*(L, eta)))
legend = ax.legend(loc='upper right', shadow=False)
plt.savefig("objective.pdf")

# theta
for (L, eta), d in inner_theta.items():
    fig, ax = plt.subplots()
    if (L, eta) not in good:
        continue

    for k, v in d.items():
        ax.plot(range(len(v)), v)

    legend = ax.legend(loc='upper right', shadow=False)
    plt.savefig("theta:lambda={0},eta{1}.pdf".format(*(L, eta)))
    
# W
for (L, eta), d in inner_W.items():
    fig, ax = plt.subplots()
    if (L, eta) not in good:
        continue

    for k, v in d.items():
        ax.plot(range(len(v)), v)

    legend = ax.legend(loc='upper right', shadow=False)
    plt.savefig("W:lambda={0},eta{1}.pdf".format(*(L, eta)))
    
# H
for (L, eta), d in inner_W.items():
    fig, ax = plt.subplots()
    if (L, eta) not in good:
        continue

    for k, v in d.items():
        ax.plot(range(len(v)), v)

    legend = ax.legend(loc='upper right', shadow=False)
    plt.savefig("H:lambda={0},eta{1}.pdf".format(*(L, eta)))
