import sys
from collections import defaultdict as dd
import matplotlib.pyplot as plt

D = dd(lambda : dd(list))

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
                except:
                    pass
                
# objective
fig, ax = plt.subplots()

for (L, eta), lst in D['obj'].items():
    if (L, eta) not in good:
        continue
    #print L, eta
    #print lst

    ax.plot(range(len(lst)), lst, label="(lambda={0}, eta={1})".format(*(L, eta)))
legend = ax.legend(loc='upper right', shadow=False)
plt.show()
