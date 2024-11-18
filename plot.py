import matplotlib.pyplot as plt
import numpy as np
import os

class Mol:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.asarray([float(x), float(y)])
        
    def __str__(self):
        return f'{self.id} {self.pos}'

def plotMolCoo(mol: Mol, ts, outfile):
    x = []
    y = []
    
    for m in range(len(mol)):
        x.append(mol[m].pos[0])
        y.append(mol[m].pos[1])

    mark_1 = int(len(mol) / 2 + len(mol) / 8)
    mark_2 = int(len(mol) / 2 + len(mol) / 8 + 1)
    
    E = "{0:.4f}".format(totEnergy.sum1)
    Sigma_E = "{0:.4f}".format(totEnergy.sum2)
    Ek = "{0:.4f}".format(kinEnergy.sum1)
    Sigma_Ek = "{0:.4f}".format(kinEnergy.sum2)
    P_1 = "{0:.4f}".format(pressure.sum1)
    P_2 = "{0:.4f}".format(pressure.sum2)
    
    plt.plot(x, y, 'o', color='blue')
    
    plt.plot(x[mark_1], y[mark_1], 'o', color='red')
    plt.plot(x[mark_2], y[mark_2], 'o', color='cyan')
    
    plt.title('$\Delta t$:' + "{0:.4f}".format(ts) + '; ' +
                'E:' + E + '; ' +
                '$\sigma E$:' + Sigma_E + ';\n' +
                'Ek:' + Ek + '; ' +
                '$\sigma Ek$:' + Sigma_Ek + '; ' +
                'P.sum1:' + P_1 + '; ' +
                'P.sum2:' + P_2 + '; ', loc='left')
    plt.savefig(outfile)


def readOutput(path):
    n = 0
    mols = []
    with open(path, 'r') as f:
        lines = f.readlines()
        step = lines[0].split()[1]
        ts = lines[1].split()[1]
        # loop through the lines and extract the data
        n = 0
        for line in lines[3:]:
            vals = line.split()
            mol = Mol(vals[0], vals[1], vals[2])
            n += 1
            mols.append(mol)
            
    return n, mols    
            
            
            
if __name__ == '__main__':
    # read the output file
    output_dir = 'output/'
    # get the list of files in the output directory
    modes = os.listdir(output_dir)
    # loop through the files
    for mode in modes:
        sizes = os.listdir(output_dir + mode)
        for size in sizes:
            outs = os.listdir(output_dir + mode + '/' + size)
            # filter the files, only need .out files
            outs = [out for out in outs if out.endswith('.out')]
            # loop through the files
            for out in outs:
                # read the output file
                path = output_dir + mode + '/' + size + '/' + out
                readOutput(path)