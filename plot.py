import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

class Mol:
    def __init__(self, id, x, y):
        self.id = id
        self.pos = np.asarray([float(x), float(y)])
        
    def __str__(self):
        return f'{self.id} {self.pos}'

class Stats:
    def __init__(self, e1, e2, ke1, ke2, p1, p2, ts, step):
        self.e1 = e1
        self.e2 = e2
        self.ke1 = ke1
        self.ke2 = ke2
        self.p1 = p1
        self.p2 = p2
        self.ts = ts
        self.step = step
        
    def __str__(self):
        return f' {self.step} {self.ts} {self.e1} {self.e2} {self.ke1} {self.ke2} {self.p1} {self.p2}'

def plotMolCoo(mol: Mol, ts, outfile, stats: Stats):
    x = []
    y = []
    
    for m in range(len(mol)):
        x.append(mol[m].pos[0])
        y.append(mol[m].pos[1])

    mark_1 = int(len(mol) / 2 + len(mol) / 8)
    mark_2 = int(len(mol) / 2 + len(mol) / 8 + 1)
    
    E = "{0:.4f}".format(stats.e1)
    Sigma_E = "{0:.4f}".format(stats.e2)
    Ek = "{0:.4f}".format(stats.ke1)
    Sigma_Ek = "{0:.4f}".format(stats.ke2)
    P_1 = "{0:.4f}".format(stats.p1)
    P_2 = "{0:.4f}".format(stats.p2)
    
    plt.clf()
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
    stats : Stats = None
    with open(path, 'r') as f:
        lines = f.readlines()
        step = int(lines[0].split()[1])
        ts = float(lines[1].split()[1])
        E = float(lines[2].split()[1])
        Ek = float(lines[3].split()[1])
        P_1 = float(lines[4].split()[1])
        Sigma_E = float(lines[5].split()[1])
        Sigma_Ek = float(lines[6].split()[1])
        P_2 = float(lines[7].split()[1])
        stats = Stats(E, Sigma_E, Ek, Sigma_Ek, P_1, P_2, ts, step)
        # loop through the lines and extract the data
        n = 0
        for line in lines[9:]:
            vals = line.split()
            mol = Mol(vals[0], float(vals[1]), float(vals[2]))
            n += 1
            mols.append(mol)
    return n, mols, stats
  

def buildGif(images, folder):
    frames = []
    for i in images:
        temp = Image.open(i)
        keep = temp.copy()
        frames.append(keep)
        temp.close()
        
    for i in images:
        os.remove(i)

    frames[0].save(f'{folder}/result.gif', format='GIF', append_images=frames[1:], save_all=True, duration=30, loop=0)
            
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
                n, mols, stats = readOutput(path)
                # plot the data
                plotMolCoo(mols, stats.ts, 'output/' + mode + '/' + size + '/' + out.replace('.out', '.png'), stats)
            # build the gif
            images = os.listdir(output_dir + mode + '/' + size)
            images = sorted([output_dir + mode + '/' + size + '/' + image for image in images if image.endswith('.png')], key=lambda x: int(x.split('/')[-1].split('.')[0]))
            buildGif(images, output_dir + mode + '/' + size)
    