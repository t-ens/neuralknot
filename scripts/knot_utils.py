import os
import sys
import copy
import argparse
import glob
import re
import matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np

def read_object(fname):
    '''
        Quick and dirty way to convert Mathematica graphics output generated by the
        DrawPD function from KnotAtlas to matplotlib graphics objects
    '''

    def split_list(s):
        bracket = 0
        iprev = 0
        L = []
        for i, char in enumerate(s):
            if char=="{" or char=="[":
                bracket += 1
            elif char=="}" or char =="]":
                bracket -= 1
            elif char=="," and bracket==0:
                L.append(s[iprev:i])
                iprev = i+1
        L.append(s[iprev:])
        return L

    def parse_circle(s):
        s = s[8:-1]
        i,j = 0,0
        while s[j] != ',':
            j+=1
        x = float(s[i:j])

        i,j = j+1,j+1
        while s[j] != '}':
            j+=1
        y = float(s[i:j])

        i,j = j+2,j+2
        while s[j] != ',':
            j+=1
        r = float(s[i:j])

        i,j = j+2,j+2

        while s[j] != ',':
            j+=1
        a = float(s[i:j])

        i,j = j+1,j+1
        while s[j] != '}':
            j+=1
        b = float(s[i:j])
        
        return matplotlib.patches.Arc(
                (x,y), 
                2*r, 
                2*r, 
                angle=0.0, 
                theta1=math.degrees(a), 
                theta2=math.degrees(b)
        )

    def parse_line(s):
        s = s[7:-2]

        i,j = 0,0
        while s[j] != ',':
            j+=1
        x1 = float(s[i:j])

        i,j = j+1,j+1
        while s[j] != '}':
            j+=1
        y1 = float(s[i:j])

        i,j = j+3,j+3
        while s[j] != ',':
            j+=1
        x2 = float(s[i:j])

        i,j = j+1,j+1
        while s[j] != '}':
            j+=1
        y2 = float(s[i:j])
        
        return matplotlib.patches.ConnectionPatch((x1, y1), (x2,y2), "data")

    with open(fname) as fd:
        data = fd.read()
        if data == "$Aborted\n":
            return None
        data = data.replace('\n', '')
        data = data.replace(' ', '')

        L = split_list(data[9:-1])
        aspect_ratio = L[1]
        L = split_list(L[0][1:-1])

        #Ignore any color info
        L = [ x for x in L if x[0] != 'R']
   
        O = []
        for l in L:
            if l[0] == 'C':
                O.append(parse_circle(l))
            elif l[0] == 'L':
                O.append(parse_line(l))
        return O

def parse_name(name):
    """
        Extract number of crossings from knot name
    """
    i,j = 5,5
    while name[j] != ',':
        j+=1
    return int(name[i:j])

def parse_gausscode(w):
    match = re.search( '.*?\[(.*)\]', w)
    code = [ int(s) for s in match.groups(0)[0].split(',') ]

def plot_knot(knot, name, rot=0, show=True, save=False, fname = None):
    if knot is None:
        print("The graphing algorithm aborted for this knot.")
        return

    fig, ax = plt.subplots()
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.axis('off')

    t_start = ax.transData
    t = matplotlib.transforms.Affine2D().rotate_deg_around(0, 0, rot)
    t_end = t + t_start
    for i, arc in enumerate(knot):
        col = (math.pi*i % 1, i/len(knot), math.e*i % 1)
        col='k'
        arc.set_edgecolor(col)
        arc.set_transform(t_end)
        ax.add_patch(arc)

        if not save:
            plt.title(name)

    if save:
        plt.savefig(fname, dpi=300)
    if show:
        plt.show()
    else:
        return fig,ax

def find_gaps(knot, num, tol1=0, tol2=0.07, iter=0):
    max_iter = 10000
    step = 0.001
    vertices = []
    for arc in knot:
        if isinstance(arc, matplotlib.patches.Arc):
            r = arc.width/2
            theta1 = math.radians(arc.theta1)
            theta2 = math.radians(arc.theta2)
            cx,cy = arc.center
            x1 = cx + r*math.cos(theta1)
            y1 = cy + r*math.sin(theta1)
            x2 = cx + r*math.cos(theta2)
            y2 = cy + r*math.sin(theta2)
        else:
            x1,y1 = arc.xy1
            x2,y2 = arc.xy2
        vertices.append((np.array([x1,y1]), np.array([x2,y2])))

    gaps = []
    for i,_ in enumerate(knot):
        if i == len(knot)-1:
            break

        d1 = np.linalg.norm( vertices[i][0]-vertices[i+1][0] ) 
        d2 = np.linalg.norm( vertices[i][0]-vertices[i+1][1] ) 
        d3 = np.linalg.norm( vertices[i][1]-vertices[i+1][0] ) 
        d4 = np.linalg.norm( vertices[i][1]-vertices[i+1][0] ) 

        if tol1 < d1  < tol2:
            center = vertices[i][0]
            gaps.append((knot[i], knot[i+1], 0, 0, center))
        elif tol1 < d2 < tol2:
            center = vertices[i][0]
            gaps.append((knot[i], knot[i+1], 0, 1, center))
        elif tol1 < d3 < tol2:
            center = vertices[i][1]
            gaps.append((knot[i], knot[i+1], 1, 0, center))
        elif tol1 < d4 < tol2:
            center = vertices[i][1]
            gaps.append((knot[i], knot[i+1], 1, 1, center))

    if len(gaps) == num or iter >= max_iter:
        return gaps
    elif len(gaps) < num:
        return find_gaps(knot, num, tol2 = tol2 + step, iter = iter+1)
    elif len(gaps) < num:
        return find_gaps(knot, num, tol2 = tol2 - step, iter = iter+1)

        
        
def check_aborted(fname):
    with open(fname) as fd:
        data = fd.read()
        if data == "$Aborted\n":
            return True
        else:
            return False

def make_argparser():
    desc = """
        A set of functions to parse the Mathematica graphics object output by
        the DrawPD routine from the KnotTheory mathematica package into a series
        of Matplotlib patches.
           """
    parser = argparse.ArgumentParser(
                        prog='qutebrowser',
                        description=desc) 

    parser.add_argument('-p', '--plot', action='store_true')
    parser.add_argument('-pc', '--plot_crossings', nargs=1, type=float)
    parser.add_argument('-f', '--first_file', nargs=1, type=int)
    parser.add_argument('-l', '--last_file', nargs=1, type=int)
    parser.add_argument('-kr', '--keep_rejects', action='store_true')
    parser.add_argument('-cr', '--count-rejects', action='store_true')
    parser.add_argument('-lr', '--list-rejects', action='store_true')
    parser.add_argument('-gnc', '--generate-number-of-crossings', action='store_true')
    parser.add_argument('-gd', '--generate-dir', nargs=1, type=str)
    parser.add_argument('-ggc', '--generate-gauss-codes', action='store_true')

    return parser

def flip_knot():
    pass

def main():
    parser = make_argparser()
    args = parser.parse_args(sys.argv[1:])

    data_path = '../../KnotTheory/dataset/objects'
    if not os.path.isdir(data_path):
        print(f'Could not find directory: \"{data_path}\"')
        raise ImportError('Knot data directory not found.')


    all_fnames = glob.glob(data_path + '/*')
    all_fnames.sort()
    with open('../../KnotTheory/dataset/knots.txt') as fd:
        all_names = fd.readlines()[:len(all_fnames)]

    with open('dataset/gauss_codes.txt', 'r') as fd:
        all_gausscodes = fd.readlines()

    if args.keep_rejects:
        fnames = all_fnames
        names = all_names
        gausscodes = all_gausscodes
    else:
        selected = [ False if check_aborted(fname) else True for  fname in all_fnames ]
        fnames = [ fname for i,fname in enumerate(all_fnames) if selected[i] ]
        names = [ name for i,name in enumerate(all_names) if selected[i] ]
        gausscodes = [ gc for i,gc in enumerate(all_gausscodes) if selected[i] ]
        #print(f"Mathematica Graphics objects found: {len(fnames)}") 

    if isinstance(args.first_file, list):
        args.first_file = args.first_file[0]
    if isinstance(args.last_file, list):
        args.last_file = args.last_file[0]+1
    fnames = fnames[args.first_file:args.last_file]
   
    if  args.count_rejects or args.list_rejects:
        rejects = [ fname for fname in all_fnames if check_aborted(fname) ]
        rejects.sort()
        if args.list_rejects:
            for reject in rejects:
                print(reject)
        elif args.count_rejects:
                print(f'Rejected files: {len(rejects)}')

    knots = [ read_object(name) for name in fnames ]

    if args.plot:
        for i,knot in enumerate(knots):
            fig,ax = plot_knot(knot, names[i], show=False)
            if args.plot_crossings is not None:
                length=args.plot_crossings[0]
                gaps = find_gaps(knot, parse_name(names[i]), tol1=0.001, tol2=0.05)
                if gaps is not None and len(gaps) > 0:
                    for _,_,_,_,center in  gaps:
                        ax.add_patch(
                            matplotlib.patches.Rectangle(
                                (center[0]-length/2, center[1]-length/2),
                                length,
                                length,
                                fc=(1,1,0,0.5)
                            )
                        )

        plt.show(block=False)
        print("Press enter to close all graphs.")
        input()

    if args.generate_number_of_crossings:
        if args.generate_dir is None:
            sdir = '../neuralknot/numcrossings/dataset/'

        total_num = 10000
        num_crossings = [ parse_name(name) for name in names ] 
        crossing_dict = dict([ (i, 0) for i in set(num_crossings) ]) 
        for num in num_crossings:
            crossing_dict[num] += 1

        numperclass = total_num // len(crossing_dict) + 1
        rotation_dict = dict( [ (i, numperclass // crossing_dict[i]) for i in crossing_dict.keys()])


        image_base_dir = '/'.join([sdir, 'images'])
        num = 0
        name_list = []
        for i,knot in  enumerate(knots):
            cur_dir = '/'.join([image_base_dir, str(num_crossings[i])])
            if not os.path.isdir(cur_dir):
                os.mkdir(cur_dir)
            for angle in np.linspace(0, 360, num=rotation_dict[num_crossings[i]], endpoint=False):
                knotc = copy.deepcopy(knot) #Matplotlib can only attach a given artist to one figure
                name_list.append(names[i])
                num_zeros = 7-len(str(num))
                fname = '/'.join([cur_dir, num_zeros*'0' + str(num) + '.png'])
                plot_knot(knotc, names[i], rot=angle, show=False, save=True, fname=fname)
                plt.close('all')
                num += 1
                
        with open(sdir + "names.txt", 'a+') as fd:
            fd.writelines( [ name + '\n' for name in name_list ])

    if args.generate_gauss_codes:
        if args.generate_dir is None:
            sdir = '../neuralknot/gaussencoder/dataset/'

        total_num = 10000
        num_crossings = [ parse_name(name) for name in names ] 
        crossing_dict = dict([ (i, 0) for i in set(num_crossings) ]) 
        for num in num_crossings:
            crossing_dict[num] += 1

        numperclass = total_num // len(crossing_dict) + 1
        rotation_dict = dict( [ (i, numperclass // crossing_dict[i]) for i in crossing_dict.keys()])


        image_base_dir = '/'.join([sdir, 'images'])
        num = 0
        name_list = []
        gauss_code_list = []
        for i,knot in  enumerate(knots):
            if not os.path.isdir(image_base_dir):
                os.mkdir(image_base_dir)
            for angle in np.linspace(0, 360, num=rotation_dict[num_crossings[i]], endpoint=False):
                #knotc = copy.deepcopy(knot) #Matplotlib can only attach a given artist to one figure
                name_list.append(names[i])
                gauss_code_list.append(gausscodes[i])
                num_zeros = 7-len(str(num))
                fname = '/'.join([image_base_dir, num_zeros*'0' + str(num) + '.png'])
                #plot_knot(knotc, names[i], rot=angle, show=False, save=True, fname=fname)
                #plt.close('all')
                num += 1
                
        with open(sdir + "names.txt", 'a+') as fd:
            fd.writelines( [ name for name in name_list ])

        with open(sdir+'gauss_codes.txt', 'a+') as fd:
            fd.writelines( [ gc for gc in gauss_code_list ])

if __name__ == '__main__':
    main()
