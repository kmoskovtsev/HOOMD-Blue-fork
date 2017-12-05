from matplotlib.animation import FuncAnimation
import gsd.fl
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from hoomd.data import boxdim

def __init__():
    """
    This package is supposed to operate on data calculated in a rectangular box.
    The package assumes this and doesn't check.
    """
    pass

def correct_jumps(pos, pos_m1, lx, ly):
    """    
    Correct jumps over the periodic boundary. Jumps by lx or ly make diffusion calculation
    nonsense.
    pos - positions in current frame
    pos_m1 - position in the previous frame
    lx, ly - unit cell length and width
    
    return: corrected positions in current frame
    """
    
    dr = pos - pos_m1
    # x-direction jumps:
    ind = np.where(dr[:,0] > lx/2)
    dr[ind, 0] = dr[ind, 0] - lx
    ind = np.where(dr[:,0] < - lx/2)
    dr[ind, 0] = dr[ind, 0] + lx
    
    #y-jumps:
    ind = np.where(dr[:,1] > ly/2)
    dr[ind, 1] = dr[ind, 1] - ly
    ind = np.where(dr[:,1] < - ly/2)
    dr[ind, 1] = dr[ind, 1] + ly

    return pos_m1 + dr
    


def get_file_list(folder_path):
    """ 
        Read filenames and corresponding Gamma and T_values in a folder
    """
    f_list, gamma_list, T_list, dt_list = [], [], [], []
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'
    try:
        fl = open(folder_path + 'list.txt', 'r')
    except:
        print('Could not find list.txt in {}'.format(folder_path))
        raise FileNotFoundError
    for line in fl.readlines():
        if line[0] != '#':
            words = line.split('\t')
            f_list.append(words[0])
            gamma_list.append(float(words[1]))
            T_list.append(float(words[2]))
            dt_list.append(float(words[3]))
    fl.close()
    return f_list, gamma_list, T_list, dt_list

    
    
def diffusion_from_gsd(folder_path, center_fixed = True, useframes = -1):
    """
    Calculate diffusion coefficients vs Gamma from gsd files located in folder_path.
    The folder must have list.txt that has the following structure:
    
    # file    Gamma    T    dt
    00000.gsd	145.00000000	1.09746597	0.00100000
    00001.gsd	144.28571429	1.10289897	0.00100000
    
    Lines starting with # are ignored. The columns are tab-separated
    Each gsd file should contain a long enough trajectory in thermalized state.
    
    Diffusion constant D is calculated from 4Dt = <(r(t) - r(0))^2>.
    The average is calculated over all particles and over different time origins.
    Time origins go from 0 to n_frames/2, and t goes from 0 to n_frames/2. This way,
    the data are always within the trajectory.
    
    center_fixed = True: eliminate oveall motion of center of mass
    
    return D_x, D_y, gamma_list, T_list
    D_x, D_y diffusion for x- and y-coordinates;
    all arrays are ordered for right correspondence, e.g. D_x[i] <-> gamma_list[i]
    """
    f_list, gamma_list, T_list, dt_list = get_file_list(folder_path)
    
    D_x_list = np.zeros(len(f_list))
    D_y_list = np.zeros(len(f_list))
    
    if folder_path[-1] != '/':
        folder_path = folder_path + '/'

    for i, f_name in enumerate(f_list):
        print(f_name, end = '\r')
        with gsd.fl.GSDFile(folder_path + f_name, 'rb') as f:
            n_frames = f.nframes
            box = f.read_chunk(frame=0, name='configuration/box')
            half_frames = int(n_frames/2) - 1 #sligtly less than half to avoid out of bound i
            if useframes < 1 or useframes > half_frames:
                useframes = half_frames
            t_step = f.read_chunk(frame=0, name='configuration/step')
            n_p = f.read_chunk(frame=0, name='particles/N')
            if i == 0: #create square-average displacement once
                x_sq_av = np.zeros((useframes, len(f_list)))
                y_sq_av = np.zeros((useframes, len(f_list)))
            for t_origin in range(n_frames - useframes - 1):
                pos_0 = f.read_chunk(frame=t_origin, name='particles/position')
                mean_pos_0 = np.mean(pos_0, axis = 0)
                pos = pos_0
                for j_frame in range(useframes):
                    pos_m1 = pos
                    pos = f.read_chunk(frame=j_frame + t_origin, name='particles/position') - pos_0
                    pos = correct_jumps(pos, pos_m1, box[0], box[1])
                    if center_fixed:
                        pos -= np.mean(pos, axis = 0) - mean_pos_0 #correct for center of mass movement
                    
                    x_sq_av[j_frame, i] += np.mean(pos[:,0]**2)
                    y_sq_av[j_frame, i] += np.mean(pos[:,1]**2)
            x_sq_av[:, i] /= (n_frames - useframes - 1)
            y_sq_av[:, i] /= (n_frames - useframes - 1)
            # OLS estimate for beta_x[0] + beta_x[1]*t = <|x_i(t) - x_i(0)|^2>
            a = np.ones((useframes, 2)) # matrix a = ones(half_frames) | (0; dt; 2dt; 3dt; ...)
            a[:,1] = t_step*dt_list[i]*np.cumsum(np.ones(useframes), axis = 0) - dt_list[i]
            b_cutoff = int(useframes/10) #cutoff to get only linear part of x_sq_av, makes results a bit more clean
            beta_x = np.linalg.lstsq(a[b_cutoff:, :], x_sq_av[b_cutoff:,i], rcond=-1)
            beta_y = np.linalg.lstsq(a[b_cutoff:, :], y_sq_av[b_cutoff:,i], rcond=-1)
            D_x_list[i] = beta_x[0][1]/4
            D_y_list[i] = beta_y[0][1]/4
    return D_x_list, D_y_list, gamma_list, T_list

    
    
def gsd_trajectory(fpath, axis, periodic = False, center_fixed = True):
    """
    Return a trajectory data for all particles in gsd file located at fpath;
    axis = 0 for x coordinates, 1 for y coordinates;
    periodic = True to contain particles within box, False - allow particles to diffuse out of box;
    
    """
    if axis != 0 and axis != 1:
        raise ValueError('axis must be 0 or 1')
    
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = np.zeros((n_frames, n_p[0]))
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        mean_pos_0 = np.mean(pos_frame, axis = 0)
        for j_frame in range(n_frames):
            pos_m1 = pos_frame
            pos_frame = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            if not periodic:
                pos_frame = correct_jumps(pos_frame, pos_m1, box[0], box[1])
            if center_fixed:
                pos_frame -= np.mean(pos_frame, axis = 0) - mean_pos_0
            pos[j_frame, :] = pos_frame[:, axis]
    return pos

def find_neighbors(pos, box, rcut = 1.4):
    """ Find neighbors for each particle. The neighbors are determined to be within rcut distance from the
    particle.
    \param pos - N x 3 array of positions
    \param box - hoomd box object, simulation box
    \param rcut - cutoff radius for neighbors
    Return neighbor_list and neighbor_num.
    neighbor_list - N x 30 array of int, each row i containing indices of neighbors of i-th particle,
                    including the particle itself.
                    The remaining part of the row is filled with -1 (e.g. for 6 neighbors, remaining 23 sites are -1).
    neighbor_num - int array of size N containing numbers of neighbors for each particle. 
    """
    neighbor_list = np.zeros((pos.shape[0], 30), dtype=int) - 1
    neighbor_num = np.zeros(pos.shape[0], dtype = int)
    dist_list = np.zeros((pos.shape[0], 30), dtype=float) - 1

    for i, r in enumerate(pos):
        pos_ref = reshape_to_box(pos, r, box)
        box_ind = np.where((pos_ref[:,0] > -rcut)*(pos_ref[:,0] < rcut)*(pos_ref[:,1] > -rcut)*(pos_ref[:,1] < rcut))[0]
        box_pos = pos_ref[box_ind, :]
        dist = np.sqrt(np.sum(box_pos**2, axis = 1))
        box_neighbors = np.where(dist < rcut)[0]
        box_dist = dist[box_neighbors]
        neighbor_ind = box_ind[box_neighbors]
        neighbor_num[i] = len(neighbor_ind) - 1
        for j, ind in enumerate(neighbor_ind):
            neighbor_list[i, j] = ind
            dist_list[i, j] = box_dist[j]
    return neighbor_list, neighbor_num


    
def animate_gsd(fpath, savefile = None, periodic = False, center_fixed = True, interval = 100, figsize = (12, 12), rcut = 1.4,\
               neighb = False):
    """
    Create animation from a gsd file, where fpath is the path+filename to the gsd file.
    lx, ly - dimensions of the rectangular simulation box;
    savefile - filename for saving the animation. Not trying to save if the value is None;
    periodic - show with periodic boundary conditions if True, let particles migrate out of box if False;
    interval - time interval between frames in microseconds;
    
    return fig, ax
    """
    def init():
        scat.set_offsets(pos[0, :, :])
        
        time_text.set_text('')
        return scat, time_text

    def update(frame_number):
        scat.set_offsets(pos[frame_number, :, :])
        time_text.set_text('frame = {} of {}'.format(frame_number, n_frames))
        if neighb:
            five_ind = np.where(neighbor_num[frame_number, :] == 5)[0]
            seven_ind = np.where(neighbor_num[frame_number, :] == 7)[0]
            five_scat.set_offsets(pos[frame_number, five_ind, :])
            seven_scat.set_offsets(pos[frame_number, seven_ind, :])
        else:
            five_scat.set_offsets(empty_pos)
            seven_scat.set_offsets(empty_pos)
        return scat, time_text, five_scat, seven_scat
    
    with gsd.fl.GSDFile(fpath, 'rb') as f_gsd:
        n_frames = f_gsd.nframes
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box = f_gsd.read_chunk(frame=0, name='configuration/box')
        pos = np.zeros((n_frames, n_p[0], 2))
        neighbor_num = np.zeros((n_frames, int(n_p)), dtype = int)
        pos_frame = f_gsd.read_chunk(frame=0, name='particles/position')
        mean_pos_0 = np.mean(pos_frame, axis = 0)
        for j_frame in range(n_frames):
            pos_m1 = pos_frame
            pos_frame = f_gsd.read_chunk(frame=j_frame, name='particles/position')
            if not periodic:
                pos_frame = correct_jumps(pos_frame, pos_m1, box[0], box[1])
            if center_fixed:
                pos_frame -= np.mean(pos_frame, axis = 0) - mean_pos_0
            pos[j_frame, :, :] = pos_frame[:, 0:2]
            if neighb:
                boxdim_box = boxdim(box[0], box[1], box[2])
                neighbor_list, neighbor_num[j_frame, :] = find_neighbors(pos[j_frame, :,:], boxdim_box, rcut = rcut)
    
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-box[0], box[0]), ylim=(-box[1], box[1]))

    scat = ax.scatter(pos[0, :, 0], pos[0, :, 1],
                      s = 3,
                      facecolors='blue')
    empty_pos = np.zeros(0)
    if neighb:
        five_ind_0 = np.where(neighbor_num[0, :] == 5)[0]
        seven_ind_0 = np.where(neighbor_num[0, :] == 7)[0]
        seven_scat = ax.scatter(pos[0, seven_ind_0, 0], pos[0, seven_ind_0, 1],
                          s = 5,
                          facecolors='green')
        five_scat = ax.scatter(pos[0, five_ind_0, 0], pos[0, five_ind_0, 1],
                          s = 5,
                          facecolors='red')
    else:
        seven_scat = ax.scatter(empty_pos, empty_pos)
        five_scat = ax.scatter(empty_pos, empty_pos)
    time_text = ax.text(0.02, 1.05, '', transform=ax.transAxes)

    animation = FuncAnimation(fig, update, interval=100, frames=n_frames, blit=True)
    
    if not savefile == None:
        try:
            animation.save(savefile, fps=30)
        except Exception as ee: print(ee)
    return fig, ax, animation
    
    
def plot_DxDy(Dx, Dy, gamma_list, timestamp, text_list = [], text_pos = 'c', folder = ''):
    """Create two-panel plot for Dx, Dy vs Gamma
    text_list = list of strings that will be displayed on the plot
    text_pos = 'l', 'c', or 'r' for left, center, and right alignment of the text
    timestamp = string that will serve as a base for the filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (9, 6))
    ax1.scatter(gamma_list, Dx)
    ax1.set_xlabel('$\Gamma$')
    ax1.set_ylabel('$D_x$')

    ax2.scatter(gamma_list, Dy)
    ax2.set_xlabel('$\Gamma$')
    ax2.set_ylabel('$D_y$')
    fig.tight_layout()
    fig.patch.set_alpha(1)
    
    #determine text position
    y_lim = ax1.get_ylim()
    x_lim = ax1.get_xlim()
    h = y_lim[1] - y_lim[0]
    w = x_lim[1] - x_lim[0]
    text_y = y_lim[1] - 0.1*h

    text_x = {
        'l': x_lim[0] + 0.1*w,
        'c': x_lim[0] + 0.4*w,
        'r': x_lim[0] + 0.7*w
    }.get(text_pos, x_lim[0] + 0.4*w)
    
    #print text
    if type(text_list) == list: 
        n_str = len(text_list)
        for i in range(n_str):
            ax1.text(text_x, text_y - 0.05*h*i, text_list[i])
    elif type(text_list) == str:
        ax1.text(text_x, text_y, text_list)
    else:
        raise TypeError('text_list must be a list of strings or a string')
    if folder != '':
        if folder[-1] != '/':
            folder = folder + '/'
    fig.savefig(folder + timestamp + '_diff.png')
    return fig, ax1, ax2
    
    
def plot_positions(system=None, pos=None, box=None, figsize = (7, 7), gridon = True, ax=None, fig=None):
    """ Show positions of all particles in the system,
    where system is the hoomd system, produced by hoomd.init.
    Show grid lines if gridon == True
    """
    if system != None:
        snapshot = system.take_snapshot(all=True)
        box = snapshot.box
        pos = snapshot.particles.position
    if ax==None or fig==None:
        fig = plt.figure(figsize = figsize)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.6*box.Lx, 0.6*box.Lx), ylim=(-0.6*box.Ly, 0.6*box.Ly))
    
    scat = ax.scatter(pos[:, 0], pos[:, 1],
                      s = 3,
                      facecolors='blue')
    if gridon:
        ax.grid()
    return fig, ax
    
    
def reshape_to_box(pos, ref, box):
    """
    Center the simulation box arund ref, and reshape the simulation box to the size of corr_box, 
    potentially filling extra space with images.
    """
    pos_ref = pos - ref
    mask = np.fix(pos_ref[:, 0]/(0.5*box.Lx)) #-1 or 1 if to the left or right from the centered box 
    pos_ref[:, 0] -= mask*box.Lx
    
    mask = np.fix(pos_ref[:, 1]/(0.5*box.Ly))
    pos_ref[:, 1] -= mask*box.Ly
    return pos_ref

def is_odd(num):
    return num & 0x1

def pair_correlation(pos, box, n_bins = (100, 100)):
    """ Calculate pair correlation function for a given snapshot.
        \param positions: N x 3 array, particle positions
        \param box: simulation box object with fields Lx and Ly
        \param n_bins: tuple of size 2, numbers of bins in x and y direction for correlation function, both even
        Return: array of shape n_bins, pair correlation function normalized to (N - 1). Zero of coordinate is in the middle
        of the pixel grid (between pixels indexed (n_bins[i]/2 - 1) and (n_bins[i]/2)).
    """
    try:
        if is_odd(n_bins[0]) or is_odd(n_bins[1]):
            raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
    except:
        raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
        
    bins_x = np.linspace(-box.Lx/2, box.Lx/2, n_bins[0] + 1)
    bins_y = np.linspace(-box.Ly/2, box.Ly/2, n_bins[1] + 1)
    g = np.zeros(n_bins)
    for r in pos:
        pos_ref = reshape_to_box(pos, r, box)
        ind_x = np.digitize(pos_ref[:,0], bins_x) - 1 
        ind_y = np.digitize(pos_ref[:,1], bins_y) - 1
        if np.max(ind_x) >= n_bins[0]:
            ind_x[np.where(ind_x >= n_bins[0])] = n_bins[0] - 1 
        if np.max(ind_y) >= n_bins[1]:
            ind_y[np.where(ind_y >= n_bins[1])] = n_bins[1] - 1
            
        g[ind_x[:], ind_y[:]] += 1
    g[int(n_bins[0]/2), int(n_bins[1]/2)] = 0
    #Normalize so that the sum over all pixels is N - 1
    g /= pos.shape[0]
    return g

def pair_correlation_from_gsd(filename, n_bins = (100, 100), frames =(0, -1)):
    """ Calculate pair correlation function, averaged over all frames in a gsd file.
        \param filename: name of the gsd file
        \param n_bins: tuple of size 2, numbers of bins in x and y direction for correlation function, both even
        Return: array of shape n_bins, pair correlation function normalized to (N - 1). Zero of coordinate is in the middle
        of the pixel grid (between pixels indexed (n_bins[i]/2 - 1) and (n_bins[i]/2)).
    """
    try:
        if is_odd(n_bins[0]) or is_odd(n_bins[1]):
            raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
    except:
        raise ValueError("n_bins must be 2 x 1 tuple of even numbers")
        
    g = np.zeros(n_bins)
    with gsd.fl.GSDFile(filename, 'rb') as f_gsd:
        n_frames_total = f_gsd.nframes
        if frames[0] > n_frames_total or frames[1] > n_frames_total:
            raise ValueError('frames beyond n_frames_total')
        #translate negative indices into positive domain:
        abs_frames = (frames[0] -(frames[0]//n_frames_total)*n_frames_total, \
                     frames[1] -(frames[1]//n_frames_total)*n_frames_total)
        if abs_frames[0] > abs_frames[1]:
            raise ValueError('frames[0] must be to the left from frames[1]')
        all_frames = np.arange(0, n_frames_total, 1)
        selected_frames = all_frames[abs_frames[0]:abs_frames[1] + 1]
        n_frames = abs_frames[1] - abs_frames[0] + 1
        
        n_p = f_gsd.read_chunk(frame=0, name='particles/N')
        box_array = f_gsd.read_chunk(frame=0, name='configuration/box')
        box = boxdim(*box_array[0:3])
        pos = np.zeros((n_frames, n_p[0], 2))
        for j_frame in range(n_frames):
            pos_frame = f_gsd.read_chunk(frame=selected_frames[j_frame], name='particles/position')
            g += pair_correlation(pos_frame, box, n_bins = n_bins)
    g /= n_frames
    return g
    
    
def plot_pair_correlation(g, box, figsize = (8,8), cmap = "plasma", interpolation = 'none', alpha=1):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(np.transpose(g), cmap = cmap, extent = (-box.Lx/2, box.Lx/2, -box.Ly/2, box.Ly/2),\
    origin='lower', interpolation = interpolation, alpha=alpha)
    ax.scatter(0,0, c='r', marker = '+')
    return fig, ax