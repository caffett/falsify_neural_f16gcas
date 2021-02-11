import math
import time
import numpy as np
from numpy import rad2deg

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
from matplotlib.collections import PolyCollection
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from scipy.io import loadmat

import pathlib
current_path = pathlib.Path(__file__).parent.absolute()


def scale3d(pts, scale_list):
    'scale a 3d ndarray of points, and return the new ndarray'

    assert len(scale_list) == 3

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        for d in range(3):
            rv[i, d] = scale_list[d] * pts[i, d]

    return rv


def rotate3d(pts, theta, psi, phi):
    'rotates an ndarray of 3d points, returns new list'

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    sinPsi = math.sin(psi)
    cosPsi = math.cos(psi)
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    transform_matrix = np.array([
        [cosPsi * cosTheta, -sinPsi * cosTheta, sinTheta],
        [cosPsi * sinTheta * sinPhi + sinPsi * cosPhi,
         -sinPsi * sinTheta * sinPhi + cosPsi * cosPhi,
         -cosTheta * sinPhi],
        [-cosPsi * sinTheta * cosPhi + sinPsi * sinPhi,
         sinPsi * sinTheta * cosPhi + cosPsi * sinPhi,
         cosTheta * cosPhi]], dtype=float)

    rv = np.zeros(pts.shape)

    for i in range(pts.shape[0]):
        rv[i] = np.dot(pts[i], transform_matrix)

    return rv


def multi_agent_plot3d_anim(env, skip=1, filename=None):
    start = time.time()

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)

    data = loadmat(f'{current_path}/f-16.mat')
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color="k")
    ax.add_collection3d(plane_polys)

    ax.set_xlabel('X [ft]')
    ax.set_ylabel('Y [ft]')
    ax.set_zlabel('Altitude [ft] ')

    frames = len(env.plant_dict["plant_0"].times)
    trail_lines = {k: ax.plot([], [], [], lw=1)[0] for k in env.plant_dict}

    def anim_func(frame):
        'updates for the animation frame'

        verts = []
        fc = []

        for k, single_env in env.plant_dict.items():
            pos_xs = [pt[9] for pt in single_env.states]
            pos_ys = [pt[10] for pt in single_env.states]
            pos_zs = [pt[11] for pt in single_env.states]

            ax.set_xlim([min(pos_xs), max(pos_xs)])
            ax.set_ylim([min(pos_ys), max(pos_xs)])
            ax.set_zlim([min(pos_zs), max(pos_zs)])
            states = single_env.states

            speed = states[frame][0]
            alpha = states[frame][1]
            beta = states[frame][2]
            alt = states[frame][11]
            phi = states[frame][3]
            theta = states[frame][4]
            psi = states[frame][5]
            dx = states[frame][9]
            dy = states[frame][10]
            dz = states[frame][11]

            # do trail
            trail_lines[k].set_data(pos_xs[:frame],
                                pos_ys[:frame])
            trail_lines[k].set_3d_properties(pos_zs[:frame])

            scale = 15
            pts = scale3d(f16_pts, [-scale, scale, scale])

            pts = rotate3d(pts, theta, -psi, phi)

            size = 1000
            minx = dx - size
            maxx = dx + size
            miny = dy - size
            maxy = dy + size
            minz = dz - size
            maxz = dz + size

            ax.set_xlim([minx, maxx])
            ax.set_ylim([miny, maxy])
            ax.set_zlim([minz, maxz])

            count = 0

            for face in f16_faces:
                face_pts = []

                count = count + 1

                if count % skip != 0:
                    continue

                for index in face:
                    face_pts.append((pts[index - 1][0] + dx,
                                     pts[index - 1][1] + dy,
                                     pts[index - 1][2] + dz))

                verts.append(face_pts)
                fc.append('k')

        # draw ground
        if minz <= 0 and maxz >= 0:
            z = 0
            verts.append([(minx, miny, z), (maxx, miny, z),
                          (maxx, maxy, z), (minx, maxy, z)])
            fc.append('0.8')

        plane_polys.set_verts(verts)
        plane_polys.set_facecolors(fc)

    anim_obj = animation.FuncAnimation(fig, anim_func, frames, interval=30,
                                       blit=False, repeat=True)

    if filename is not None:

        if filename.endswith('.gif'):
            print("\nSaving animation to '{}' using 'imagemagick'...".format(filename))
            anim_obj.save(filename, dpi=80, writer='imagemagick')
            print("Finished saving to {} in {:.1f} sec".format(
                filename, time.time() - start))
        else:
            fps = 50
            codec = 'libx264'

            print("\nSaving '{}' at {:.2f} fps using ffmpeg with codec '{}'.".format(
                filename, fps, codec))

            # if this fails do: 'sudo apt-get install ffmpeg'
            try:
                extra_args = []

                if codec is not None:
                    extra_args += ['-vcodec', str(codec)]

                anim_obj.save(filename, fps=fps, extra_args=extra_args)
                print("Finished saving to {} in {:.1f} sec".format(
                    filename, time.time() - start))
            except AttributeError:
                traceback.print_exc()
                print(
                    "\nSaving video file failed! Is ffmpeg installed? Can you run 'ffmpeg' in the terminal?")
    else:
        plt.show()


def multi_agent_plot3d(env):
    '''
    make a 3d plot of the GCAS maneuver
    '''

    full_plot = True

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)

    data = loadmat(f'{current_path}/f-16.mat')
    f16_pts = data['V']
    f16_faces = data['F']

    plane_polys = Poly3DCollection([], color=None if full_plot else 'k')
    ax.add_collection3d(plane_polys)

    ax.set_xlabel('X [ft]')
    ax.set_ylabel('Y [ft]')
    ax.set_zlabel('Altitude [ft] ')

    verts = []
    fc = []

    for k in env.plant_dict:
        state = env.plant_dict[k].states[-1]
        speed = state[0]
        alpha = state[1]
        beta = state[2]
        alt = state[11]
        phi = state[3]
        theta = state[4]
        psi = state[5]
        dx = state[9]
        dy = state[10]
        dz = state[11]

        scale = 15
        pts = scale3d(f16_pts, [-scale, scale, scale])
        pts = rotate3d(pts, theta, -psi, phi)

        size = 1000
        minx = dx - size
        maxx = dx + size
        miny = dy - size
        maxy = dy + size
        minz = dz - size
        maxz = dz + size

        ax.set_xlim([minx, maxx])
        ax.set_ylim([miny, maxy])
        ax.set_zlim([minz, maxz])

        count = 0

        for face in f16_faces:
            face_pts = []

            count = count + 1

            if not full_plot and count % 10 != 0:
                continue

            for index in face:
                face_pts.append((pts[index - 1][0] + dx,
                                 pts[index - 1][1] + dy,
                                 pts[index - 1][2] + dz))

            verts.append(face_pts)
            fc.append('k')

    # draw ground
    if minz <= 0 and maxz >= 0:
        z = 0
        verts.append([(minx, miny, z), (maxx, miny, z),
                      (maxx, maxy, z), (minx, maxy, z)])
        fc.append('0.8')

    plane_polys.set_verts(verts)
    plane_polys.set_facecolors(fc)
    plt.show()
