import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from weight_init import init_weight_grid
from neural_net import L_model_forward, compute_cost_entropy


def compute_single_err(X, y, weight_comb, dim_list, act_function='relu'):
    parameters = dict()
    for l, layer in enumerate(weight_comb):
        parameters['W' + str(l+1)], parameters['b' + str(l+1)] = layer[:, :-1], layer[:, -1]
    
    pred = L_model_forward(X, parameters, hidden_act=act_function)[0]
#     print('pred', pred.shape, 'y', y.shape)
    return compute_cost_entropy(pred, y)

    
def visualize_3D_err_space(X, y, dim_list, nr_of_points_in_dir, act_function='relu'):
    cross_entropies = []
    flattened_ls = []
    weight_combs = init_weight_grid(dim_list, -1, 1, nr_of_points_in_dir)
    
    for comb in weight_combs:
        flattened = np.concatenate(comb).flatten()
        flattened_ls.append(flattened)
        cross_entropies.append(compute_single_err(X, y, comb, dim_list, act_function=act_function))
    
    pca = PCA(n_components=2)
    new_coords = pca.fit_transform(flattened_ls)
    
    print(new_coords)
    xs = new_coords[:, 0]
    ys = new_coords[:, 1]
    
#     xs_sorted = np.sort(xs)
#     print(np.argwhere(xs_sorted==xs_sorted.max())[0], np.argwhere(xs_sorted==xs_sorted.min())[0])
#     nr_in_x = int(np.argwhere(xs_sorted==xs_sorted.max())[0] - np.argwhere(xs_sorted==xs_sorted.min())[0] + 1)
#     print(nr_in_x)
    
#     xs = xs.reshape((nr_in_x, -1))
#     ys = xs.reshape((nr_in_x, -1))
#     z = cross_entropies.reshape(nr_in_x, -1)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
#     ax.plot_surface(xs, ys, z)
    ax.scatter(xs, ys, cross_entropies, s=5)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    
    