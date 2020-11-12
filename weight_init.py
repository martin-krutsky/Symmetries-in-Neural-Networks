import numpy as np

def init_weight_grid(dim_list, start, end, num):
    total_nr = 0
    shapes = []
    for l in range(1, len(dim_list)):
        total_nr += dim_list[l]*(dim_list[l-1] + 1)  # +1 for bias
        shapes.append((dim_list[l], (dim_list[l-1] + 1)))  # +1 for bias
        
    linspaces = [np.linspace(start, end, num) for _ in range(total_nr)]
    mesh_grid = np.array(np.meshgrid(*linspaces)).T.reshape(-1, total_nr)
    
    final_grid = []
    for row in mesh_grid:
        final_grid.append([])
        last_i = 0
        for shape in shapes:
            nr_needed = shape[0]*shape[1]
            final_grid[-1].append(row[last_i:last_i+nr_needed].reshape(*shape))
            last_i += nr_needed

    return final_grid
