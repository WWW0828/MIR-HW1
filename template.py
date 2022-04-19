binary_template = [[1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
                   [0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
                   [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1],
                   [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
                   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0], [0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1]]

KS_major_template = [[6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], 
                     [2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29],
                     [2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66], 
                     [3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39],
                     [2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19], 
                     [5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52],
                     [2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38, 4.09], 
                     [4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33, 4.38],
                     [4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48, 2.33], 
                     [2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23, 3.48],
                     [3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35, 2.23], 
                     [2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.28, 6.35]]

KS_minor_template = [[6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], 
                     [3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34],
                     [3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69], 
                     [2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98],
                     [3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75], 
                     [4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54],
                     [2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60, 3.53], 
                     [3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38, 2.60],
                     [2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52, 5.38], 
                     [5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68, 3.52],
                     [3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33, 2.68], 
                     [2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17, 6.33]]

KS_template = KS_major_template + KS_minor_template