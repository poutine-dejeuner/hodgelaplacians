from hodgelaplacians import HodgeLaplacians
from hodgelaplacians import WeightedHodgeLaplacians
import numpy as np

# faceset ordering test
hl = HodgeLaplacians(((1,2,3), (1,3,4)), oriented=True)
assert hl.face_set == ((1,), (1, 2), (1, 2, 3), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3), (3,), (3, 4), (4,))

# boundary operator and Laplacians tests
B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0], [-1, 1], [0,-1], [1, 0], [0, 1]])).all()
B = hl.getBoundaryOperator(1).todense()
assert (B == np.matrix([[-1,-1,-1,0,0],[1,0,0,-1,0],[0,1,0,1,-1],[0,0,1,0,1]])).all()
L = hl.getHodgeLaplacian(1).todense()
assert (L == np.matrix([[3, 0, 1, 0, 0],[0, 4, 0, 0, 0],[1, 0, 3, 0, 0],[0, 0, 0, 3, -1],[0, 0, 0, -1, 3]])).all()
L = hl.getHodgeLaplacian(2).todense()
assert (L == np.matrix([[3,-1],[-1,3]])).all()

# larger complex faces and boundary tests
simplices = sorted(((0,1,2),(0,2,3),(0,3,4),(0,4,1)))
hl = HodgeLaplacians(simplices, oriented=True)
L = hl.getHodgeLaplacian(2).todense()
assert hl.simplices == hl.n_faces(2)
B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0, 0, -1],[-1, 1, 0, 0],[0, -1, 1, 0],[0, 0, -1, 1],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])).all()
assert ( L == np.matrix([[3, -1, 0,-1],[-1, 3, -1, 0],[0, -1, 3, -1],[-1, 0, -1, 3]]) ).all()

# weighted complexes laplacian tests
weighted_simplices = {(1,2,3):3,(1,3,4):3, (1,2):2, (1,3):2, (2,3):2, (1,4):2, (3,4):2, (1,):1, (2,):1, (3,):1, (4,):1}
whl = WeightedHodgeLaplacians(weighted_simplices)
assert (whl.simplices == weighted_simplices)

# weighted simplicial complex from point cloud
simplices = []
point_coordinates = []