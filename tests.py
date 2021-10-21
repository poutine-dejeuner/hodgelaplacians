from hodgelaplacians import HodgeLaplacians
import numpy as np

hl = HodgeLaplacians(((1,2,3), (1,3,4)), oriented=True)
assert hl.face_set == ((1,), (1, 2), (1, 2, 3), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3), (3,), (3, 4), (4,))

B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0], [-1, 1], [0,-1], [1, 0], [0, 1]])).all()

L = hl.getHodgeLaplacian(2).todense()
assert (L == np.matrix([[3,-1],[-1,3]])).all()

simplices = sorted(((0,1,2),(0,2,3),(0,3,4),(0,4,1)))
hl = HodgeLaplacians(simplices, oriented=True)
L = hl.getHodgeLaplacian(2).todense()
assert hl.simplices == hl.n_faces(2)

B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0, 0, -1],[-1, 1, 0, 0],[0, -1, 1, 0],[0, 0, -1, 1],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])).all()
assert ( L == np.matrix([[3, -1, 0,-1],[-1, 3, -1, 0],[0, -1, 3, -1],[-1, 0, -1, 3]]) ).all()