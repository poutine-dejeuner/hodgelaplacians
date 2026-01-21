from hodgelaplacians import HodgeLaplacians
from hodgelaplacians import WeightedHodgeLaplacians
import numpy as np

from discretediffgeo import random_pointcloud_sphere, alpha_shape_3D, orient, PointCloudSphereToWeightedComplex

# faceset ordering 
hl = HodgeLaplacians(((1,2,3), (1,3,4)), oriented=True)
assert hl.face_set == ((1,), (1, 2), (1, 2, 3), (1, 3), (1, 3, 4), (1, 4), (2,), (2, 3), (3,), (3, 4), (4,))

# boundary operator and Laplacians 
B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0], [-1, 1], [0,-1], [1, 0], [0, 1]])).all()
B = hl.getBoundaryOperator(1).todense()
assert (B == np.matrix([[-1,-1,-1,0,0],[1,0,0,-1,0],[0,1,0,1,-1],[0,0,1,0,1]])).all()
L = hl.getHodgeLaplacian(1).todense()
assert (L == np.matrix([[3, 0, 1, 0, 0],[0, 4, 0, 0, 0],[1, 0, 3, 0, 0],[0, 0, 0, 3, -1],[0, 0, 0, -1, 3]])).all()
L = hl.getHodgeLaplacian(2).todense()
assert (L == np.matrix([[3,-1],[-1,3]])).all()

# larger complex faces and boundary
simplices = sorted(((0,1,2),(0,2,3),(0,3,4),(0,4,1)))
hl = HodgeLaplacians(simplices, oriented=True)
L = hl.getHodgeLaplacian(2).todense()
assert hl.simplices == hl.n_faces(2)
B = hl.getBoundaryOperator(2).todense()
assert (B == np.matrix([[1, 0, 0, -1],[-1, 1, 0, 0],[0, -1, 1, 0],[0, 0, -1, 1],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])).all()
assert ( L == np.matrix([[3, -1, 0,-1],[-1, 3, -1, 0],[0, -1, 3, -1],[-1, 0, -1, 3]]) ).all()

# non-planar surfaces: tetrahedron
pts = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]])


# weighted complexes initialisation
weighted_simplices = {(1,2,3):3,(1,3,4):3, (1,2):2, (1,3):2, (2,3):2, (1,4):2, (3,4):2, (1,):1, (2,):1, (3,):1, (4,):1}
whl = WeightedHodgeLaplacians(weighted_simplices)
assert (whl.weighted_simplices == weighted_simplices)

# weighted Laplacians
L = whl.getHodgeLaplacian(2).todense()
assert(L == np.array([[2., -2/3],[-2/3, 2.]])).all()

# weighted simplicial complex from point cloud
def round(arr, decimals=2):
    output = arr.copy()
    output = np.ceil(output*10**decimals)/(10**decimals)
    return output
pts = random_pointcloud_sphere(10)
verts, edges, faces = alpha_shape_3D(pts, alpha=10)
weighted_simplices = PointCloudSphereToWeightedComplex(pts)
whl = WeightedHodgeLaplacians(weighted_simplices)
L = np.array(whl.getHodgeLaplacian(2).todense())
LL = orient(L)
print(round(L))
print(round(LL))
assert (LL.transpose() == LL).all()