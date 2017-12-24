from linalg import dot_product,matrix_mult,get_singular_values,get_eigen_values_and_vectors;
import numpy as np;

M=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]);
a=np.array([1,1,0]);
b=np.array([[-1],[2],[5]]);

aDotB = dot_product(a, b)
print (aDotB);

ans = matrix_mult(M, a, b)
print (ans)

print(get_singular_values(M, 1))
print(get_singular_values(M, 2))

M = np.matrix([[1,2,3],[4,5,6],[7,8,9]])
val, vec = get_eigen_values_and_vectors(M[:,:3], 1)
print("Values = \n", val)
print("Vectors = \n", vec)
val, vec = get_eigen_values_and_vectors(M[:,:3], 2)
print("Values = \n", val)
print("Vectors = \n", vec)
