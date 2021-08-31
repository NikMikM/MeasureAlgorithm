import numpy as np
import matplotlib.pyplot as plt
A = np.array([[0],
              [0],
              [-1]])
Mp = np.array([[-1/np.sqrt(2), 1/np.sqrt(2), 0],
               [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
               [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])
π = np.pi
f = 100
α = 45*π/180
b = 10
n = 1000
Y123 = np.zeros(n)
X123 = np.zeros(n)
Y321 = np.zeros(n)
X321 = np.zeros(n)
Y231 = np.zeros(n)
X231 = np.zeros(n)
Y132 = np.zeros(n)
X132 = np.zeros(n)

for i in range(0,n,1):
    j = n/b
    δ = (i/j)* π / 180
    Nx = np.cos(α)*np.sin(δ)
    Ny = np.sin(α)*np.sin(δ)
    Nz = np.cos(δ)
    Rzc = np.array([[1-2*Nx*Nx, -2*Nx*Ny, -2*Nx*Nz],
                [-2*Nx*Ny, 1-2*Ny*Ny, -2*Ny*Nz],
                [-2*Nx*Nz, -2*Ny*Nz, 1-2*Nz*Nz]])
    Rx = np.array([[-1, 0, 0],
               [0, 1, 0],
               [0, 0, 1]])
    Ry = np.array([[1, 0, 0],
               [0, -1, 0],
               [0, 0, 1]])
    R123 = Rzc @ Ry @ Rx
    R231 = Rx @ Rzc @ Ry
    R321 = Rx @ Ry @ Rzc
    R132 = Ry @ Rzc @ Rx
    B123 = Mp @ R123 @ Mp.T @ A
    B321 = Mp @ R321 @ Mp.T @ A
    B231 = Mp @ R231 @ Mp.T @ A
    B132 = Mp @ R132 @ Mp.T @ A
    Y123[i] = f*np.tan(np.arcsin(B123[1]))
    X123[i] = f*B123[0]/B123[2]
    Y231[i] = f*np.tan(np.arcsin(B231[1]))
    X231[i] = f*B231[0]/B231[2]
    Y321[n-i-1] = f*np.tan(np.arcsin(B321[1]))
    X321[n-i-1] = f*B321[0]/B321[2]
    Y132[n-i-1] = f*np.tan(np.arcsin(B132[1]))
    X132[n-i-1] = f*B132[0]/B132[2]
X = np.concatenate((X321, X123))
Y = np.concatenate((Y321, Y123))
X1 = np.concatenate((X132, X231))
Y1 = np.concatenate((Y132, Y231))
print(X1)
print(Y1)
plt.plot(X, Y)
plt.plot(X1, Y1)
plt.grid()
plt.show()

