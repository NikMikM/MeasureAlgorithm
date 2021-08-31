import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import optimize
π = np.pi
f = 100 #фокусное расстояние в мм
α1 = 45 #угол поворота нормали (град)
b = 10 #максимальный угол отклонения нормали (град)
n = 1000 # точек на графике
PTS = 10000 #колличество циклов выбора пяти точек при аппроксимации
Θ11 = 0 #угол поворота вокруг Х(град)
Θ22 = 0 #угол поворота вокруг Y(град)
Θ33 = 15 #угол поворота вокруг Z(град)
α = α1*π/180 #перевод в радианы
Θ1 = Θ11*π/180
Θ2 = Θ22*π/180
Θ3 = Θ33*π/180

η=1 #угловая расходимость


Mp = np.array([[-1/np.sqrt(2), 1/np.sqrt(2), 0], #матрица преобразования в приборную систему координат
               [-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
               [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]])
Mt = np.array([[np.cos(Θ2)*np.cos(Θ3), -np.cos(Θ2)*np.sin(Θ3), np.sin(Θ2)], #матрица поворота
              [np.cos(Θ1)*np.sin(Θ3)+np.cos(Θ3)*np.sin(Θ1)*np.sin(Θ2), np.cos(Θ1)*np.cos(Θ3)-np.sin(Θ1)*np.sin(Θ2)*np.sin(Θ3), -np.cos(Θ2)*np.sin(Θ1)],
             [np.sin(Θ1)*np.sin(Θ3)-np.cos(Θ1)*np.cos(Θ3)*np.sin(Θ2), np.cos(Θ3)*np.sin(Θ1)+np.cos(Θ1)*np.sin(Θ2)*np.sin(Θ3), np.cos(Θ1)*np.cos(Θ2)]])
Y123 = np.zeros(n) #создание пустых массивов координат
X123 = np.zeros(n)
Y321 = np.zeros(n)
X321 = np.zeros(n)
Y231 = np.zeros(n)
X231 = np.zeros(n)
Y132 = np.zeros(n)
X132 = np.zeros(n)
for h in range (-η, η, 1):
    h=h/100
    for j in range (-η, η, 1):
        j=j/100
        A = np.array([[np.sin(j)*np.cos(h)],  # вектор падающего пучка (с расходимостью)
                    [np.sin(h)],
                    [-np.cos(j)*np.cos(h)]])
        A = np.array([[0], [0], [-1]]) #обычный вектор падающего пучка( закоментить если расходимость выбрана)
        for i in range(0,n,1):
                j = n/b
                δ = (i/j)* π / 180 #разбиение максимального угла отклонения нормали на n частей
                Nx = np.cos(α)*np.sin(δ) #нормали к цилиндрической поверхности
                Ny = np.sin(α)*np.sin(δ)
                Nz = np.cos(δ)
                Rzc = np.array([[1-2*Nx*Nx, -2*Nx*Ny, -2*Nx*Nz], #матрицы действия граней
                            [-2*Nx*Ny, 1-2*Ny*Ny, -2*Ny*Nz],
                            [-2*Nx*Nz, -2*Ny*Nz, 1-2*Nz*Nz]])
                Rx = np.array([[-1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]])
                Ry = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 1]])
                R123 = Rzc @ Ry @ Rx #матрицы действия КЭ
                R231 = Rx @ Rzc @ Ry
                R321 = Rx @ Ry @ Rzc
                R132 = Ry @ Rzc @ Rx
                B123 = Mt @ Mp @ R123 @ Mp.T @ Mt.T @ A #вектора отражённого пучка
                B321 = Mt @ Mp @ R321 @ Mp.T @ Mt.T @ A
                B231 = Mt @ Mp @ R231 @ Mp.T @ Mt.T @ A
                B132 = Mt @ Mp @ R132 @ Mp.T @ Mt.T @ A
                Y123[i] = f*np.tan(np.arcsin(B123[1])) #координаты на матрице
                X123[i] = f*B123[0]/B123[2]
                Y231[i] = f*np.tan(np.arcsin(B231[1]))
                X231[i] = f*B231[0]/B231[2]
                Y321[n-i-1] = f*np.tan(np.arcsin(B321[1]))
                X321[n-i-1] = f*B321[0]/B321[2]
                Y132[n-i-1] = f*np.tan(np.arcsin(B132[1]))
                X132[n-i-1] = f*B132[0]/B132[2]
        X = np.concatenate((X321, X123)) #преобразования координат в 2 кривые
        Y = np.concatenate((Y321, Y123))
        X1 = np.concatenate((X132, X231))
        Y1 = np.concatenate((Y132, Y231))

        plt.plot(X, Y) #график последовательностей 1-2-3 и 3-2-1
        plt.plot(X1, Y1) #график последовательностей 1-3-2 и 2-3-1
        plt.grid()
        plt.gca().set_aspect("equal")
#plt.show()
X = X1 # ИЗМЕНЕНИЕ КРИВЫХ ТО ЕСТЬ ГОРИЗОНТАЛЬНАЯ РАВНА ВЕРТИКАЛЬНАЯ
Y = Y1

rationalFn = lambda w,x: (w[0]*x-w[1])
errFn = lambda w,x,y: rationalFn(w,x)-y
res=optimize.least_squares(errFn, (0,0), args=(X,Y), method = 'lm', verbose=True)
w = res.x
print(f'x = {w}')
print(f'Измереный угол Θ2={np.arctan(w[0])*180/π}')
print(f'Заданный угол Θ2={Θ2*180/π}')
print(len(X),len(Y))
# γ=10
# γ1 = γ*π/180
# #γ1 = w[2]
# print (np.sin(γ1))
# # aa=w[0]
# # bb=w[1]
# aa=20
# bb=10
#
#
# x = np.zeros(n)
# y = np.zeros(n)
# x1 = np.zeros(n)
# y1 = np.zeros(n)
# z1 = np.zeros(n)
# z2 = np.zeros(n)
# Mγ=[[np.cos(γ1), -np.sin(γ1),],
#     np.sin(γ1), np.cos(γ1)]
# for i in range(0, n, 1):
#     x[i] = 10 -20*i/(n-1)
#     y[i] = np.sqrt(bb*bb*(1-(((x[i])*(x[i]))/(aa*aa))))
#     x1[i]= x[i]*np.cos(γ1)-y[i]*np.sin(γ1)
#     y1[i]= x[i]*np.sin(γ1)+y[i]*np.cos(γ1)
#     z1[i]= (np.sqrt(bb*bb*(1-((x1[i]-bb*np.sin(γ1))*(x1[i]-bb*np.sin(γ1))/(aa*aa)))))*np.cos(γ1)+(x1[i]-bb*np.sin(γ1))*np.sin(γ1)-bb*np.cos(γ1)
#     z2[i]=-(np.sqrt(bb*bb*(1-((x[i]-bb*np.sin(γ1))*(x[i]-bb*np.sin(γ1))/(aa*aa)))))*np.cos(γ1)+(x[i]-bb*np.sin(γ1))*np.sin(γ1)-bb*np.cos(γ1)
# plt.plot(x,z1)
# plt.plot(bb*np.sin(γ1),-bb*np.cos(γ1), marker = 'x')
# plt.grid()
# plt.gca().set_aspect("equal")
# plt.show()
cenX = 0
cenY = 0
teta3=np.zeros(PTS)#создаём массив измерений
for i in range(0,PTS,1):
    g1= np.random.randint(1,np.floor(len(X)/5))
    g2 = np.abs(g1 + np.random.randint(1, np.floor(len(X)/5)))
    g3 = np.abs(g2 + np.random.randint(1, np.floor(len(X)/5)))
    g4 = np.abs(g3 + np.random.randint(1, np.floor(len(X)/5)))
    g5 = np.abs(g4 + np.random.randint(1, np.floor(len(X)/5)-1))
    x = np.array([X[g1],X[g2],X[g3],X[g4],X[g5]])
    y = np.array([Y[g1],Y[g2],Y[g3],Y[g4],Y[g5]])
    mat = np.array([[x[0] ** 2, x[0] * y[0], y[0] ** 2, x[0], y[0]],
                    [x[1] ** 2, x[1] * y[1], y[1] ** 2, x[1], y[1]],
                    [x[2] ** 2, x[2] * y[2], y[2] ** 2, x[2], y[2]],
                    [x[3] ** 2, x[3] * y[3], y[3] ** 2, x[3], y[3]],
                    [x[4] ** 2, x[4] * y[4], y[4] ** 2, x[4], y[4]]])
    if np.linalg.det(mat)==0:
        continue
    res = np.array([[-1], [-1], [-1], [-1], [-1]])
    koef = np.linalg.solve(mat, res)

    # формула взята с сайта https://mathworld.wolfram.com/ConicSection.html
    # формула взята с сайта https://abakbot.ru/online-2/202-kasatelnaya-k-krivoj-vtorogo-poryadka коэффициенты а2 и а3 поменяны местами
    teta2 = np.arctan(-(koef[0] * cenX + (koef[1] * cenY + koef[3]) / 2) / (koef[1] * cenX / 2 + koef[4] / 2 + koef[2] * cenY))
    teta2 = teta2 * 180 / π
    teta3[i] = teta2
########################### Вычисляем значения угла ( для идеального алгоритма используются квантили 0.8 при знаке угла '+'
# и 0.3 при знаке угла '-'
if (np.median(teta3)>0):
    quantile = np.quantile(teta3, 0.8)
else:
    quantile = np.quantile(teta3,0.3)

print(koef)
print(f'измеренный по 5 точкам угол:{quantile}')
print(f'Измереный угол (апроксимация) Θ2={np.arctan(w[0])*180/π}')
print(f'Заданный угол Θ2={Θ2*180/π}')
print(f'Заданный угол Θ3={Θ3*180/π}')
# print(x)
# print(y)
# print(cenX)
# print(cenY)
# print(Y[int(len(Y)-50)])
# print(X[int(len(Y)-50)])