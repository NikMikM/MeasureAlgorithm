import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy
from scipy import optimize
from scipy import ndimage
import random
import cv2
from sklearn.cluster import KMeans, MeanShift
from sklearn.cluster import DBSCAN
import math
import operator
from functools import reduce

π = np.pi
f = 174 #100 # фокусное расстояние в мм
α1 = -45 #45 # угол поворота нормали (град)
α2 = 0
b = 10 #10 # максимальный угол отклонения нормали (град)
n = 1000 #1000 # точек на графике на 1 последоавтельность
Θ11 = 0  # угол поворота вокруг Х(град)
Θ22 = 0   # угол поворота вокруг Y(град)
Θ33 = 0 # угол поворота вокруг Z(град)
ernum = 3 #3 # колличество шагов уменьшения толщины изображения
BLACK = 75  #75 в центре делаем чёрный квадрат размером BLACkxBLACk
PTS = 10000 #10000 # колличество циклов выбора пяти точек при аппроксимации
AGN = 0 # амплитуда геометрического шума в градац
APN = 0  # амплитуда фотонного шума в размерах пиксела
ATN = 0  # амплитуда теплового шума в размерах пиксела
pixsize = 3.45 * 10 ** (-6)  # размер пиксела 5 мкм
sdvigX = 0 #0 сдвиг вверх в пикселах
sdvigY = 0 #0 сдвиг влево в пикселах
predel = 0.05 # ищем значение teta 3 в окрестность -predel<teta3<+predel
########################## добавить дробовый шум! но он только при матрице

α = α1 * π / 180  # перевод в радианы
α2 = α2 *π / 180
Θ1 = Θ11 * π / 180
Θ2 = Θ22 * π / 180
Θ3 = Θ33 * π / 180
Mp = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2), 0],  # матрица преобразования в приборную систему координат
                   [-1 / np.sqrt(6), -1 / np.sqrt(6), 2 / np.sqrt(6)],
                   [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)]])
Mt = np.array([[np.cos(Θ2) * np.cos(Θ3), -np.cos(Θ2) * np.sin(Θ3), np.sin(Θ2)],  # матрица поворота
                   [np.cos(Θ1) * np.sin(Θ3) + np.cos(Θ3) * np.sin(Θ1) * np.sin(Θ2),
                    np.cos(Θ1) * np.cos(Θ3) - np.sin(Θ1) * np.sin(Θ2) * np.sin(Θ3), -np.cos(Θ2) * np.sin(Θ1)],
                   [np.sin(Θ1) * np.sin(Θ3) - np.cos(Θ1) * np.cos(Θ3) * np.sin(Θ2),
                    np.cos(Θ3) * np.sin(Θ1) + np.cos(Θ1) * np.sin(Θ2) * np.sin(Θ3), np.cos(Θ1) * np.cos(Θ2)]])

Y123 = np.zeros(n)  # создание пустых массивов координат
X123 = np.zeros(n)
Y321 = np.zeros(n)
X321 = np.zeros(n)
Y231 = np.zeros(n)
X231 = np.zeros(n)
Y132 = np.zeros(n)
X132 = np.zeros(n)
Y312 = np.zeros(n)
X312 = np.zeros(n)
Y213 = np.zeros(n)
X213 = np.zeros(n)
X2321 = np.zeros(n)
Y2321 = np.zeros(n)
X1322 = np.zeros(n)
Y1322 = np.zeros(n)


A = np.array([[0], [0], [-1]])  # обычный вектор падающего пучка( закоментить если расходимость выбрана)
for i in range(0, n, 1):
        j = n / b
        δ = (i / j) * π / 180  # разбиение максимального угла отклонения нормали на n частей
        Nx = np.cos(α) * np.sin(δ)  # нормали к цилиндрической поверхности
        Ny = np.sin(α) * np.sin(δ)
        Nz = np.cos(δ)
        Rzc = np.array([[1 - 2 * Nx * Nx, -2 * Nx * Ny, -2 * Nx * Nz],  # матрицы действия граней
                        [-2 * Nx * Ny, 1 - 2 * Ny * Ny, -2 * Ny * Nz],
                        [-2 * Nx * Nz, -2 * Ny * Nz, 1 - 2 * Nz * Nz]])
        Rx = np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])
        Ry = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]])
        Nx2 = np.cos(α2) * np.sin(δ)  # нормали к цилиндрической поверхности
        Ny2 = np.sin(α2) * np.sin(δ)
        Nz2 = np.cos(δ)
        Rzc2 = np.array([[1 - 2 * Nx2 * Nx2, -2 * Nx2 * Ny2, -2 * Nx2 * Nz2],  # матрицы действия граней
                        [-2 * Nx2 * Ny2, 1 - 2 * Ny2 * Ny2, -2 * Ny2 * Nz2],
                        [-2 * Nx2 * Nz2, -2 * Ny2 * Nz2, 1 - 2 * Nz2 * Nz2]])
        R123 = Rzc @ Ry @ Rx  # матрицы действия КЭ
        R231 = Rx @ Rzc @ Ry
        R321 = Rx @ Ry @ Rzc
        R132 = Ry @ Rzc @ Rx
        R2321 = Rx @ Rzc2 @ Ry
        R1322 = Ry @ Rzc2 @ Rx

        B123 = Mt @ Mp @ R123 @ Mp.T @ Mt.T @ A  # вектора отражённого пучка
        B321 = Mt @ Mp @ R321 @ Mp.T @ Mt.T @ A
        B231 = Mt @ Mp @ R231 @ Mp.T @ Mt.T @ A
        B2321 = Mt @ Mp @ R2321 @ Mp.T @ Mt.T @ A
        B1322 = Mt @ Mp @ R1322 @ Mp.T @ Mt.T @ A
        B132 = Mt @ Mp @ R132 @ Mp.T @ Mt.T @ A

        Y123[i] = f * np.tan(np.arcsin(B123[1]))  # координаты на матрице
        X123[i] = f * B123[0] / B123[2]
        Y231[i] = f * np.tan(np.arcsin(B231[1]))
        X231[i] = f * B231[0] / B231[2]
        Y2321[i] = f * np.tan(np.arcsin(B2321[1]))
        X2321[i] = f * B2321[0] / B2321[2]
        Y321[n - i - 1] = f * np.tan(np.arcsin(B321[1]))
        X321[n - i - 1] = f * B321[0] / B321[2]
        Y132[n - i - 1] = f * np.tan(np.arcsin(B132[1]))
        X132[n - i - 1] = f * B132[0] / B132[2]
        Y1322[n - i - 1] = f * np.tan(np.arcsin(B1322[1]))
        X1322[n - i - 1] = f * B1322[0] / B1322[2]

X = np.concatenate((X321, X123))  # преобразования координат в 2 кривые
Y = np.concatenate((Y321, Y123))  # ВЕРТИКАЛЬНАЯ КРИВАЯ
X1 = np.concatenate((X132,X231))  # ГОРИЗОНтАЛьНАЯ КРИВАЯ
Y1 = np.concatenate((Y132,Y231))
X2 = np.concatenate((X1322,X2321))
Y2 = np.concatenate((Y1322,Y2321))


################################################## Рисуем график и сохраняем как png
fig, ax = plt.subplots(dpi=1000)
ax.patch.set_facecolor('black')
plt.plot(X, Y, color='w')  # график последовательностей 1-2-3 и 3-2-1
plt.plot(X1, Y1, color='w')  # график последовательностей 1-3-2 и 2-3-1
plt.plot(X2, Y2, color='w')
#plt.grid()
plt.ylim(-3, 3)
plt.xlim(-3, 3)
plt.gca().set_aspect("equal")
plt.show()
fig.savefig('C:\pycharmProjects\my_plot.png')

################################################## Читаем изображение и обрезаем, имитируя матрицу
img = cv2.imread(r"C:\pycharmProjects\my_plot.png", 0)
cropimg = img[1675+sdvigX:3175+sdvigX, 2530+sdvigY:4030+sdvigY]  # img[1700:3200,2500:4000] #массив, иммитирующий матрицу размером 1500 на 1500
############################################################################# ШУМЫ
rows, cols = cropimg.shape
noise = np.zeros(cropimg.shape)
for i in range(0, rows, 1):
    for j in range(0, cols, 1):
        noise[i,j] = cropimg[i,j] - AGN + 2 * AGN * random.random()
        if 0<noise[i,j]<255:
            cropimg[i,j]=noise[i,j]
print(cropimg[600,750])
# cv2.namedWindow("1",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("1", 600,600)
# cv2.imshow("1", cropimg)
# cv2.waitKey(0)
##################################################
kernel = np.ones((5, 5), np.uint8)
if ernum > 0:
    erosion = cv2.erode(cropimg, kernel,
                        iterations=ernum)  # https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html
else:
    erosion = cropimg
if BLACK > 0:
    rows, cols = cropimg.shape
    x1 = int(np.floor(int(rows) / 2 - BLACK / 2))  # задаём вырезание чёрного прямоугольника
    y1 = int(np.floor(int(cols) / 2 - BLACK / 2))
    x2 = int(np.floor(int(rows) / 2 + BLACK / 2))
    y2 = int(np.floor(int(cols) / 2 + BLACK / 2))
    for i in range(0, rows, 1):
        for j in range(0, cols, 1):
            if (x2 > i > x1) and (y2 > j > y1):
                erosion[i, j] = 0
cropimg = erosion
# cv2.namedWindow("1",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("1", 600,600)
# cv2.imshow("1", cropimg)
# cv2.waitKey(0)
##################                       HOUGH LINE TRANSFORM
# dst = cropimg#cv2.Canny(cropimg, 50, 200, None, 3)
# cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# for i in range(100,1200,50):
#
#     lines = cv2.HoughLines(dst, 1, np.pi / 1800, i, None, 0, 0)
#
#     if len((lines))<6:
#         lines = cv2.HoughLines(dst, 1, np.pi / 1800, i-50, None, 0, 0)
#         break
# lines = cv2.HoughLines(dst, 1, np.pi / 1800, 200, None, 0, 0)
# lin1 = list()
# lin2 = list()
# hough= np.zeros(len(lines))
# if lines is not None:
#     for i in range(0, len(lines)):
#         rho = lines[i][0][0]
#         theta = lines[i][0][1]
#         a = math.cos(theta)
#         b = math.sin(theta)
#         x0 = a * rho
#         y0 = b * rho
#         pt1 = (int(x0 + 1500 * (-b)), int(y0 + 1500 * (a)))
#         pt2 = (int(x0 - 1500 * (-b)), int(y0 - 1500 * (a)))
#         cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
#         lin1.append(list(pt1))
#         lin2.append(list(pt2))
#         if pt2[0]-pt1[0]==0:
#             continue
#         hough[i] = np.arctan((pt2[1]-pt1[1])/(pt2[0]-pt1[0]))*180/π
# hough1 = list()
# hough2 = list()
# houghG = list()
# houghV = list()
# for i in range(len(hough)):
#     if hough[i]>0:
#         hough1.append(hough[i])
#     if hough[i]<0:
#         hough2.append(hough[i])
# hough1 = np.array(hough1)
# hough2 = np.array(hough2)
#
# if np.abs(np.median(hough1))>np.abs(np.median(hough2)):
#     houghV = np.quantile(hough1,0.6)
#     houghG = np.quantile(hough2,0.4)
# if np.abs(np.median(hough2))>np.abs(np.median(hough1)):
#     houghG = np.quantile(hough1,0.6)
#     houghV = np.quantile(hough2,0.4)
# print(hough1)
# print(hough2)
# print(houghV)
# print(houghG)
# cv2.namedWindow("1",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("1", 600,600)
# cv2.imshow("1", cropimg)
# cv2.waitKey(0)
cv2.imwrite(r"C:\pycharmProjects\my1.jpg",cropimg)
#####################################################
OB = 0
OBX = 0
OBY = 0
rows, cols = cropimg.shape
print(cropimg.shape)
XX = []
YY = []

for i in range(0, rows, 1):
    for j in range(0, cols, 1):

        if cropimg[i, j] > 200:
            # OB = OB + cropimg[i, j]
            # OBX = OBX + i * cropimg[i, j]
            # OBY = OBY + i * cropimg[i, j]
            XX.append(i)
            YY.append(j)


# при высичлении угла в переменные cenX cenY ужно занести координаты пересечения кривых
X = np.array(XX)
Y = np.array(YY)
cenX = 750
cenY = 750

Z = list(zip(X, Y))

# https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html#sphx-glr-auto-examples-cluster-plot-cluster-comparison-py
DB = DBSCAN(eps=10, min_samples=5 )#20,2
#DB = MeanShift(2)
model = DB.fit(Z)
labels = model.labels_.tolist()
zipp = list(zip(labels, Z))
print(np.unique(labels))
def get_cluster(n):
    return (list(filter(lambda triple: triple[0] == n, zipp)))

def extract_coords(cluster):
    x = list(map(lambda triple: triple[1][0], cluster))
    y = list(map(lambda triple: triple[1][1], cluster))
    return x, y

x0, y0 = extract_coords(get_cluster(0))
x1, y1 = extract_coords(get_cluster(1))
x2, y2 = extract_coords(get_cluster(2))
x3, y3 = extract_coords(get_cluster(3))
x4, y4 = extract_coords(get_cluster(4))
x5, y5 = extract_coords(get_cluster(5))

x0.extend(x5)
y0.extend(y5)
x2.extend(x3)
y2.extend(y3)
x4.extend(x1)
y4.extend(y1)

#x2,x3 (синяя)    x0,x5  (желтая)    x1,x4 (зелая)
fig2, ax = plt.subplots()
ax.patch.set_facecolor('black')
plt.scatter(x2, y2) #горизонтальая линия
plt.scatter(x0, y0) # вертикальная линия
plt.scatter(x4,y4) # наклонная линия
plt.show()
X = np.array(x4)
Y = np.array(y4)
Y2 = np.unique(Y)
ii = 0
xx = list()
yy = list()
for jj in range(len(Y)):
    if Y[jj] == Y2[ii]:
        xx.append(X[jj])
    else:
        if ii<len(Y2)-1:
            ii=ii+1


print(xx)
print(len(x0), len(x1),len(x2),len(x3))
print(cenX, cenY, len(X), len(Y))
##############################################################################
rationalFn = lambda w, x: (w[0] * x - w[1])
errFn = lambda w, x, y: rationalFn(w, x) - y
res = optimize.least_squares(errFn, (0, 0), args=(X, Y), method='lm', verbose=True)
w = res.x
print(f'x = {w}')
print(f'Измереный угол Θ2={np.arctan(w[0]) * 180 / π}')
print(f'Заданный угол Θ2={Θ2 * 180 / π}')
print(len(X), len(Y))


################################################### Начинаем перебор точек и поиск угла
teta3 = np.zeros(PTS)  # создаём массив измерений
teta5 = list()
for i in range(0, PTS, 1):
    g1 = np.random.randint(1, np.floor(len(X) / 5))
    g2 = np.abs(np.random.randint(np.floor(len(X) / 5), np.floor(len(X)*2 / 5)))
    g3 = np.abs(np.random.randint(np.floor(len(X)*2 / 5), np.floor(len(X)*3 / 5)))
    g4 = np.abs(np.random.randint(np.floor(len(X)*3 / 5), np.floor(len(X)*4 / 5)))
    g5 = np.abs(np.random.randint(np.floor(len(X)*4/ 5), np.floor(len(X)) - 1))
    x = np.array([X[g1], X[g2], X[g3], X[g4], X[g5]])
    y = np.array([Y[g1], Y[g2], Y[g3], Y[g4], Y[g5]])
    mat = np.array([[x[0] ** 2, x[0] * y[0], y[0] ** 2, x[0], y[0]],
                    [x[1] ** 2, x[1] * y[1], y[1] ** 2, x[1], y[1]],
                    [x[2] ** 2, x[2] * y[2], y[2] ** 2, x[2], y[2]],
                    [x[3] ** 2, x[3] * y[3], y[3] ** 2, x[3], y[3]],
                    [x[4] ** 2, x[4] * y[4], y[4] ** 2, x[4], y[4]]])
    if np.linalg.det(mat) == 0:
        teta3[i]=0
        continue
    res = np.array([[-1], [-1], [-1], [-1], [-1]])
    koef = np.linalg.solve(mat, res)
    # формула взята с сайта https://mathworld.wolfram.com/ConicSection.html
    # формула взята с сайта https://abakbot.ru/online-2/202-kasatelnaya-k-krivoj-vtorogo-poryadka коэффициенты а2 и а3 поменяны местами
    teta2 = np.arctan(
        -(koef[0] * cenX + (koef[1] * cenY + koef[3]) / 2) / (koef[1] * cenX / 2 + koef[4] / 2 + koef[2] * cenY))
    teta2 = teta2 * 180 / π
    teta3[i] = teta2
    teta5.append(teta2)
########################### Вычисляем значения угла ( для идеального алгоритма используются квантили 0.8 при знаке угла '+'
# и 0.3 при знаке угла '-'
print(koef)
if (np.median(teta3) < 90):
    m = np.median(teta3) + 90
    qua = -0.0003 * m ** 6 + 0.0101 * m ** 5 - 0.1235 * m ** 4 + 0.7644 * m ** 3 - 2.5372 * m ** 2 + 4.308 * m - 2.37
    qua = 0.6 if (qua > 0.6) else qua
    qua = 0.41 if (qua < 0.42) else qua
    qua = 0.1
    quantile = np.quantile(teta3, qua) + 90
    медиана = np.median(teta3) + 90
    среднее = np.mean(teta3) + 90
else:
    # quantile = np.quantile(teta3, 0.3) - 90
    # медиана = np.median(teta3) - 90
    # среднее = np.mean(teta3) - 90
    quantile = np.quantile(teta3, 0.5)
    медиана = np.median(teta3)
    среднее = np.mean(teta3)
maxx = 0
max = 0
izm=[]
for i in range(len(teta3)):
    cnt = 0
    for j in range(len(teta3)):
        if teta3[j]-predel<teta3[i]<teta3[j]+predel:
            cnt = cnt+1
    max = cnt
    if max>maxx:
        maxx = max
        izm = teta3[i]+90

teta4 = np.quantile(teta3, 0.9)

###########2 = 0.53 3=0.57 4 = 0.6 5 = 0.61 6 = 0.61 7 = 0.61 8 =0.62 9 = 0.62 10=0.61 15=0.62 20=0.63 30=0.63
####34.614 = 0.6 39.644=0.59
квантиль2 = np.quantile(teta3, 0.8)
квантиль3 = np.quantile(teta3, 0.2)
макс = np.max(teta3)
мин = np.min(teta3)

print(f'квантиль 0.8:{квантиль2}')
print(f'квантиль 0.2:{квантиль3}')

print(f'среднее:{среднее}')
print(f'медиана:{медиана}')
print(f'макс:{макс}')
print(f'мин:{мин}')
print(f'qua:{qua}')

print(f'max:{max}')


print(f'заданныйΘ3:{Θ3 * 180 / π}')
print(f'измеренный Θ3:{izm}')
print(len(np.unique(teta5, axis=0)))
zer = np.zeros(PTS)
for i in range(PTS):
    zer[i] = i
fig2, ax = plt.subplots()
ax.patch.set_facecolor('black')
plt.scatter(teta3,zer)
plt.show()
Θ22 = 8.17961761 * 10 ** (-15) * izm ** 10-1.72961250 * 10 ** (-12) * izm ** 9+1.56949563 * 10 ** (-10) * izm ** 8-7.9211477 * 10 ** (-9) * izm ** 7+2.42546043 * 10 ** (-7) * izm ** 6 - 4.58325316 * 10 ** (-6) * izm ** 5 + 5.29455012 * 10 ** (
       -5) * izm ** 4 - 2.57479072 * 10 ** (-4) * izm ** 3 + 1.13326955 * 10 ** (-3) * izm ** 2 + 7.05873103 * 10 ** (
             -1) * izm + 1.18139400 * 10 ** (-4)
print(f'заданныйΘ2:{Θ2 * 180 / π}')
print(f'измеренный угол θ2:{Θ22}')
print(f'заданныйΘ1:{Θ1 * 180 / π}')
izm=izm-60
Θ111 = 8.72959770 * 10 ** (-7) * izm ** 8 +4.53992464 * 10 ** (-5) * izm ** 7 +9.64368156 * 10 ** (-4) * izm ** 6 +1.06983485 * 10 ** (-2) * izm ** 5 +6.60447962 * 10 ** (-2) * izm ** 4 +2.20442664 * 10 ** (-1) * izm ** 3 + 3.98207937 * 10 ** (-1) * izm ** 2 -1.41002829 * 10 ** (0) * izm ** 1 + 1.37210079* 10 ** (-2)
print(f'измеренный угол θ1:{Θ111}')
#####################################
#Θ3 = 0.0002855470*ξ^3-0.0088215312*ξ^2+0.8166691469*ξ-0.2578202082
# ##########################################
#решение матриц другими методами
# OM=np.linalg.inv(mat)
# OX=np.matmul(OM, res)
# koef2 = OX
# ##########################################
# mat = [[x[0]**2, x[0]*y[0], y[0]**2, x[0],y[0]],
#               [x[1]**2, x[1]*y[1], y[1]**2, x[1],y[1]],
#                [x[2]**2, x[2]*y[2], y[2]**2, x[2],y[2]],
#                [x[3]**2, x[3]*y[3], y[3]**2, x[3],y[3]],
#                [x[4]**2, x[4]*y[4], y[4]**2, x[4],y[4]]]
# res = [-1,-1,-1,-1,-1]
# def Kram(A,B):
#     m=len(A)
#     op = np.linalg.det(A)
#     r = list()
#     for i in range(m):
#         vm = np.copy(A)
#         vm[:,i] = B
#         r.append(np.linalg.det(vm)/op)
#     return r
# koef3 = Kram(mat,res)
# koef = koef2
# ##########################################
# попытка сегментации изображения
# img =  cv2.cvtColor(cropimg, cv2.COLOR_GRAY2BGR)
# gray = cropimg
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# # noise removal
# kernel = np.ones((100,100),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# # sure background area
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
# # Finding sure foreground area
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# # Finding unknown region
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
# # Marker labelling
# ret, markers = cv2.connectedComponents(sure_fg)
# # Add one to all labels so that sure background is not 0, but 1
# markers = markers+1
# # Now, mark the region of unknown with zero
# markers[unknown==255] = 0
# markers = cv2.watershed(img, markers)
# img[markers == -1] = [255,0,0]
# print(len(markers))
# print(np.unique(markers))
# print(img.shape)
######################################################################
#попытка кластеризации

# def sort_clockwise (coords):
#     center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), coords), [len(coords)] * 2))
#     return sorted(coords,key=lambda coord: (-135 - math.degrees(math.atan2(map(operator.sub, coord, center)[::-1]))) % 360)
# Z = list(zip(X, Y))
# kmeans = KMeans(init="k-means++", n_clusters=4, n_init=10, random_state=0, tol=0.001)
# model = kmeans.fit(Z)
# labels = model.labels_.tolist()
# centroids = model.cluster_centers_
# zipp = list(zip(labels, Z))
#
# def get_cluster(n):
#     return (list(filter(lambda triple: triple[0] == n, zipp)))
#
# def extract_coords(cluster):
#     x = list(map(lambda triple: triple[1][0], cluster))
#     y = list(map(lambda triple: triple[1][1], cluster))
#     return x, y
# #sorted = sort_clockwise(centroids)
# print(sorted)
# x0, y0 = extract_coords(get_cluster(0))
# x1, y1 = extract_coords(get_cluster(1))
# x2, y2 = extract_coords(get_cluster(2))
# x3, y3 = extract_coords(get_cluster(3))
# fig2, ax = plt.subplots()
# ax.patch.set_facecolor('black')
#
# x0.extend(x1)
# y0.extend(y1)
# x2.extend(x3)
# y2.extend(y3)
# plt.scatter(x2, y2)
# plt.scatter(x0, y0)
# plt.show()
#
# print(len(x0), len(x1),len(x2),len(x3))
