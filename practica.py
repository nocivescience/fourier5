from manim import *
punto=ORIGIN
vectorizado=VectorizedPoint(punto)
print(vectorizado.get_location)
print(vectorizado.get_location()==np.array([0,0,0]))


# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Point
# from shapely.geometry import LineString

# def triangle_point(t):
#     if t < 0.25:
#         x = 4 * t
#         y = 0
#     elif t < 0.5:
#         x = 1
#         y = 4 * t - 1
#     elif t < 0.75:
#         x = -4 * t + 3
#         y = 1
#     else:
#         x = 0
#         y = -4 * t + 4
#     return Point(x, y)

# path = LineString([triangle_point(t) for t in np.linspace(0, 1, 100)])

# class FourierTransform:
#     def __init__(self, freqs):
#         self.freqs = freqs
#         self.CONFIG = {
#             'center': np.array([0, 0])
#         }

#     def get_coefficients_of_path(self, path, n_samples=1000, freqs=None):
#         if freqs is None:
#             freqs=self.freqs
#         dt=1/n_samples
#         ts=np.arange(0,1,dt)
#         samples=np.array([
#             path.interpolate(t, normalized=True).coords[0] for t in ts
#         ])
#         samples-=self.CONFIG['center']
#         complex_samples=samples[:,0]+1j*samples[:,1]
#         return [
#             np.array([
#                 np.exp(-2*np.pi*1j*freq*t)*cs
#                 for t,cs in zip(ts,complex_samples)
#             ]).sum()*dt for freq in freqs
#         ]

# fourier = FourierTransform(range(-10, 11))
# coefficients = fourier.get_coefficients_of_path(path)

# plt.plot(fourier.freqs, np.abs(coefficients))
# plt.xlabel('Frequencies')
# plt.ylabel('Amplitudes')
# plt.show()


# from shapely.geometry import Point
# from shapely.geometry import LineString
# import numpy as np

# def circle_point(t, r=2):
#     x = r * np.cos(2 * np.pi * t)
#     y = r * np.sin(2 * np.pi * t)
#     return Point(x, y)

# path = LineString([circle_point(t) for t in np.linspace(0, 1, 100)])


# class FourierTransform:
#     def __init__(self, freqs):
#         self.freqs = freqs
#         self.CONFIG = {
#             'center': np.array([0, 0])
#         }

#     def get_coefficients_of_path(self, path, n_samples=1000, freqs=None):
#         if freqs is None:
#             freqs=self.freqs
#         dt=1/n_samples
#         ts=np.arange(0,1,dt)
#         samples=np.array([
#             path.interpolate(t, normalized=True).coords[0] for t in ts
#         ])
#         samples-=self.CONFIG['center']
#         complex_samples=samples[:,0]+1j*samples[:,1]
#         return [
#             np.array([
#                 np.exp(-2*np.pi*1j*freq*t)*cs
#                 for t,cs in zip(ts,complex_samples)
#             ]).sum()*dt for freq in freqs
#         ]

# fourier = FourierTransform([0, 1, 2, -1])
# coefficients = fourier.get_coefficients_of_path(path, freqs=[0, 1, 2, -1])
# print(coefficients)



# import numpy as np
# import matplotlib.pyplot as plt
# from shapely.geometry import Point, LineString

# def circle_point(t, r=2):
#     x = r * np.cos(2 * np.pi * t)
#     y = r * np.sin(2 * np.pi * t)
#     return Point(x, y)

# path = LineString([circle_point(t) for t in np.linspace(0, 1, 100)])

# class FourierTransform:
#     def __init__(self, freqs):
#         self.freqs = freqs
#         self.CONFIG = {
#             'center': np.array([0, 0])
#         }

#     def get_coefficients_of_path(self, path, n_samples=1000, freqs=None):
#         if freqs is None:
#             freqs=self.freqs
#         dt=1/n_samples
#         ts=np.arange(0,1,dt)
#         samples=np.array([
#             path.interpolate(t, normalized=True).coords[0] for t in ts
#         ])
#         samples-=self.CONFIG['center']
#         complex_samples=samples[:,0]+1j*samples[:,1]
#         return [
#             np.array([
#                 np.exp(-2*np.pi*1j*freq*t)*cs
#                 for t,cs in zip(ts,complex_samples)
#             ]).sum()*dt for freq in freqs
#         ]

# def plot_fourier_approximation(fourier, coefficients, n_samples=1000):
#     dt = 1/n_samples
#     ts = np.arange(0, 1, dt)
#     samples = np.zeros((n_samples, 2))
#     for i, t in enumerate(ts):
#         for freq, coeff in zip(fourier.freqs, coefficients):
#             samples[i] += coeff * np.exp(2*np.pi*1j*freq*t)
#     samples += fourier.CONFIG['center']
#     plt.plot(samples[:, 0], samples[:, 1])
#     plt.axis('equal')
#     plt.show()

# fourier = FourierTransform([0, 1, 2, -1])
# coefficients = fourier.get_coefficients_of_path(path, freqs=[0, 1, 2, -1])
# plot_fourier_approximation(fourier, coefficients)