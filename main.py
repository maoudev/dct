# Importar las librerías necesarias
from PIL import Image  # Para leer y escribir imágenes
import numpy as np  # Para hacer operaciones matemáticas con matrices
import scipy.fftpack  # Para hacer la transformada discreta del coseno

# Leer la imagen y convertirla a escala de grises
img = Image.open(
    "input.jpg"
)  # Cambiar el nombre del archivo por el de la imagen que se quiere comprimir
img = img.convert("L")  # Convertir a escala de grises
img = np.array(img)  # Convertir a una matriz de NumPy

# Obtener las dimensiones de la imagen
h, w = img.shape

# Dividir la imagen en bloques de 8x8 píxeles
bloques = []
for i in range(0, h, 8):
    for j in range(0, w, 8):
        bloque = img[i : i + 8, j : j + 8]  # Extraer el bloque de 8x8 píxeles
        bloques.append(bloque)  # Añadir el bloque a la lista de bloques

# Aplicar la DCT a cada bloque
bloques_dct = []
for bloque in bloques:
    bloque_dct = scipy.fftpack.dct(
        scipy.fftpack.dct(bloque.T, norm="ortho").T, norm="ortho"
    )  # Aplicar la DCT bidimensional
    bloques_dct.append(bloque_dct)  # Añadir el bloque DCT a la lista de bloques DCT

# Aplicar una matriz de cuantización a cada bloque DCT para reducir el número de bits necesarios para representarlos
# La matriz de cuantización se puede ajustar según el nivel de compresión deseado
matriz_cuantizacion = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

bloques_cuantizados = []
for bloque_dct in bloques_dct:
    bloque_cuantizado = np.round(
        bloque_dct / matriz_cuantizacion
    )  # Dividir el bloque DCT por la matriz de cuantización y redondear al entero más cercano
    bloques_cuantizados.append(
        bloque_cuantizado
    )  # Añadir el bloque cuantizado a la lista de bloques cuantizados

# Reconstruir la imagen a partir de los bloques cuantizados
img_reconstruida = np.zeros(
    (h, w)
)  # Crear una matriz vacía para la imagen reconstruida
k = 0  # Contador para recorrer la lista de bloques cuantizados
for i in range(0, h, 8):
    for j in range(0, w, 8):
        bloque_cuantizado = bloques_cuantizados[k]  # Obtener el bloque cuantizado
        bloque_dct = (
            bloque_cuantizado * matriz_cuantizacion
        )  # Multiplicar el bloque cuantizado por la matriz de cuantización
        bloque = scipy.fftpack.idct(
            scipy.fftpack.idct(bloque_dct.T, norm="ortho").T, norm="ortho"
        )  # Aplicar la IDCT bidimensional
        img_reconstruida[
            i : i + 8, j : j + 8
        ] = bloque  # Sumar el bloque a la matriz de la imagen reconstruida
        k += 1  # Incrementar el contador

img_reconstruida = Image.fromarray(
    img_reconstruida
)  # Convertir el arreglo de numpy a una imagen de Pillow
img_reconstruida = img_reconstruida.convert(
    "RGB"
)  # Convertir la imagen a modo L (escala de grises)
img_reconstruida.save("imagen_reconstruida.webp")  #
