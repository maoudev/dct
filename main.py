import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct


def dct2(a):
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


def idct2(a):
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def compress_channel(channel, quality):
    # Aplicar la DCT al canal de color
    channel_dct = dct2(channel)

    # Umbral de calidad para la compresión
    threshold = np.max(channel_dct) / quality

    # Aplicar el umbral a los coeficientes DCT
    channel_dct[channel_dct < threshold] = 0

    # Aplicar la IDCT para obtener el canal de color comprimido
    channel_compressed = idct2(channel_dct)

    # Normalizar los valores al rango 0-255
    channel_compressed = np.clip(channel_compressed, 0, 255)

    # Convertir a enteros de 8 bits
    channel_compressed = channel_compressed.astype("uint8")

    return channel_compressed


def compress_image(image_path, quality):
    # Cargar la imagen
    img = Image.open(image_path)

    # Convertir la imagen a una matriz y separar los canales de color
    img_data = np.array(img)
    red, green, blue = img_data[:, :, 0], img_data[:, :, 1], img_data[:, :, 2]

    # Comprimir cada canal de color
    red_compressed = compress_channel(red, quality)
    green_compressed = compress_channel(green, quality)
    blue_compressed = compress_channel(blue, quality)

    # Combinar los canales de color comprimidos
    img_compressed = np.stack(
        (red_compressed, green_compressed, blue_compressed), axis=2
    )

    # Crear y guardar la imagen comprimida
    img_compressed = Image.fromarray(img_compressed)
    img_compressed.save("compressed_image.jpg")


# Uso del código
compress_image("input.jpg", 600)
