import streamlit as st
import numpy as np
from PIL import Image

st.write('# Illustrating olor depth !!!')

width = 2**11
height = 64
levels = int(st.number_input('Depth ?', min_value=1, max_value=11, step=1))
matrix = np.zeros((8, 256), dtype=np.uint8)

for idx, bit_depth in enumerate(range(1, 9)):
    num_colors = 2 ** bit_depth
    step = 255 // (num_colors - 1)
    greyscale_values = np.arange(0, 256, step, dtype=np.uint8)
    out = np.repeat(greyscale_values, 256 // num_colors)
    matrix[idx] = out

st.write(f'## Levels = {levels}')
image = matrix[:levels]
image = np.repeat(image, 10, axis=1)
image = np.repeat(image, 200, axis=0)

image = Image.fromarray(image)
st.image(image,
         caption='Grayscale Gradient (Black to White)', 
         use_column_width=True,
         output_format='png')
