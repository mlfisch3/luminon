from bimef import bimef, entropy, normalize_array
import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
import time

st.set_page_config(page_title="Luminon", layout="wide")

def run_app():

    @st.cache(max_entries=1)
    def array_info(array, print_info=True, return_info=False):

        array = array / 255.0
        info = {}
        info['dtype'] = array.dtype
        info['ndim'] = array.ndim
        info['shape'] = array.shape
        info['max'] = array.max()
        info['min'] = array.min()
        info['mean'] = array.mean()
        info['std'] = array.std()
        info['size'] = array.size
        info['nonzero'] = np.count_nonzero(array)
        info['layer_variation'] = 0
        info['entropy'] = entropy(array)

        if array.ndim > 2:
            info['layer_variation'] = array.std(axis=array.ndim-1).mean()

        info['pct'] = 100 * info['nonzero'] / info['size']

        if print_info:
            print('{dtype}  {shape}'.format(**info))
            print('nonzero: {nonzero} / {size}  ({pct:.1f} %)'.format(**info))
            print('min:  {min:.2f}   max: {max:.2f}'.format(**info))
            print('mean: {mean:.2f}   std: {std:.2f}'.format(**info), end="")
            if array.ndim > 2:
                print('     layer_variation: {layer_variation:.2f}'.format(**info))

            print('entropy: {entropy:.2f}'.format(**info), end="")

        if return_info:
            return info

    @st.cache(max_entries=1)
    def adjust_intensity(array):
        return bimef(array[:,:,[2,1,0]])

    fImage = st.sidebar.file_uploader("Upload image file:")

    col1, col2 = st.columns(2)

    if fImage is not None:

        input_file_name = str(fImage.__dict__['name'])
        input_file_ext = '.' + str(input_file_name.split('.')[-1])
        input_file_basename = input_file_name.replace(input_file_ext, '')
        np_array = np.frombuffer(fImage.getvalue(), np.uint8) 
        image_np = cv2.imdecode(np_array, cv2.IMREAD_COLOR)       

        with col1:        
            st.header(f'Original Image')
            st.image(image_np[:,:,[2,1,0]])

            input_file_name = st.text_input('Download Original Image As', input_file_name)
    
            button = st.button("Download Original Image")
            if button:
                img.imsave(input_file_name, image_np[:,:,[2,1,0]])
                st.text("Download complete")

        process_time=0.
        with st.spinner(text="Enhancing image ..."):
            start = time.time()
            image_np_ai = adjust_intensity(image_np)
            end = time.time()
            process_time = end - start
            st.sidebar.text(f'Processing time: {process_time:.5f} s')

        processed_file_name = input_file_basename + '_AI' + input_file_ext
        with col2:
            st.header(f'Enhanced Image')
            #st.image(normalize_array(image_np_ai))
            st.image(image_np_ai, clamp=True)
        
            output_file_name = st.text_input('Download Enhanced Image As', processed_file_name)


            button_ai = st.button("Download Enhanced Image", 'adjusted_intensity')
            if button_ai:
                img.imsave(output_file_name, image_np_ai)
                st.text("Download complete")

        st.text('\n\n\n\n\n\n\n\n')
        st.text('*Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
        image_np_info = array_info(image_np, print_info=False, return_info=True)
        image_np_info_str = '\n'.join(sorted({f'{key}: {value}' for key, value in image_np_info.items()}))

        r = process_time / image_np_info['size']
        st.sidebar.text(f'{r*1000000:.5f} microseconds / pixel')

        image_np_ai_info = array_info(image_np_ai, print_info=False, return_info=True)
        image_np_ai_info_str = '\n'.join(sorted({f'{key}: {value}' for key, value in image_np_ai_info.items()}))
 
        st.sidebar.text("..................................")
        st.sidebar.text("..................................")
        st.sidebar.text("Pixel Statistics [Original Image]:")
        st.sidebar.text("..................................\n")
        st.sidebar.text(image_np_info_str)
        st.sidebar.text("\n..................................")
   
        st.sidebar.text("\n\n\n\n\n")
        st.sidebar.text("..................................")
        st.sidebar.text("..................................")
        st.sidebar.text("Pixel Statistics [Enhanced Image]:")
        st.sidebar.text("..................................\n")
        st.sidebar.text(image_np_ai_info_str)
        st.sidebar.text("\n..................................")

if __name__ == '__main__':

    run_app()