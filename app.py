from bimef import bimef, entropy, xentropy, KL, normalize_array, array_info
import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
import time

st.set_page_config(page_title="Luminon", layout="wide")

def run_app(default_granularity=0.1, default_power=0.8, default_smoothness=0.3, 
            default_dim_size=(50), default_dim_threshold=0.5, default_a=-0.3293, default_b=1.258, default_exposure_ratio=-1):


    @st.cache(max_entries=1, show_spinner=False)
    def adjust_intensity(
                         array, 
                         exposure_ratio=-1, enhance=0.8, 
                         a=-0.3293, b=1.1258, lamda=0.3, 
                         sigma=5, scale=0.1, sharpness=0.001, 
                         dim_threshold=0.5, dim_size=(50,50), 
                         solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
                         lo=1, hi=7, npoints=20
                         ):
        
        enhanced = bimef(
                         array[:,:,[2,1,0]], 
                         exposure_ratio=exposure_ratio, enhance=enhance, 
                         a=a, b=b, lamda=lamda, 
                         sigma=sigma, scale=scale, sharpness=sharpness, 
                         dim_threshold=dim_threshold, dim_size=dim_size, 
                         solver=solver, CG_prec='ILU', CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL, 
                         lo=lo, hi=hi, npoints=npoints
                         ) 

        return (enhanced * 255).astype(np.uint8)


    fImage = st.sidebar.file_uploader("Upload image file:")

    granularity = float(st.sidebar.text_input('Resolution   (default = 0.1)', str(default_granularity)))
    power = float(st.sidebar.text_input('Power     (default = 0.8)', str(default_power)))
    smoothness = float(st.sidebar.text_input('Smoothness   (default = 0.3)', str(default_smoothness)))
    exposure_sample = int(st.sidebar.text_input('Sample   (default = 50)', str(default_dim_size)))
    sensitivity = float(st.sidebar.text_input('Sensitivity   (default = 0.5)', str(default_dim_threshold)))
    a = float(st.sidebar.text_input('Camera A   (default = -0.3293)', str(default_a)))
    b = float(st.sidebar.text_input('Camera B   (default = 1.1258)', str(default_b)))
    exposure_ratio = float(st.sidebar.text_input('Exposure Ratio   (default = -1 (auto))', str(default_exposure_ratio)))

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
            ext = '.' + input_file_name.split('.')[-1]
            image_np_binary = cv2.imencode(ext, image_np)[1].tobytes()

            button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")

        start = time.time()
        image_np_ai = adjust_intensity(image_np, exposure_ratio=exposure_ratio, scale=granularity, enhance=power, 
                                       lamda=smoothness, dim_size=(exposure_sample,exposure_sample), dim_threshold=sensitivity, a=a, b=b)
        end = time.time()
        process_time = end - start
        print(f'Processing time: {process_time:.5f} s')

        processed_file_name = input_file_basename + '_AI' + input_file_ext
        with col2:        
            st.header(f'Enhanced Image')
            st.image(image_np_ai, clamp=True)
        
            output_file_name = st.text_input('Download Enhanced Image As', processed_file_name)
            ext = '.' + output_file_name.split('.')[-1]
            image_np_ai_binary = cv2.imencode(ext, image_np_ai[:,:,[2,1,0]])[1].tobytes()

            button = st.download_button(label = "Download Enhanced Image", data = image_np_ai_binary, file_name = output_file_name, mime = "image/png")

        st.text('\n\n\n\n\n\n\n\n')
        st.text('*Supported file extensions: jpg, jpeg, png, gif, bmp, pdf, svg, eps')
        image_np_info, image_np_info_str = array_info(image_np, print_info=False, return_info=True, return_info_str=True)

        r = process_time / image_np.size
        print(f'{r*1000000:.5f} microseconds / pixel')

        image_np_ai_info, image_np_ai_info_str = array_info(image_np_ai, print_info=False, return_info=True, return_info_str=True)

        relative_entropy = xentropy(image_np, image_np_ai)
        kl_divergence = KL(image_np, image_np_ai)

        entropy_change_abs = image_np_ai_info['entropy'] - image_np_info['entropy']
        entropy_change_rel = (image_np_ai_info['entropy'] / image_np_info['entropy']) - 1.0

        st.sidebar.text(f'entropy change: {entropy_change_abs:.4f} ({entropy_change_rel * 100.0:.4f} %)')        
        st.sidebar.text(f'relative entropy: {relative_entropy:.4f}')
        st.sidebar.text(f'KL divergence: {kl_divergence:.4f}\n')
        
        st.sidebar.text("Pixel Statistics [Original Image]:")
        
        st.sidebar.text(image_np_info_str)
        
        st.sidebar.text("\n\n\n\n\n")
        
        st.sidebar.text("Pixel Statistics [Enhanced Image]:")

        st.sidebar.text(image_np_ai_info_str)


if __name__ == '__main__':

    run_app()