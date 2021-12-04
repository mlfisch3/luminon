from bimef import bimef, entropy, normalize_array
import cv2
import numpy as np
from matplotlib import image as img
import streamlit as st
import time

st.set_page_config(page_title="Luminon", layout="wide")

def run_app(default_granularity=0.3, default_power=0.5, default_sensitivty=0.5):

    @st.cache(max_entries=1)
    def array_info(array, print_info=True, return_info=False, return_info_str=False):

        array = array
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

        out = []
        if return_info:
            out.append(info)
        if return_info_str:
            info_str = f'shape: {info["shape"]}\n'
            info_str += f'size: {info["size"]}\nnonzero: {info["nonzero"]}  ({info["pct"]:.4f} %)\n'
            info_str += f'min: {info["min"]}    max: {info["max"]}\n'
            info_str += f'mean: {info["mean"]:.4f}    std: {info["std"]:.4f}\n'
            if array.ndim > 2:
                info_str += f'layer_variation: {info["layer_variation"]:.4f}\n'

            info_str += f'entropy: {info["entropy"]:.4f}\n'

            out.append(info_str)

        return out

    @st.cache(max_entries=1)
    def adjust_intensity(
                         array, 
                         exposure_ratio=-1, enhance=0.5, 
                         a=-0.3293, b=1.1258, lamda=0.5, 
                         sigma=5, scale=0.3, sharpness=0.001, 
                         dim_threshold=0.5, dim_size=(50,50), 
                         solver='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, 
                         clip=False, normalize=True, nbins=100, lo=1, hi=7, npoints=20
                         ):
        
        enhanced = bimef(
                         array[:,:,[2,1,0]], 
                         exposure_ratio=exposure_ratio, enhance=enhance, 
                         a=a, b=b, lamda=lamda, 
                         sigma=sigma, scale=scale, sharpness=sharpness, 
                         dim_threshold=dim_threshold, dim_size=dim_size, 
                         solver=solver, CG_prec='ILU', CG_TOL=CG_TOL, LU_TOL=LU_TOL, MAX_ITER=MAX_ITER, FILL=FILL, 
                         clip=clip, normalize=normalize, nbins=nbins, lo=lo, hi=hi, npoints=npoints
                         ) 

        return (enhanced * 255).astype(np.uint8)


    enhancement_granularity = float(st.sidebar.text_input('Enhancement Granularity   (default = 0.3)', str(default_granularity)))
    enhancement_power = float(st.sidebar.text_input('Enhancement Strength     (default = 0.5)', str(default_power)))
    enhancement_sensitivity = float(st.sidebar.text_input('Enhancement Sensitivity   (default = 0.5)', str(default_sensitivty)))

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
            ext = '.' + input_file_name.split('.')[-1]
            image_np_binary = cv2.imencode(ext, image_np)[1].tobytes()

            button = st.download_button(label = "Download Original Image", data = image_np_binary, file_name = input_file_name, mime = "image/png")

        #process_time=0.
        #with st.spinner(text="Enhancing image ..."):
        start = time.time()
        image_np_ai = adjust_intensity(image_np, scale=enhancement_granularity, enhance=enhancement_power, lamda=enhancement_sensitivity)
        end = time.time()
        process_time = end - start
        #st.sidebar.text(f'Processing time: {process_time:.5f} s')
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

        entropy_change_abs = image_np_ai_info['entropy'] - image_np_info['entropy']
        entropy_change_rel = (image_np_ai_info['entropy'] / image_np_info['entropy']) - 1.0

        st.sidebar.text(f'entropy change: {entropy_change_abs:.4f} ({entropy_change_rel * 100.0:.4f} %)\n')        
        
        st.sidebar.text("Pixel Statistics [Original Image]:")
        
        st.sidebar.text(image_np_info_str)
        
        st.sidebar.text("\n\n\n\n\n")
        
        st.sidebar.text("Pixel Statistics [Enhanced Image]:")

        st.sidebar.text(image_np_ai_info_str)


if __name__ == '__main__':

    run_app()