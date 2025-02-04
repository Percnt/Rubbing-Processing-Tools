from PIL import Image
import numpy as np
from scipy.ndimage import median_filter

def remove_noise_and_invert(image_path, output_path, filter_size=1):

    # 打开图片文件
    with Image.open(image_path) as img:

        # 将图片转换为灰度模式，以便简单滤波
        gray_img = img.convert('L')

        # 将图片转换为numpy数组以应用中值滤波
        img_array = np.array(gray_img)

        # 应用中值滤波去除噪点
        filtered_array = median_filter(img_array, size=filter_size)

        # 创建一个新的Image对象从过滤后的数组
        filtered_img = Image.fromarray(filtered_array)

        # 对像素颜色取反（黑白颠倒）
        inverted_img = Image.eval(filtered_img, lambda x: 255 - x)

        # 保存新的图片
        inverted_img.save(output_path)
        print(f"Processed image saved to {output_path}")

# 使用函数
image_path = r'C:\Users\24078\Desktop\項目\D Python预处理程序\输入文件夹\2-7-3唐故安国寺清源律师墓志.png'# 输入图片的路径
output_path = r'C:\Users\24078\Desktop\項目\D Python预处理程序\输出文件夹\2-7-3唐故安国寺清源律师墓志（降噪1取反）.png'# 输出图片的路径
remove_noise_and_invert(image_path, output_path)