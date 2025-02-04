from PIL import Image, ImageEnhance, ImageChops
import numpy as np
from scipy import ndimage
import cv2


def find_black_pixel_block_boundaries(image, threshold=50, iterations=1):
    # 将PIL图像转换为灰度图像
    if image.mode != 'L':
        image = image.convert('L')
    pixels = np.array(image)

    # 将灰度图像转换为二值图像，黑色像素阈值
    binary = pixels <= threshold

    # 使用膨胀操作来扩展黑色区域的边界
    structure = np.ones((3, 3))  # 定义结构元素，这里使用3x3的正方形
    dilated = ndimage.binary_dilation(binary, structure, iterations=iterations)

    # 找到膨胀后的图像与原二值图像的差异，即边界
    boundaries = np.uint8(dilated) - np.uint8(binary)

    # 将边界转换回PIL图像
    boundary_image = Image.fromarray(boundaries * 255).convert('L')

    return boundary_image


def process_image(image, MIN_VALUE, grayscale, type='L', inverse=False):
    if len(image.shape) == 3:  # 判断是否为彩色图像
        # 转换为灰度图像
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    _, binary = cv2.threshold(gray_image, grayscale, 255, cv2.THRESH_BINARY)
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 使用原始二值图作为基础
    mask = binary.copy()

    # 过滤并绘制符合条件的轮廓
    for cnt in contours:
        if type == 'L':
            if cv2.arcLength(cnt, True) <= MIN_VALUE:  # 过滤掉周长小于阈值的轮廓
                cv2.drawContours(mask, [cnt], -1, 0, -1)  # 用黑色填充小轮廓
        if type == 'A':
            if cv2.contourArea(cnt) <= MIN_VALUE:
                cv2.drawContours(mask, [cnt], -1, 0, -1)  # 用黑色填充小轮廓
    if inverse:
        return 255 - mask
    else:
        return mask


def masking_1(image, binary_mask, v_point=60):
    # 确保二值图片和图A大小相同
    if binary_mask.size != image.size:
        binary_mask = binary_mask.resize(image.size)

    binary_mask = binary_mask.convert('L')

    # 创建一个提高亮度的图像
    # 这里简单地将图像转换为HSV，增加值（V）来提高亮度，然后再转换回RGB
    hsv_image = image.convert('HSV')
    h, s, v = hsv_image.split()
    v = v.point(lambda i: i + v_point)  # 增加亮度
    hsv_image = Image.merge('HSV', (h, s, v))

    # 将HSV图像转换回RGB
    brighter_image = hsv_image.convert('RGB')

    # 将二值图片作为蒙版，作用于图A
    # 在白色部分（255）提高亮度，在黑色部分（0）保持不变
    result_image = Image.composite(brighter_image, image, binary_mask)

    return result_image


def masking_2(image, mask, color='red'):
    mask = mask.convert('L')  # 确保蒙版是灰度模式

    # 检查尺寸是否一致
    if image.size != mask.size:
        raise ValueError("Image and mask must have the same size")

    # 创建一个与原图同样大小的红色图层
    red_layer = Image.new('RGB', image.size, color)

    # 将蒙版中的白色部分应用到红色图层上
    mask_inv = mask.point(lambda p: 255 if p == 0 else 0)  # 反转蒙版，黑色变白色，白色变黑色
    red_layer.paste(image, mask=mask_inv)

    # 合并原图和红色图层
    result = Image.composite(red_layer, image, mask)

    return result


if __name__ == '__main__':
    # 打开图像文件
    image = Image.open(r'C:\Users\24078\Desktop\項目\D Python预处理程序\输入文件夹\M6 墓志志石拓片.png') # 这个地方是输入图片的路径

    # 创建一个对比度增强器对象
    enhancer = ImageEnhance.Contrast(image)

    # 提高对比度，factor值大于1将增加对比度，小于1将减少对比度
    factor = 100  # 你可以根据需要调整这个值
    enhanced_image = enhancer.enhance(factor)

    # 检查图片模式并相应地转换
    if enhanced_image.mode == 'RGB':
        # 如果图片是RGB模式，转换为OpenCV的BGR模式
        cv2_image = cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)
    elif enhanced_image.mode == 'L':
        # 如果图片是灰度模式，直接转换为NumPy数组
        cv2_image = np.array(enhanced_image)
    elif enhanced_image.mode == 'RGBA':
        r, g, b, a = cv2.split(np.array(enhanced_image))
        rgb_image = cv2.merge((r, g, b))
        cv2_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    else:
        # 对于其他模式，需要根据需要进行处理
        # 例如，如果图片包含透明度通道（RGBA）可能需要先去除
        raise ValueError("Unsupported image mode. The image must be in RGB, RGBA or L mode.")

    # 处理图像,黑底True白底False,grayscale数值越大呈现字体部件越少
    boundary_img = process_image(cv2_image, 50, 0, type='L', inverse=True)

    # 显示边界图像
    if cv2_image.shape[2] == 3:  # 检查是否有颜色通道
        pil_image = Image.fromarray(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB))
    else:  # 如果是灰度图，直接转换
        pil_image = Image.fromarray(boundary_img)

    # 下面是输出的结果
    # pil_image.show()
    pil_image.save('输出结果/OP1.png')
    masking_1(ImageChops.invert(image), pil_image, 100).save('输出结果/OP2.png')
    masking_2(image, find_black_pixel_block_boundaries(masking_1(image, pil_image, -100), iterations=2),
              '#FFFFFF').save('输出结果/OP3.png')