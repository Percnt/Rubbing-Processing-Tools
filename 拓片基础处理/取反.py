from PIL import Image

def invert_image(image_path, output_path):

    # 打开图片文件
    with Image.open(image_path) as img:

        # 将图片转换为RGB模式，以防图片不是RGB格式
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 获得图片尺寸
        width, height = img.size

        # 创建一个与原图一样大小的新图片
        inverted_img = Image.new('RGB', (width, height))

        # 获取原始图片的像素数据
        pixels = img.load()
        inverted_pixels = inverted_img.load()

        # 遍历所有像素并取反
        for x in range(width):
            for y in range(height):
                r, g, b = pixels[x, y]
                inverted_pixels[x, y] = (255 - r, 255 - g, 255 - b)

        # 保存新的图片
        inverted_img.save(output_path)
        print(f"Inverted image saved to {output_path}")

# 使用函数
image_path = r'C:\Users\24078\Desktop\項目\E Python预处理程序\输入文件夹\宋故朱学谕墓志拓片.jpg'  # 输入图片的路径
output_path = r'C:\Users\24078\Desktop\項目\E Python预处理程序\输出文件夹\宋故朱学谕墓志拓片（翻转）.jpg'     # 输出图片的路径
invert_image(image_path, output_path)