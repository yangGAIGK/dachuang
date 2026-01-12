import os

def rename_images_in_folder(folder_path):
    """
    将指定文件夹内所有 xx-yy.jpg 格式的图片重命名为 G1_xx_yy.jpg

    :param folder_path: 图片所在文件夹的路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以 .jpg 结尾
        if filename.endswith('.tiff'):
            # 分割文件名和扩展名
            name, ext = os.path.splitext(filename)
            
            # 检查文件名是否符合 xx-yy 的格式（即包含一个连字符 '-'）
            if '-' in name:
                # 用连字符分割字符串
                parts = name.split('-')
                
                # 确保分割后正好是两部分
                if len(parts) == 2:
                    xx, yy = parts
                    # 构建新的文件名
                    new_filename = f"G_{xx}_{yy}{ext}"
                    
                    # 构建旧文件和新文件的完整路径
                    old_path = os.path.join(folder_path, filename)
                    new_path = os.path.join(folder_path, new_filename)
                    
                    # 执行重命名
                    os.rename(old_path, new_path)
                    print(f"已重命名: {filename} -> {new_filename}")

# ------------------- 请修改这里 -------------------
# 将下面的路径替换为你的图片文件夹的实际路径
# 在Windows上，路径可能是这样的: 'C:\\Users\\YourUser\\Pictures\\MyFolder'
# 在macOS或Linux上，路径可能是这样的: '/home/user/images'
target_folder = 'D:\Study\大三上\science\大创\等温-RAW\RAW\zhaodu25'
# --------------------------------------------------

if __name__ == "__main__":
    if os.path.isdir(target_folder):
        rename_images_in_folder(target_folder)
        print("\n所有符合条件的图片已完成重命名！")
    else:
        print(f"错误：文件夹 '{target_folder}' 不存在。请检查路径是否正确。")