import os


# 获得根路径
def getRootPath():
    # 获取文件目录
    curPath = os.path.abspath(os.path.dirname(__file__))
    # 获取项目根路径，内容为当前项目的名字
    rootPath = curPath[:curPath.find("yolov5-pytorch-v1\\") + len("yolov5-pytorch-v1\\")]
    rootPath = rootPath.replace('\\', '/').rsplit('/', 1)[0]
    return rootPath
