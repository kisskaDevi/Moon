set dir_path=%~dp0

python3 %dir_path%/clone.py -d %dir_path%/../dependences/libs

git clone --depth=1 https://github.com/KhronosGroup/glTF-Sample-Models.git %dir_path%/../dependences/model/glTF-Sample-Models