from os.path import dirname, join
import subprocess
import platform

def clone( baseDir, name, url, commit ):
    repoDir = join( baseDir, name )
    print( "========== clone " + name + " ==========" )
    subprocess.call( ['git', 'clone', url, name], cwd=baseDir )
    subprocess.call( ['git', 'checkout', commit, '-q'], cwd=repoDir )


if __name__ == '__main__':
    externalsDir = dirname(__file__)

    clone(externalsDir, 'glm', 'https://github.com/g-truc/glm.git', '5c46b9c07008ae65cb81ab79cd677ecc1934b903')
    clone(externalsDir, 'stb', 'https://github.com/nothings/stb.git', 'af1a5bc352164740c1cc1354942b1c6b72eacb8a')
    clone(externalsDir, 'tinygltf', 'https://github.com/syoyo/tinygltf.git', 'aaf631c984c6e725573840c193ca2ff0ea216e7b')
    clone(externalsDir, 'vulkan', 'https://github.com/KhronosGroup/Vulkan-Headers.git', '2b55157592bf4c639b76cc16d64acaef565cc4b5')
    clone(externalsDir, 'glfw', 'https://github.com/glfw/glfw.git', '3fa2360720eeba1964df3c0ecf4b5df8648a8e52')

    print("========== build glfw ==========")
    osname = platform.platform()
    if osname.startswith('Windows'):
        subprocess.run( 'mkdir .\\glfw\\build', shell=True)
        subprocess.run( 'cmake -B glfw\\build -S glfw -DBUILD_SHARED_LIBS=ON', shell=True)
        subprocess.run( 'cmake --build glfw\\build --config Release', shell=True)
        subprocess.run( 'cmake --build glfw\\build --config Debug', shell=True)
    elif osname.startswith('Linux'):
        subprocess.run( 'mkdir ./glfw/build', shell=True)
        subprocess.run( 'cmake -B ./glfw/build -S ./glfw -DBUILD_SHARED_LIBS=ON', shell=True)
        subprocess.run( 'make -C ./glfw/build', shell=True)