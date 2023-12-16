import os
import subprocess


class library:
    def __init__(self, name: str, url: str, commit: str) -> None:
        self.name = name
        self.url = url
        self.commit = commit

    def clone(self, dir: str) -> None:
        print( "========== clone " + self.name + " ==========" )
        subprocess.call( ['git', 'clone', self.url, self.name], cwd=dir )
        subprocess.call( ['git', 'checkout', self.commit, '-q'], cwd=os.path.join(dir, self.name) )


if __name__ == '__main__':
    externalsDir = os.path.dirname(__file__)

    libs = [
        library('stb', 'https://github.com/nothings/stb.git', 'af1a5bc352164740c1cc1354942b1c6b72eacb8a'),
        library('tinygltf', 'https://github.com/syoyo/tinygltf.git', 'aaf631c984c6e725573840c193ca2ff0ea216e7b'),
        library('tinyply', 'https://github.com/ddiakopoulos/tinyply.git', 'e5d969413b8612de31bf96604c95bf294d406230'),
        library('vulkan', 'https://github.com/KhronosGroup/Vulkan-Headers.git', '2b55157592bf4c639b76cc16d64acaef565cc4b5'),
        library('imgui', 'https://github.com/ocornut/imgui.git', 'f8c768760b0746aa7c1652397e2d9234c8502cb1'),
        library('glfw', 'https://github.com/glfw/glfw.git', '3fa2360720eeba1964df3c0ecf4b5df8648a8e52'),
    ]
    for lib in libs:
        lib.clone(externalsDir)

    print("========== build glfw ==========")
    glfwBuildDir = os.path.join(externalsDir, 'glfw', 'build')
    subprocess.run( ' '.join(['mkdir', glfwBuildDir]), shell=True)
    subprocess.run( ' '.join(['cmake', '-B', glfwBuildDir, '-S', os.path.join(externalsDir, 'glfw'), '-DBUILD_SHARED_LIBS=ON']), shell=True)
    subprocess.run( ' '.join(['cmake', '--build', glfwBuildDir, '--config', 'Release']), shell=True)
    subprocess.run( ' '.join(['cmake', '--build', glfwBuildDir, '--config', 'Debug']), shell=True)
