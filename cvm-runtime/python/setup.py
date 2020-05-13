from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os

def config_cython():
    ret = []
    python_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(python_dir)
    path = os.path.join(python_dir, "cvm/_cython")
    for fn in os.listdir(path):
        if not fn.endswith(".pyx"):
            continue
        ret.append(Extension(
            "cvm._cy3.%s" % (fn[:-4]),
            [os.path.join(path, fn)],
            include_dirs=[os.path.join(root_dir, "include")],
            library_dirs=[os.path.join(root_dir, "build")],
            libraries=['cvm_runtime']))
    return cythonize(ret, compiler_directives={"language_level": 3})


setup(name='cvm',
      description="CVM: A Deterministic Inference Framework for Deep Learning",
      ext_modules = config_cython())
