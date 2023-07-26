# python-daw
## Collection of Dr Wagenaar's python libraries

This repository is a collection of several python libraries I wrote to aid in general data analysis. Each of the modules is decently well documented internally. I am working on collating the documentation into a more accessible format.

## Build instructions

Most of the code can be used as-is, however, a few functions have C++ backends that must be compiled before use. You will need a C++ compiler such as [GCC](https://gcc.gnu.org/), [Visual Studio Community](https://visualstudio.microsoft.com/vs/community/), or [XCode](https://developer.apple.com/xcode/) and you will need the [SCons](https://scons.org/) build system. Instructions are [here](https://scons.org/doc/production/HTML/scons-user/ch01s02.html), but basically, you open a terminal and type:

    python -m pip install scons
    
On Linux you may prefer to use the system's package manager. E.g., in Debian or Ubuntu:

    sudo apt install scons
    
with SCons installed, compile the C++ backends:

    cd python-daw
    scons

After a little verbosity, that should print out “scons: done building targets.” If not, please report any errors.
