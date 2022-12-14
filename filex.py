#!/usr/bin/python3

import os
import inspect
import importlib
import concurrent.futures
import signal

def outdated(target, *deps):
    '''OUTDATED - Checks whether a target file is outdated wrt dependencies
    OUTDATED(target, dep1, dep2, ...) returns True if any of the files DEP1,
    DEP2, etc., have been modified more recently than the file TARGET, or
    if the target file does not exist, else False.
    A caution is printed for any dependency that doesn't exist, but that does
    not cause a positive "outdated" result.'''
    if not os.path.exists(target):
        return True
    t_target = os.stat(target).st_mtime
    for dep in deps:
        if os.path.exists(dep):
            t_dep = os.stat(dep).st_mtime
            if t_dep > t_target:
                return True
        else:
            print(f"Caution: Dependency {dep} does not exist for {target}")
    return False
            


def mkpath(path):
    if os.path.isdir(path):
        return
    parent = os.path.dirname(path)
    if parent != "":
        mkpath(parent)
    os.mkdir(path)

_lastmodule = None
def M():
    return _lastmodule
    
def _buildone(name, target=None, source=None, sources=None,
              script=None, module=None, function=None, extrax=[]):
    if source is None:
        deps = sources
    else:
        deps = [source]
    if module is not None:
        deps.append("code/" + module + ".py")
    if not outdated(target, *deps):
        return
                              
    print("  Building", target)
    mkpath(os.path.dirname(target))

    if script is not None:
        import runpy
        runpy.run_path(script)
        return
    
    if module is not None:
        global _lastmodule
        _lastmodule = importlib.import_module(module)

    if function is not None:
        if source is None:
            function(target, sources, *extrax)
        else:
            function(target, source, *extrax)

class _ImmediateExecutor:
    def __init__(self, max_workers=None):
        pass
    def __enter__(self):
        return self
    def __exit__(self, typ, val, traceback):
        return False
    def submit(self, foo, *args, **kwargs):
        print("immexe", args)
        foo(*args, **kwargs)

def buildstep(name, target=None, source=None, sources=None,
              script=None, module=None, function=None, scope=None,
              thread=True):
    print("Build step:", name)

    def signal_handler(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("Interrupted")
        os._exit(5) 
    signal.signal(signal.SIGINT, signal_handler)
    
    if target is None:
        target = name
    if (source is None) == (sources is None):
        raise Exception("Either source or sources must be given")
    if (script is None) == (function is None):
        raise Exception("Either script or function must be given")
    if (script is not None) and (scope is not None):
        raise Exception("You cannot give a scope to a script")
    if (script is not None) and (module is not None):
        raise Exception("You cannot load a module for a script")

    if function is None:
        nargin = 0
    else:
        nargin = len(inspect.signature(function).parameters)
        
    if scope is None:
        # target should be a simple string
        # sources should be a list or source should be a string
        _buildone(name, target, source, sources,
                  script, module, function)
        return

    if type(scope) != tuple:
        scope = scope,

    if thread:
        execu = concurrent.futures.ThreadPoolExecutor
        if type(thread)==bool:
            thread = None
    else:
        execu = _ImmediateExecutor
    with execu(max_workers=thread) as exe:
        tasks = []
        if len(scope)==1:
            if source is None:
                source = lambda x: None
            else:
                sources = lambda x: None
            for x in scope[0]:
                if nargin==3:
                    extrax = [x]
                else:
                    extrax = []
                print("submit", x)
                tasks.append(exe.submit(_buildone, name, target(x),
                                        source(x), sources(x),
                                        script,
                                        module, function, extrax))
        elif len(scope)==2:
            if source is None:
                source = lambda x, y: None
            else:
                sources = lambda x, y: None
            for x in scope[0]:
                for y in scope[1](x):
                    if nargin==4:
                        extrax = [x, y]
                    else:
                        extrax = None
                    print("submit", x, y)
                    tasks.append(exe.submit(_buildone, name, target(x,y),
                                            source(x,y), sources(x,y),
                                            script,
                                            module, function, extrax))
        else:
            raise Exception("Only two levels of scope allowed")
