#!/usr/bin/python3

import os

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
            
