#!/usr/bin/python3

from .. import stringx
import numpy as np

def loadcsv(fn, sep=',', strip='"'):
    with open(fn, 'r') as fd:
        lines = fd.readlines()
        R = len(lines)
        if R==0:
            return None
        cells = []
        for line in lines:
            cells.append(stringx.protectedsplit(line, sep=sep))
        C = max([len(row) for row in cells])
        def isnumeric(x):
            if x.isnumeric():
                return True
            try:
                y = float(x)
                return True
            except:
                return False
        isnum = np.zeros((R,C), dtype=int) - 1
        # 0 means no, 1 means yes, -1 means missing
        for r in range(R): 
            for c in range(C): 
                if len(cells[r])>=C: 
                    isnum[r,c] = isnumeric(cells[r][c])
        colisnum = np.all(isnum[1:,:]!=0, 0)
        hdrisnum = isnum[0,:]
        havehdr = not np.all(np.logical_or(hdrisnum==colisnum, hdrisnum<0))
        if havehdr:
            fieldnames = cells.pop(0)
            R = R - 1
        else:
            fieldnames = list(np.range(C))
        columns = {}
        def tonumber(x):
            if x.isnumeric():
                return int(x)
            else:
                return float(x)
        for r in range(R):
            for c in range(C):
                if len(cells[r])>=C:
                    isnum[r,c] = cells[r][c].isnumeric()
        for c in range(C):
            col = []
            if colisnum[c]:
                for r in range(R):
                    if len(cells[r]) >= C:
                        col.append(tonumber(cells[r][c]))
                    else:
                        col.append(np.nan)
                col = np.array(col)
            else:
                for r in range(R):
                    if len(cells[r]) >= C:
                        col.append(cells[r][c])
                    else:
                        col.append("")
                if strip!="":
                    for r in range(R):
                        if len(col[r])>=2 and col[r][0] in strip:
                            col[r] = col[r][1:-1]
            columns[fieldnames[c]] = col
        return columns, fieldnames
    
                        
            
            
