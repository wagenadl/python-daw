#!/usr/bin/python

def protectedsplit(text, sep=" ", prot="'\"", esc="\\", warn=False, exc=False):
    '''PROTECTEDSPLIT - Split a string while respecting protected parts
    parts = PROTECTEDSPLIT(text) splits the TEXT at spaces, but not if
    those spaces occur inside of single or double quotes.
    Optional arguments:
      SEP - Specify separator characters (default: space)
      PROT - Specify protecting quoting characters (default single and
             double quotes).
      ESC - Character that escapes following special character (default:
            backslash). The character immediately following an ESC is
            not considered special.
      WARN - Set to True to receive warnings for unpaired protecting quotes.
      EXC - Set to True to raise exception for unpaired protecting quotes.
    To protect using parentheses and the like, put the opener in the PROT 
    string. The matching closer is inferred according to the following
    table:
      ( )
      [ ]
      { }
      < >
      “ ”
      ‘ ’
      « »
    Protecting quoting characters that do not appear in pairs in the
    source string cause the remainder of the text to be returned as 
    a single field. No warning is reported for this unless WARN is set,
    in which case a warning is printed.
'''
    fields = []
    start = 0
    L = len(text)
    idx = 0
    inprot = None
    protpairs = { '(': ')',
                  '[': ']',
                  '{': '}',
                  '<': '>',
                  '“': '”',
                  '‘': '’',
                  '«': '»' }
    while idx<L:
        c = text[idx]
        if c in esc:
            idx += 2
        else:
            if inprot is not None:
                if c==inprot:
                    inprot = None
            else:
                if c in prot:
                    if c in protpairs:
                        inprot = protpairs[c]
                    else:
                        inprot = c
                elif c in sep:
                    fields.append(text[start:idx])
                    start = idx + 1
            idx += 1
    if idx>L:
        if exc:
            raise ValueError('String terminates in escape character')
        elif warn:
            print('Warning: String terminates in escape character')
    if inprot is not None:
        if exc:
            raise ValueError('Unpaired quote character')
        elif warn:
            print('Warning: Unpaired quote character')
    return fields

                        
    
