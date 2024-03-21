""" Module utils/io/output.py (Author: Charley Zhang, 2021)

Updates Log:
  - 2022.02: added rank argument to suppress printing for distributed train.
"""

import os, sys
import warnings
import logging
import textwrap


OUT = set(('print', 'warning', 'log_debug', 'log_info', 'log_warning',
           'log_error', 'log_critical'))
# OUT_STREAM = {
#     'print': print,
#     'warning': warnings.warn,
#     'log_debug': logging.debug,
#     'log_info': logging.info,
#     'log_warning': logging.warning,
#     'log_error': logging.error,
#     'log_critical': logging.critical,
# }

class HidePrint:
    def __init__(self, hide=True):
        self._hide = hide
        
    def __enter__(self):
        if self._hide:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._hide:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def header_one(message, width=80, out='print', rank=0):
    if rank != 0:
        return
    divider = '▓' * width
    if len(message) > width - 4:
        center_message = wrap_and_center(message, width=width-4)
    else:
        center_message = ('☯ ' + message + ' ☯').center(width, ' ')
        # center_message = '⛩️⛩️⛩️ ' + message + ' ⛩️⛩️⛩️'
    log('\n\n' + divider + '\n' + center_message + '\n' + divider + '\n\n', out)
    

def header_two(message, width=80, out='print', rank=0):
    if rank != 0:
        return 
    divider = '━' * width
    if len(message) > width - 4:
        center_message = wrap_and_center(message, width=width-4)
    else:
        center_message = ('● ' + message + ' ●').center(width, ' ')
        # center_message = '⛩️⛩️⛩️ ' + message + ' ⛩️⛩️⛩️'
    log('\n\n' + divider + '\n' + center_message + '\n' + divider + '\n\n', out)
    
def header_three(message, width=80, out='print', rank=0):
    if rank != 0:
        return
    if len(message) > width - 6:
        messages = textwrap.fill(message, width - 6).split('\n')
        for i, msg in enumerate(messages):
            if i == 0 or i == len(messages) - 1:
                messages[i] = ('○ ' + messages[i] + ' ○').center(width, '⋯')
            else:
                messages[i] = messages[i].center(width, ' ')
        center_message = '\n'.join(messages)
    else:
        center_message = ('○ ' + message + ' ○').center(width, '⋯')
    log('\n' + center_message + '\n', out)

def subsection(message, width=80, out='print', rank=0):
    if rank != 0:
        return
    message = '✼ ' + message + ' ✼'
    underline = '┈' * min(width, len(message))
    message = textwrap.fill(message, width=width)
    message = message.replace('\n', '\n   ')
    log('\n' + message + '\n' + underline, out)
    
def subsubsection(message, width=80, out='print', rank=0):
    if rank != 0:
        return
    message = textwrap.fill(message, width=width)
    message = message.replace('\n', '\n   ')
    log('\n-[' + message + ']-\n', out)

def wrap_and_center(string, width=80):
    strings = textwrap.wrap(string, width)
    strings = [string.center(width, ' ') for string in strings]
    return '\n'.join(strings)

def log(message, out):
    out = parse_out(out)
    # OUT_STREAM[out](message)
    if out == 'print':
        print(message)
    elif out == 'warning':
        warnings.warn(message)
    elif out == 'log_debug':
        logging.debug(message)
    elif out == 'log_info':
        logging.info(message)
    elif out == 'log_warning':
        logging.warning(message)
    elif out == 'log_error':
        logging.error(message)
    elif out == 'log_critical':
        logging.critical(message)

def parse_out(out):
    out = out.lower()
    msg = f'Given out "{out}" is not valid. Choices: {OUT}.'
    assert out.lower() in OUT, msg
    return out


    
if __name__ == '__main__':
    strings = ('Step 1: Collect Training Components'*4, 
               'Step 1: Collect Training Components'*12, 
               'Step 2: Training'
    )
    
    for sep in (header_one, header_two, header_three, subsection, subsubsection):
        for string in strings:
            sep(string)