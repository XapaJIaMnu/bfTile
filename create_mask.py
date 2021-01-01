#!/usr/bin/env python3

def create_repetitive_mask(start, repeat):
    arr = ['0']*64
    i = start
    while i < len(arr):
        arr[i] = '1'
        i = i+repeat
    print(hex(eval('0b' + ("".join(arr))[::-1])))

def create_single_mask(start, end, width):
    arr = ['0']*width
    i = start
    while i < end:
        arr[i] = '1'
        i = i+1
    print(hex(eval('0b' + ("".join(arr))[::-1])))

def permute_mask(): # for column 1
    # This is for permute_ps because one float = 4 int 8ths which is actually what we need
    arr = ['1','0','0','0','1','1','0','1']
    print(hex(eval('0b' + ("".join(arr))[::-1])))

def permute_mask2(): # for column 2
    # This is for permute_ps because one float = 4 int 8ths which is actually what we need
    arr = ['1','1','0','1','0','0','1','0']
    print(hex(eval('0b' + ("".join(arr))[::-1])))

def permute_mask3(): # for column 3
    # This is for permute_ps because one float = 4 int 8ths which is actually what we need
    arr = ['0','1','1','1','1','0','0','0']
    print(hex(eval('0b' + ("".join(arr))[::-1])))
