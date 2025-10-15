# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 19:35:38 2025

@author: hp
"""

data = [5, 10, 15, 20, 25]
window_size = 3
moving_avg = []

for i in range(len(data) - window_size + 1):
    avg = sum(data[i:i+window_size]) / window_size
    moving_avg.append(avg)

print( data)
print( moving_avg)
