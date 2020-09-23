# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 15:49
# @Author  : Hou Jue
# @Email   : mat_wu@163.com
# @File    : Internet_20200923.py
# @Software: 廖雪峰

import socket
import threading
import time


s1=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s1.connect(('127.0.0.1',9999))
print(s1.recv(1024).decode('utf-8'))
for data in [b'Michael',b'Tracy',b'Sarah']:
    s1.send(data)
    print(s1.recv(1024).decode('utf-8'))
s1.send(b'exit')
s1.close()