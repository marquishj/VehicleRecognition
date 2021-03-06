# -*- coding: utf-8 -*-
# @Time    : 2020/9/23 15:49
# @Author  : Hou Jue
# @Email   : mat_wu@163.com
# @File    : Internet_20200923.py
# @Software: 廖雪峰

import socket
import threading
import time

s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)

s.bind('127.0.0.1',9999)
s.listen(5)
print('Waiting for connection')


def tcplink(sock,addr):
    print('Accept new connection from %s:%s...' % addr)
    sock.send(b'Welcome')
    while True:
        data=sock.recv(1024)
        time.sleep(1)
        if not data or data.decode('utf-8')=='exit':
            break
        sock.send(('Hello, %s' % data).encode('utf-8'))
    sock.close()
    print('Connection from %s:%s closed.' % addr)


while True:
    sock, addr = s.accept()
    t = threading.Thread(target=tcplink, args=(sock, addr))
    t.start()


s1=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s1.connect('127.0.0.1',9999)
print(s1.recv(1024).decode('utf-8'))
for data in [b'Michael',b'Tracy',b'Sarah']:
    s1.send(data)
    print(s.recv(1024).decode('utf-8'))
s1.send(b'exit')
s1.close()