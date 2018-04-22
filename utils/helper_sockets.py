import sys
import struct
import io
import socket
import select
import time
from threading import *
import datetime

import platform

# RaspberryPi B+ : excluded imports
if platform.machine() != 'armv6l':
    import cv2
    import numpy as np
    from PIL import Image
    from matplotlib import pyplot as plt
    from utils import helper_visuals as im


# Callback Function that does nothing
def nothing(*arg):
    pass


# Open server socket for dialog with incoming connections
# This socket is able to handle multiple client connections
# ------------------------------------------------------------------
# DialogServer class
class DialogServer:
    
    def __init__(self, port, private=True):
        self.port = port
        
        # self.srvsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # SocketDescriptor class extends socket.socket
        self.srvsock = SocketDescriptor(socket.AF_INET, socket.SOCK_STREAM)
        self.srvsock.c_name = "Server socket"
        self.srvsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srvsock.bind(("", port))
        self.srvsock.listen(2)
        
        self.descriptors = [self.srvsock]
        self.streams = []
        self.commands = {}
        self.private = private
        self.welcome = None
        
        print("DialogServer started on port {}\nPrivate mode is {}".format(port, self.private))
    
    def run(self):
        evt = 0
        while 1:
            # Await an event on a readable socket descriptor
            (sread, swrite, sexc) = select.select(self.descriptors, [], [])
            
            # Iterate through the tagged read descriptors
            for sock in sread:
                
                # Received a connect to the server (listening) socket
                if sock == self.srvsock:
                    self.accept_new_connection()
                else:
                    
                    # Received something on a client socket
                    # stin = sock.recv(100).decode()
                    stin = netCatch(sock)
                    
                    # Check to see if the peer socket closed
                    if stin == '':
                        self.close(sock)
                    else:
                        try:
                            host, port = sock.getpeername()
                        except:
                            print("[error]", sys.exc_info()[0], "Closing socket")
                            self.descriptors.remove(sock)
                            sock.close()
                            continue
                            
                        stin = stin.decode()
                        stout = '[%s:%s] %s' % (host, port, stin)
                        if stin.rstrip() == 'quit':
                            netThrow(sock, "quit")
                            self.close(sock)
                        elif stin.rstrip() == 'stream-listener':
                            # Move socket to streams list
                            self.descriptors.remove(sock)
                            self.streams.append(sock)
                            # Make a file-like object out of the connection
                            sock.c_fstream = sock.makefile('wb')
                            binThrow(sock.c_fstream, io.BytesIO(b"--EOS--"))
                            print("Stream socket request completed for {}:{}".format(host, port))
                        elif stin.rstrip() == 'close-stream-listener':
                            so = self.streams[0]
                            host, port = so.getpeername()
                            self.streams.remove(so)
                            # so.c_fstream.close()
                            binThrow(so.c_fstream, io.BytesIO(b"quit"))
                            # time.sleep(5)
                            netThrow(sock, "--EOS--")
                            # time.sleep(5)
                            stout = 'Stream socket closed %s:%s\n' % (host, port)
                            self.broadcast_string(stout, sock)
                            # so.close()
                        else:
                            self.broadcast_string(stout, sock)
                            stcmd = stin.rstrip().split(' ')
                            cmd = self.commands.get(stcmd[0])
                            if (cmd):
                                print("Executing command '{}' for {}:{}".format(cmd.__name__, host, port))
                                cmd(self, sock, stcmd)
                            
                            netThrow(sock, "--EOS--")
    
    def close(self, sock):
        host, port = sock.getpeername()
        stout = 'Client left %s:%s\n' % (host, port)
        self.broadcast_string(stout, sock)
        self.descriptors.remove(sock)
        sock.close()
    
    def broadcast_string(self, string, omit_sock):
        if not self.private:
            for sock in self.descriptors:
                if sock != self.srvsock and sock != omit_sock:
                    netThrow(sock, string)
        print("{}".format(string.rstrip()))
    
    def accept_new_connection(self):
        so, (remhost, remport) = self.srvsock.accept()
        # The .accept() method returns a socket.socket object
        # thus we need to cast it into a SocketDescriptor class
        # by invoking it's .copy() static method we defined.
        newso = SocketDescriptor.copy(so)
        newso.c_name = "Client {}:{}".format(remhost, remport)
        self.descriptors.append(newso)
        # Close the original socket for safety.
        so.close()
        
        netThrow(newso, self.welcome())
        stout = 'Client joined %s:%s\n' % (remhost, remport)
        self.broadcast_string(stout, newso)
    
    def set_commands(self, cmd_list):
        self.commands = cmd_list
    
    def append_command(self, cmd):
        self.commands.update(cmd)


class DialogClient(Thread):
    
    def __init__(self, host, port, hdl_terminate):
        Thread.__init__(self)
        self.status = "init"
        self.host = host
        self.port = port
        self.socket = socket.socket()
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.terminate = hdl_terminate

    def init_socket(self, confirm):
        try:
            self.socket.connect((self.host, self.port))
            if confirm:
                data = netCatch(self.socket)
                if data:
                    print(data.decode())
            self.start()
        except ConnectionRefusedError as SRE:
            self.status = 'failed'
            print("[error]:", SRE)

        return self.status

    def run(self):
        self.status = 'running'
        while True:
            data = netCatch(self.socket)
            if data:
                data = data.decode()
                if data == '--EOS--':
                    self.status = "running"
                elif data == 'quit':
                    self.close()
                    break
                elif data == 'pause':
                    self.status = 'paused'
                    break
                else:
                    print(" ", data)
            else:
                self.close()
                self.status = 'init'
                # pass control to the custom terminate handler
                self.terminate()
                break
                
    # This method causes sync communication to the caller
    #   if the wait param is set to true.
    def send(self, msg, wait=False):
        netThrow(self.socket, msg)
        if wait:
            self.status = "working"
            while self.status == 'working':
                time.sleep(1)
    
    def close(self):
        self.socket.close()
        self.status = "init"

class StreamClient(Thread):
    
    def __init__(self, host, port, hdl_terminate):
        Thread.__init__(self)
        self.status = "init"
        self.pipe = None
        self.consumer = None
        self.host = host
        self.port = port
        self.socket = socket.socket()
        # self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.terminate = hdl_terminate

    def init_socket(self, confirm):
        try:
            self.socket.connect((self.host, self.port))
            if confirm:
                self.status = 'negotiate'
                data = netCatch(self.socket)
                if data:
                    netThrow(self.socket, "stream-listener")
            # Make a file-like object
            self.pipe = self.socket.makefile('rb')
            self.start()
        except ConnectionRefusedError as SRE:
            self.status = 'failed'
            print("[error]:", SRE)
    
        return self.status
    
    def run(self):
        while True:
            (ln, data) = binCatch(self.pipe)
            if self.status == 'purge':
                # consume all data from the pipe
                self.pipe.flush()
                print("[info] (StreamClient.run) : Video stream purged")
                continue

            # print("  ", self.status, ln, data)
            if data:
                if self.status == 'negotiate':
                    data = data.getvalue().decode()
                    if data == '--EOS--':
                        self.status = "running"
                    elif data == 'quit':
                        break
                    else:
                        print(" ", data)
                else:
                    self.consumer(self, ln, data)
            else:
                self.close()
                # pass control to the custom terminate handler
                # self.terminate()
                break

        self.status = "init"
        print("Stream socket stopped")

    def set_consumer(self, consumer):
        self.consumer = consumer

    def close(self):
        self.status == 'purge'
        self.pipe.flush()
        self.pipe.close()
        self.socket.close()
        self.status = "init"
        print("Stream socket closed")

    def purge_negotiate(self):
        self.status = 'purge'

# SocketDescriptor extends class socket.socket
#   It introduces new attributes and a .copy() static method
#   for creating new class instances from original socket objects.
#   Since .accept() method returns an original socket obj, we can use
#   the .copy() method to cast it into a SocketDescriptor
#   Another way would be to override .accept()
class SocketDescriptor(socket.socket):
    def __init__(self, *args, **kwargs):
        super(SocketDescriptor, self).__init__(*args, **kwargs)
        self.c_name = "undefined"
        self.c_state = 'idle'
        self.c_created = time.time()
        self.c_idle_time = 0
        self.c_fstream = None
    
    @classmethod
    def copy(cls, sock):
        fd = socket.dup(sock.fileno())
        copy = cls(sock.family, sock.type, sock.proto, fileno=fd)
        copy.settimeout(sock.gettimeout())
        return copy


# Open a socket listening for connections on address = (interface-ip port)
#    (0.0.0.0 port) means all interfaces in that port
def startListener(address):
    print("> Starting listener socket {0}.\n  Listening at {1} {2}".format(socket.gethostbyname(socket.gethostname()),
                                                                           address[0], address[1]))
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(address)
    sock.listen(1)
    return sock


# Send message through a socket
#   using a <stream-length> header of 4 bytes
def netThrow(sock, message):
    try:
        sock.send("{:04}".format(len(message)).encode())
        sock.send(message.encode())
    except socket.error as SE:
        print("Connection issue. Sending message failed:\n", SE)


# Retrieve a message from a socket
#   using a <stream-length> header of 4 bytes
def netCatch(sock):
    try:
        ln = sock.recv(4)
        data = sock.recv(int(ln.decode()))
        return data
    except socket.error as SE:
        print("[error]: connection failed!")
        return False
        # raise
    except:
        data = ln + sock.recv(100)
        print("[error]: protocol failed: ", sys.exc_info()[0], "Invalid data from client")
        return data
        # raise


# Send binary data through a file like object connector
#   using a <stream-length> header of type 32-bit unsigned int
def binThrow(conn, stream):
    try:
        stream_len = stream.seek(0, 2)
        conn.write(struct.pack('<L', stream_len))
        # conn.flush()
        stream.seek(0)
        conn.write(stream.read())
        conn.flush()
    except:
        print("Connection issue. Sending message failed:\n", sys.exc_info()[0])


# Retrieve a message from a socket
#   using a <stream-length> header of 4 bytes
def binCatch(conn):
    try:
        stream = io.BytesIO()
        stream_len = struct.unpack('<L', conn.read(struct.calcsize('<L')))[0]
        stream.write(conn.read(stream_len))
        stream.seek(0)
        return stream_len, stream
    except:
        print("[error] (binCatch) : stream failed: ", sys.exc_info()[0])
        return 0, None


# Sends a string message and gets back an answer
def netAsk(sock, msgNo, str):
    # Send message
    netThrow(sock, str)
    print("[{0}]({1}): {2}".format(socket.gethostname(), msgNo, str))
    # Receive answer
    data = netCatch(sock)
    print("[{0}]({1}): {2}".format(sock.getpeername()[0], msgNo, data.decode('UTF-8')))
    # Return answer received
    return data.decode('UTF-8')


# Listens for message and sends back
def netReply(sock, msgNo, str):
    # Receive message
    data = netCatch(sock)
    print("[{0}]({1}): {2}".format(sock.getpeername()[0], msgNo, data.decode('UTF-8')))
    # Sends reply
    netThrow(sock, str)
    print("[{0}]({1}): {2}".format(socket.gethostname(), msgNo, str))
    # Return message received
    return data.decode('UTF-8')

