#!/usr/bin/env python3
#
import time
import argparse

from utils.sockets.shell_serversocket import ShellServerSocket
# from utils import helper_camera as cam


# ----------------
# --- Methods  ---
# ----------------
# Server socket welcome method.
# Put code to be executed in new socket connections.
# It returns a welcome message
def hw_welcome(server_sock, sock):
	return "Welcome to Raspberry Pi chat server."


def hw_close_socket(server_sock, client_sock, msg):
	server_sock.close(client_sock)
	return False


def hw_picam_start(server_sock, sock, msg):
	# global picam, picam_conf
	server_sock.put_msg(sock, "Initializing remote camera..")
	# picam = cam.PICameraServer()
	
	# if picam.init_camera(conf=picam_conf, stream=server_sock.streams[0].c_fstream) == 'idle':
	# 	server_sock.put_msg(sock, "- Remote camera initialized")
	# 	server_sock.put_msg(sock, "- Switching to capture mode..")
	# 	# set caller
	# 	picam.socket = sock
	#
	# 	# start a new thread
	# 	picam.start()
	
	# while picam.state != 'streaming':
	# 	# wait until camera starts steaming
	# 	time.sleep(1)
	
	server_sock.put_msg(sock, "- Capture started")
	# server_sock.put_msg(sock, "Remote camera state: {}".format(picam.state))
	# server_sock.put_msg(sock, cam.get_camera_settings(picam))
	return True


# Main code
def main():
	# Start server in private mode in the main thread
	tstamp = time.time()
	myServer = ShellServerSocket("", port, False)
	myServer.welcome = hw_welcome
	myServer.set_commands({
		"start": hw_picam_start,
		"quit": hw_close_socket
	})
	myServer.init("Shell Server")
	if myServer.active:
		myServer.run()
	
	print("Bye..")


# ----------------
# ---   MAIN   ---
# ----------------

# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", type=int, default=5501, help="port number")
args = ap.parse_args()
port = args.port
tstamp = time.time()

if __name__ == "__main__":
	# execute only if run as a script
	main()
