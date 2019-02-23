#!/usr/bin/env python3
#
import argparse

import time

from utils.sockets.shell_clientsocket import ShellClientSocket


# ----------------
# --- Classes  ---
# ----------------
# Formatter subclass for argparse. It is used to combine functionality of both classes defined.
class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.MetavarTypeHelpFormatter):
	pass


# Quit connection action
def hw_quit():
	print("Press any key to quit!")


# ----------------
# --- Methods  ---
# ----------------
def main():
	# Start server in private mode in the main thread
	hostso = ShellClientSocket(host, port, hw_quit)
	hostso.set_commands({
	})
	if hostso.init("Client Socket", True) == 'failure':
		print("Quiting program!")
		quit()

	# Enter interactive console mode.
	# Type 'quit' to exit
	q = ""
	while q.lower() not in {'quit'}:
		q = input("> ")
		if q.rstrip() != '':
			# send and wait for reply
			hostso.send(q, wait=(q.rstrip() != 'quit'))
	
	print("Quiting..")
	hostso.close(hostso.socket, remote=True)
	print("Bye!")


# ----------------
# ---   MAIN   ---
# ----------------

# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser(
	description="PiCam computer vision remote camera",
	formatter_class=MyFormatter)
ap.add_argument("--host", type=str, required=True, help="host address")
ap.add_argument("--port", type=int, required=True, help="port number")
args = ap.parse_args()

host = args.host
port = args.port

if __name__ == "__main__":
	# execute only if run as a script
	main()

