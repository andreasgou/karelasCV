import sys
import time

# animated cursor
char_ani = ['|', '/', '-', '\\']
char_pos = 0
char_run = True

def waitcursor(run='idle'):
	global char_pos, char_ani, char_run
	if run != 'idle':
		char_run = run == 'yes'
	if char_run:
		sys.stdout.write("\r[{}]> ".format(char_ani[char_pos % 4]))
		sys.stdout.flush()
		char_pos = (char_pos % 4) + 1
