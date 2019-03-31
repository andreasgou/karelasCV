import sys

# animated cursor
char_ani = ['|', '/', '-', '\\']
char_pos = 0

def waitcursor():
	global char_pos, char_ani
	sys.stdout.write("{}\b".format(char_ani[char_pos % 4]))
	sys.stdout.flush()
	char_pos = (char_pos % 4) + 1
