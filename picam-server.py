#!/usr/bin/env python3
#
import time
import json
import glob
import argparse

from utils.sockets.dialogserversocket import DialogServer
from utils.sockets.helper_sockets import *
from utils import helper_camera as cam

# Local functions
# ---------------
def hw_listconn(dialog, sock, command):
    stout = "Active connections\n----------------------"
    netThrow(sock, stout)
    for so in dialog.descriptors:
        stout = "- {} {} {}".format(so.c_name, so.c_state, so.c_created, so.c_idle_time)
        netThrow(sock, stout)
    for so in dialog.streams:
        stout = "- {} {} {} (stream)".format(so.c_name, so.c_state, so.c_created, so.c_idle_time)
        netThrow(sock, stout)

def hw_video_start(dialog, sock, command=[]):
    global picam, picam_conf
    picam = cam.PICameraServer()
    netThrow(sock, "Initializing remote camera..")
    
    if picam.init_camera(conf=picam_conf, stream=dialog.streams[0].c_fstream) == 'idle':
        netThrow(sock, "- Remote camera initialized")
        netThrow(sock, "- Switching to capture mode..")
        # set caller
        picam.socket = sock

        # set debug mode
        # picam.plugins.append(hw_camera_debug)
        # print("Debug to client stream ENABLED")

        # start a new thread
        picam.start()
        
    while picam.state != 'streaming':
        # wait until camera starts steaming
        time.sleep(1)

    netThrow(sock, "- Capture started")
    netThrow(sock, "Remote camera state: {}".format(picam.state))
    netThrow(sock, cam.get_camera_settings(picam))

def hw_video_stop(dialog, sock, command=None):
    global picam
    if picam:
        picam.close_camera()
        stout = "Pi camera closed."
        # picam = None
    else:
        stout = "Pi camera not active."
    netThrow(sock, stout)

def hw_video_pause(dialog, sock, command=None):
    global picam
    if picam:
        if picam.state == "pause":
            picam.pause(False)
        elif picam.state == "streaming":
            picam.pause(True)
        netThrow(sock, "- Camera state: {}".format(picam.state))
    
def hw_video_command(dialog, sock, cmd):
    global picam, picam_conf
    ret = "Command '{}' executed.".format(cmd[0])
    while True:
        if (cmd[0] == 'start'):
            hw_video_start(dialog, sock, cmd)
            break
        if (cmd[0] == 'stop'):
            hw_video_stop(dialog, sock)
            break
        if (cmd[0] == 'pause'):
            hw_video_pause(dialog, sock)
            break
        if (cmd[0] == 'restart'):
            hw_video_stop(dialog, sock)
            hw_video_start(dialog, sock, cmd)
            break
        if (cmd[0] == 'load'):
            ret = cmd_loadset(cmd)
            break
        if (cmd[0] == 'save'):
            cmd_saveset(cmd)
            break
        if (cmd[0] == 'sets'):
            cmd_listsets(sock)
            break
            
        ret = "Video command '{}' not found!".format(cmd[0])
        break
        
    ret += "\nCamera state: " + (picam.state if picam else 'n/a')
    print(ret)
    netThrow(sock, ret)

def hw_video_setprop(dialog, sock, command):
    global picam, picam_conf
    isoffline = not(picam) or picam.state == 'closed'
    if len(command) > 1:
        cmd = command[1].split(' ')
        for prop in cmd:
            p = prop.split('=')
            if len(p) == 2:
                if isoffline:
                    ret = cam.set_camera_property(picam, p[0], p[1], False)
                else:
                    ret = cam.set_camera_property(picam, p[0], p[1], True)
                
    if not isoffline:
        netThrow(sock, cam.get_camera_settings(picam))

def hw_video_debug(dialog, sock, command):
    global picam, picam_conf
    st = ""
    if not picam:
        st = "Debug disabled. Pi camera not active."
    else:
        st = "usage: debug [on|off]"
        if len(command) > 1:
            cmd = command[1]
            if cmd == 'on':
                if hw_camera_debug not in picam.plugins:
                    picam.plugins.append(hw_camera_debug)
                    st = "Debug to client stream ENABLED"
            elif cmd == 'off':
                if hw_camera_debug in picam.plugins:
                    picam.plugins.remove(hw_camera_debug)
                    st = "Debug to client stream DISABLED"
    netThrow(sock, st)
    print(st)

def hw_camera_debug(cam):
    cam.log = "Frame [ {2} ]: Captured {1:.2f}MB in {0:.2f}s, " \
        .format(cam.streamLength/(1024 * 1024), time.time()-cam.stm, cam.frame_no)
    print(cam.log)
    netThrow(cam.socket, cam.log)
    
def cmd_saveset(cmd):
    global picam
    if not picam or picam.state == 'init':
        print("Cannot save set while camera is not running!")
    pinit, pimg, pdep, pview = cam.camera_settings(picam)
    props = {}
    props['sensor_mode'] = pinit['sensor_mode']
    props['framerate'] = float(pinit['framerate'])
    props['resolution'] = "{}x{}".format(pinit['resolution'].width, pinit['resolution'].height)
    props['zoom'] = pdep['zoom']
    props['drc_strength'] = pdep['drc_strength']
    props['file_name'] = pview['file_name']
    props['video_port'] = pview['video_port']
    props.update(pimg)
    fname = "camset-{}.json".format(cmd[1]) if len(cmd) > 1 else 'camera-settings.json'
    if len(cmd) > 1:
        props['file_name'] = fname
        picam.setfname = fname
    f = open(fname, 'w')
    json.dump(props, f, indent=4)
    print(json.dumps(props, indent=4))
    f.close()

def cmd_loadset(cmd):
    global picam, picam_conf
    fname = "camset-{}.json".format(cmd[1]) if len(cmd) > 1 else 'camera-settings.json'
    try:
        f = open(fname, 'r')
        picam_conf = json.load(f)
        if picam:
            picam.conf = picam_conf
            return "Camera set '{}' loaded. Restart video to activate.".format(fname)
        else:
            return "Camera set '{}' loaded. Start video to activate.".format(fname)
    except FileNotFoundError as FNF:
        return "[error] (cmd_loadset) : File not found '{}'".format(fname)
        
def cmd_listsets(sock, path=None):
    templates = sorted(glob.glob("camset*.json"), key=str.lower)
    for f in templates:
        print(f)
        netThrow(sock, f)

def hw_welcome():
    global picam_conf
    return "Connected to Pi camera server\nDefault set: %s " % picam_conf['file_name']

# Parse command line arguments
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", type=int, default=5501,
                help = "port number")
args = ap.parse_args()

port = args.port

tstamp = time.time()
# Create global variables and classes
# -----------------------------------
picam = None
f = open('camera-settings.json', 'r')
picam_conf = json.load(f)
if not picam_conf:
    picam_conf = {'resolution'   : (1024, 768),
                  'iso'          : 0,
                  'exposure_mode': 'auto',
                  'shutter_speed': 1000000,
                  'framerate'    : 5,
                  'awb_mode'     : 'auto',
                  'zoom'         : (0.0, 0.0, 1.0, 1.0),
                  'sensor_mode'  : 3,
                  'file_name'    : 'no-file'
                  }
# Create and configure camera module instance

# ----------------
# ---   MAIN   ---
# ----------------

# Start server in private mode in the main thread
tstamp = time.time()
myServer = DialogServer(port, True)
myServer.set_commands({
    'list-conn'  : hw_listconn,
    'start'      : hw_video_command,
    'stop'       : hw_video_command,
    'pause'      : hw_video_command,
    'restart'    : hw_video_command,
    'load'       : hw_video_command,
    'save'       : hw_video_command,
    'sets'       : hw_video_command,
    'debug'      : hw_video_debug,
    'set'        : hw_video_setprop
})
myServer.welcome = hw_welcome
myServer.run()

print("This is the end of code..")
