import picamera
from threading import *
from fractions import Fraction
import time
import io
import struct
import sys

# Global camera attributes
allprops = {}

# Not working, revision attribute error!
# Strange because manual says otherwise
def getPiCameraRevision(camera):
    rev = camera.revision
    if rev == 'ov5647':
        rev += " (V1)"
    if rev == 'ov5647':
        rev += " (V2)"
    return rev


def set_camera_property(picam, index, value, live):
    global allprops
    prop = [index]
    
    try:
        default = True
        prop = allprops[int(index)]
        while default:
            if (prop[0] == 'video_port'):
                # if live:
                #     picam.camera.sensor_mode = int(value)
                picam.conf[prop[0]] = bool(value)
                break
            if (prop[0] == 'sensor_mode'):
                # if live:
                #     picam.camera.sensor_mode = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'resolution'):
                # if not live:
                #     picam.camera.resolution = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'framerate'):
                # if live:
                #     picam.camera.framerate = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'iso'):
                if live:
                    picam.camera.iso = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'exposure_compensation'):
                if live:
                    picam.camera.exposure_compensation = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'shutter_speed'):
                if live:
                    picam.camera.shutter_speed = int(value)
                picam.conf[prop[0]] = int(value)
                # this setting will be applied during the next camera restart
                frames = Fraction(1000000, int(value))
                picam.conf['framerate'] = frames
                break
            if (prop[0] == 'exposure_mode'):
                if live:
                    picam.camera.exposure_mode = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'awb_mode'):
                if live:
                    picam.camera.awb_mode = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'brightness'):
                if live:
                    picam.camera.brightness = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'contrast'):
                if live:
                    picam.camera.contrast = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'saturation'):
                if live:
                    picam.camera.saturation = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'flash_mode'):
                if live:
                    picam.camera.flash_mode = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'sharpness'):
                if live:
                    picam.camera.sensor_mode = int(value)
                picam.conf[prop[0]] = int(value)
                break
            if (prop[0] == 'awb_gains'):
                if live:
                    picam.camera.awb_gains = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'drc_strength'):
                if live:
                    picam.camera.drc_strength = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'zoom'):
                roi = value.split(',')
                if live:
                    picam.camera.zoom = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
                picam.conf[prop[0]] = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
                break
            if (prop[0] == 'image_denoise'):
                if live:
                    picam.camera.image_denoise = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'image_effect'):
                if live:
                    picam.camera.image_effect = value
                picam.conf[prop[0]] = value
                break
            if (prop[0] == 'exposure_speed'):
                if live:
                    picam.camera.exposure_speed = bool(value)
                picam.conf[prop[0]] = bool(value)
                break
            default = False
        
        # if (default):
        #     picam.conf['file_name'] += " (edited)"

    except:
        default = False
        print("[error]: ", sys.exc_info()[0])

    if default:
        print("Camera property set: {}={}".format(prop[0], value))
    else:
        print("Invalid camera property or format: {}={}".format(prop[0], value))
    return default


def camera_settings(picam):
    camera = picam.camera
    # properties editable while recording is stopped
    camprops_init = {'sensor_mode': camera.sensor_mode,
                     'resolution' : camera.resolution,
                     'framerate'  : camera.framerate}
    # properties for camera light adjustment. Always editable
    camprops_img = {'iso'                  : camera.iso,
                    'exposure_compensation': camera.exposure_compensation,
                    'shutter_speed'        : camera.shutter_speed,
                    'exposure_mode'        : camera.exposure_mode,
                    'awb_mode'             : camera.awb_mode,
                    'brightness'           : camera.brightness,
                    'contrast'             : camera.contrast,
                    'saturation'           : camera.saturation,
                    'flash_mode'           : camera.flash_mode,
                    'sharpness'            : camera.sharpness}
    # mode and configuration properties
    camprops_dep = {'awb_gains'    : camera.awb_gains,
                    'drc_strength' : camera.drc_strength,
                    'zoom'         : camera.zoom,
                    'image_denoise': camera.image_denoise,
                    'image_effect' : camera.image_effect}
    # General properties and status
    camprops_view = {'exposure_speed': camera.exposure_speed,
                     'video_port'    : picam.videoport,
                     'file_name'     : picam.setfname}

    return camprops_init, camprops_img, camprops_dep, camprops_view


def get_camera_settings(picam):
    global allprops

    camprops_init, camprops_img, camprops_dep, camprops_view = camera_settings(picam)
    
    inc = 0
    view_screen = "Camera properties:"
    view_screen+= "\nBase Settings (no live camera)\n" + \
                  "------------------------------------\n"
    for k, v in camprops_init.items():
        allprops.update({inc: [k, v]})
        view_screen += "{:2}.{:>16} = {}\n".format(inc, k, v)
        inc += 1
    view_screen+= "\nImage Settings (live camera)\n" + \
                  "------------------------------------\n"
    for k, v in camprops_img.items():
        allprops.update({inc: [k, v]})
        view_screen += "{:2}.{:>16} = {}\n".format(inc, k, v)
        inc += 1
    view_screen+= "\nConfiguration Settings (live camera)\n" + \
                  "------------------------------------\n"
    for k, v in camprops_dep.items():
        allprops.update({inc: [k, v]})
        view_screen += "{:2}.{:>16} = {}\n".format(inc, k, v)
        inc += 1

    # General camera properties and status
    view_screen+= "\nCamera Status (view only)\n" + \
                  "------------------------------------\n"
    for k, v in camprops_view.items():
        allprops.update({inc: [k, v]})
        view_screen += "{:2}.{:>16} = {}\n".format(inc, k, v)
        inc += 1
        
    return view_screen


def get_picam(conf):
    
    # Force sensor mode 3 (the long exposure mode), set
    # the framerate to 1/6fps, the shutter speed to 6s,
    # and ISO to 800 (for maximum gain)
    smode = conf['sensor_mode']
    if not smode:
        smode = 3 if conf['shutter_speed'] >= 1000000 else 2
    camera = picamera.PiCamera(
        resolution=conf['resolution'],
        sensor_mode=smode)
    # frames = Fraction(conf['framerate'], 1) if smode == 2 else Fraction(1000000, conf['shutter_speed'])
    if smode == 2:
        camera.framerate = conf['framerate'] if 'framerate' in conf else 1
    elif conf['shutter_speed'] > 0:
        camera.framerate = Fraction(1000000, conf['shutter_speed'])

    camera.exposure_mode = conf['exposure_mode']
    camera.exposure_compensation = conf['exposure_compensation']
    camera.sharpness = conf['sharpness']
    camera.saturation = conf['saturation']
    camera.flash_mode = conf['flash_mode']
    camera.contrast = conf['contrast']
    camera.brightness = conf['brightness']
    camera.iso = conf['iso']
    camera.awb_mode = conf['awb_mode']
    # camera.awb_gains = (1.05, 1.57)
    camera.drc_strength = 'high'
    # camera.still_stats=True
    if 'zoom' in conf:
        camera.zoom = conf['zoom']
    
    # Give the camera a good long time to set gains and
    # measure AWB (you may wish to use fixed AWB instead)
    st = 1 if smode == 2 else int(20 * conf['shutter_speed'] / 6000000) + 10
    print("Warming up camera for {0}s..".format(st))
    time.sleep(st)
    
    if conf['shutter_speed'] == 0:
        camera.shutter_speed = camera.exposure_speed
    else:
        camera.shutter_speed = conf['shutter_speed']
        camera.exposure_mode = 'off'
    
    # Final color adjustments
    # camera.contrast = 50
    
    # Finally, capture an image with a 6s exposure. Due
    # to mode switching on the still port, this will take
    # longer than 6 seconds
    return camera


class PICameraServer(Thread):
    
    def __init__(self):
        Thread.__init__(self)
        self.camera = None
        self.conf = None
        self.stream = None
        self.log = None
        self.plugins = []
        self.socket = None
        self.setfname = None
        self.state = 'init'
        self.videoport = False
        self.frame_no = 0
        self.streamLength = 0
        self.stm = time.time()
        
        self.conf = {'resolution'   : (1024, 768),
                     'iso'          : 0,
                     'exposure_mode': 'auto',
                     'shutter_speed': 1000000,
                     'framerate'    : 5,
                     'awb_mode'     : 'auto',
                     'zoom'         : (0.0, 0.0, 1.0, 1.0),
                     'video_port'   : False
                     }

    def init_camera(self, conf=None, stream=None, fname="undefined.json"):
        if self.state == 'init':
            if conf:
                self.conf = conf

            self.setfname = self.conf['file_name'] if 'file_name' in conf else fname
            self.camera = get_picam(self.conf)
            if 'video_port' in conf:
                self.videoport = bool(self.conf['video_port'])
            self.stream = stream
            self.state = "idle"
        else:
            print("PiCamera already initialized. Current state is '{}'".format(self.state))
        
        self.log = get_camera_settings(self)
        # print(self.log)
        
        return self.state

    def close_camera(self):
        if self.state == 'init':
            return
        if self.state == 'streaming':
            self.state = 'closing'
            i=0
            while self.state != 'closed':
                i += 1
                print("Finishing current capture{}".format('.' * i), end='\r')
                time.sleep(1)
        self.stream.flush()
        self.camera.close()
        # self.stream = None
        
    def set_stream(self, stream):
        self.stream = stream
    
    def run(self):
        self.state = "streaming"
        stream = io.BytesIO()
        self.stm = time.time()
        self.frame_no = 0

        while True:
            if self.state == 'streaming':
                for foo in self.camera.capture_continuous(stream, 'jpeg', use_video_port=self.videoport):
                    self.frame_no += 1
                    self.streamLength = stream.tell()
        
                    
                    # Write the length of the capture to the stream
                    self.stream.write(struct.pack('<L', self.streamLength))
                    # Rewind the stream and send the image data over the wire
                    stream.seek(0)
                    self.stream.write(stream.read())
                    self.stream.flush()
        
                    # Reset the stream for the next capture
                    stream.seek(0)
                    stream.truncate()
        
                    # persistent processors
                    for plugin in self.plugins:
                        plugin(self)

                    self.stm = time.time()

                    if self.state in ['pause', 'closing']:
                        break
    
                if self.state == 'closing':
                    break

        self.state = "closed"
        print("PiCamera thread stopped!")
