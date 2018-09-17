import sys
import time
import datetime
import cv2
from threading import *
from imutils import paths
from utils.datasets import SimpleDatasetLoader
from utils.preprocessing import SimplePreprocessor

from .helper_sockets import *


# Server socket for streaming images from a local folder
# ------------------------------------------------------------------
class StreamVideoClient(Thread):
	
	def __init__(self, dim, hdl_terminate, video_path=None):
		Thread.__init__(self)
		self.status = "init"
		self.pipe = None
		self.consumer = None
		self.data = None
		self.idletime = 0.0
		self.terminate = hdl_terminate
		self.dim = dim
		self.capture = None
		self.video_path = video_path
		self.pause = False
		self.pause_time = 0.0
	
	def init_socket(self):
		try:
			if self.video_path is None:
				self.capture = cv2.VideoCapture(0)
			else:
				self.capture = cv2.VideoCapture(self.video_path)

			if self.capture.isOpened():
				# Default delay equivalent to fps
				self.idletime = 1 / self.capture.get(cv2.CAP_PROP_FPS)
				self.set_command("res", [self.dim[0], self.dim[1]])
				print("""Video capture properties:
     Total frames: {}
     Capture mode: {}
   Capture format: {}
Frames per second: {}
            Codec: {}
   Soft Idle time: {} ms
""".format(self.capture.get(cv2.CAP_PROP_FRAME_COUNT),
           self.capture.get(cv2.CAP_PROP_MODE),
           self.capture.get(cv2.CAP_PROP_FORMAT),
           self.capture.get(cv2.CAP_PROP_FPS),
		   self.capture.get(cv2.CAP_PROP_FOURCC),
		   self.idletime * 1000
           ))
		except:
			self.status = 'failed'
			print("[error]: protocol failed: ", sys.exc_info())
		
		return self.status
	
	def run(self):
		self.status = 'running'
		print("[debug] Thread is running..")
		frameno = frame_drop = 0
		tic_on = time.time()
		self.pause_time = 0.0
		while True:
			if not self.capture.isOpened():
				break
			if self.pause:
				continue

			try:
				if self.status == 'purge':
					# consume all data from the pipe
					# self.pipe.flush()
					self.consumer = None
					print("[info] (StreamClient.run) : Video stream purged")
					break
				
				else:
					tic = time.time()
					status, data = self.capture.read()
					if status:
						ln = len(data)
						if ln > 0 and self.consumer is not None:
							self.consumer(self, ln, data)
						else:
							self.status = 'purge'
							print("[warning] (StreamVideoClient.run) : no data")
							# pass control to the custom terminate handler
							# self.terminate()
							break
					elif self.video_path is not None:
						# print("\r[info] Video finished in frame {}. Restarting..\n> "
						#       .format(self.capture.get(cv2.CAP_PROP_POS_FRAMES)), end='')
						print("\rframe: {}, position: {}, average delay: {:.2f} ms ".format(frameno, timepos, avgtime*1000))
						self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
						tic_on = time.time()
						self.pause_time = 0.0
						# frameno = frameno_run = 0
						# self.status = 'purge'
					else:
						print("[warning] Camera module not working. Restart camera module.")

					toc = time.time()
					
					# calculate dropped frames
					caltime = toc - tic
					avgtime = (toc - tic_on - self.pause_time) / (frameno+1)
					frameno_run = round((toc - tic_on - self.pause_time) / self.idletime)
					if self.video_path is not None:
						timepos = datetime.datetime.utcfromtimestamp(
							self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0).strftime('%H:%M:%S')
						runpos = datetime.datetime.utcfromtimestamp(toc-tic_on-self.pause_time).strftime('%H:%M:%S')
						frameno = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
						frameno_lag = int(frameno_run - frameno)
					else:
						timepos = datetime.datetime.utcfromtimestamp(toc-tic_on-self.pause_time).strftime('%H:%M:%S')
						runpos = timepos
						frameno += 1
						frameno_lag = int(frameno_run - frameno)

					# print("\rFrame:{}, lag:{}, drop:{}, vpos:{}, rpos:{}, interval:{:.2f} ms "
					#       .format(frameno, frameno_lag, frame_drop, timepos, runpos, avgtime*1000), end='')

				if self.video_path is not None:
					# sync time with dropped frames
					if frameno_lag <= 0:
						sleeptime = self.idletime - caltime
						if sleeptime >= 0:
							time.sleep(sleeptime)
					elif frameno_lag > 5:
						for i in range(frameno_lag):
							frame_drop += 1
							self.capture.grab()
					elif frameno_lag > 2:
						frame_drop += 1
						self.capture.grab()
				
				if self.status == 'purge':
					self.consumer = None
					print("[info] (StreamClient.run) : Video stream purged")
					break

			except UserWarning:
				print("[error] stream failed: ", sys.exc_info()[0])
				self.status = 'failed'
				self.close()
				break
		
		self.status = "init"
		print("Stream socket stopped")
	
	def set_consumer(self, consumer):
		self.consumer = consumer
	
	def close(self):
		print("Closing data stream..")
		while self.status != "init":
			time.sleep(1)
		self.capture.release()
		print("Stream socket closed")
	
	def set_command(self, cmd, args):
		if cmd == "res":
			self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(args[0]))
			self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args[1]))
			print("Width:{} Height:{}".format(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH), self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	
	def purge_negotiate(self):
		self.status = 'purge'

