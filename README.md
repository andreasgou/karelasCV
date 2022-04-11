# karelasCV

A client-server remote CV python app.

**WARNING:** 
Development is not ready, thus commits are buggy !

## Features
**Servers**
- Pi chat server. Supports multiple clients. It's basic usage is to control Pi camera video streaming 
via commands send from the client.
- Pi video streaming server. Video source from Pi's camera module.
- File list server/client. Video source comes from images in rotation from a local path.  

**Clients**
- Chat server client.
- Pi video streaming client.
- HTTP streaming client. Connects to IP webcam android app.
https://play.google.com/store/apps/details?id=com.pas.webcam   

##Installation
###MacOS
- `git clone https://github.com/andreasgou/karelasCV.git`
- `cd karelasCV`
- Mac Intel (x86):
  - `brew install hdf5`
  - `export HDF5_DIR=/usr/local/..?../hdf5`
  - `pip install -r requirements-mac.txt`
- Mac M1 (arm64):
  - `arch -arm64 brew install hdf5`
  - `export HDF5_DIR=/opt/homebrew/opt/hdf5`
  - `pip install -r requirements-m1.txt`

