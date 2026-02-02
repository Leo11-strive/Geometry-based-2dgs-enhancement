#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
import socket
import errno
import json
import struct
from scene.cameras import MiniCam

host = "127.0.0.1"
port = 6009

conn = None
addr = None

listener = None

def init(wish_host, wish_port, allow_fallback=True, port_retries=100, fallback_to_ephemeral=True):
    global host, port, listener
    host = wish_host
    port = wish_port

    if listener is not None:
        try:
            listener.close()
        except Exception:
            pass

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    selected_port = None

    wish_port = int(wish_port)
    if wish_port == 0:
        # Let OS choose an available ephemeral port.
        listener.bind((host, 0))
        selected_port = listener.getsockname()[1]
    else:
        last_err = None
        tries = int(port_retries) if int(port_retries) >= 0 else 0
        for i in range(tries + 1):
            try_port = wish_port + i
            try:
                listener.bind((host, try_port))
                selected_port = try_port
                break
            except OSError as e:
                last_err = e
                if (not allow_fallback) or (e.errno not in (errno.EADDRINUSE, errno.EACCES)):
                    raise

        if selected_port is None and allow_fallback and fallback_to_ephemeral:
            # As a last resort, bind to an ephemeral port.
            try:
                listener.bind((host, 0))
                selected_port = listener.getsockname()[1]
            except OSError:
                if last_err is not None:
                    raise last_err
                raise

    port = int(selected_port) if selected_port is not None else int(wish_port)
    listener.listen()
    listener.settimeout(0)
    return port

def send_json_data(conn, data):
    # Serialize the list of strings to JSON
    serialized_data = json.dumps(data)
    # Convert the serialized data to bytes
    bytes_data = serialized_data.encode('utf-8')
    # Send the length of the serialized data first
    conn.sendall(struct.pack('I', len(bytes_data)))
    # Send the actual serialized data
    conn.sendall(bytes_data)

def try_connect(render_items):
    global conn, addr, listener
    try:
        conn, addr = listener.accept()
        # print(f"\nConnected by {addr}")
        conn.settimeout(None)
        send_json_data(conn, render_items)
    except Exception as inst:
        pass
        # raise inst
            
def read():
    global conn
    messageLength = conn.recv(4)
    messageLength = int.from_bytes(messageLength, 'little')
    message = conn.recv(messageLength)
    return json.loads(message.decode("utf-8"))

def send(message_bytes, verify, metrics):
    global conn
    if message_bytes != None:
        conn.sendall(message_bytes)
    conn.sendall(len(verify).to_bytes(4, 'little'))
    conn.sendall(bytes(verify, 'ascii'))
    send_json_data(conn, metrics)

def receive():
    message = read()
    width = message["resolution_x"]
    height = message["resolution_y"]

    if width != 0 and height != 0:
        try:
            do_training = bool(message["train"])
            fovy = message["fov_y"]
            fovx = message["fov_x"]
            znear = message["z_near"]
            zfar = message["z_far"]
            keep_alive = bool(message["keep_alive"])
            scaling_modifier = message["scaling_modifier"]
            world_view_transform = torch.reshape(torch.tensor(message["view_matrix"]), (4, 4)).cuda()
            world_view_transform[:,1] = -world_view_transform[:,1]
            world_view_transform[:,2] = -world_view_transform[:,2]
            full_proj_transform = torch.reshape(torch.tensor(message["view_projection_matrix"]), (4, 4)).cuda()
            full_proj_transform[:,1] = -full_proj_transform[:,1]
            custom_cam = MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)
            render_mode = message["render_mode"]
        except Exception as e:
            print("")
            traceback.print_exc()
            # raise e
        return custom_cam, do_training, keep_alive, scaling_modifier, render_mode
    else:
        return None, None, None, None, None
