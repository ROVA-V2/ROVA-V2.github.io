"""
APE detection service for Robust-TO's detect_objects tool.

A TCP socket server that loads APE once and answers detection requests.  Uses
length-prefixed framing (8-byte big-endian length header) so large payloads of
frame paths / results are never truncated -- the previous fixed recv(8192)
truncated multi-frame requests and the bare send() could short-write large
results.  Mirrors send_msg / recv_msg in
rovid_pipeline/tools/perception_tools.py.

Protocol
--------
Request : pickle((frame_paths: list[str], text_prompt: str))
Response: pickle(list-per-image of list of (label, [x,y,w,h], score))
"""
import socket
import struct
import pickle

from ape_api import setup_cfg, ape_inference
from predictor_lazy import VisualizationDemo


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    header = _recv_exact(sock, 8)
    if header is None:
        return None
    (length,) = struct.unpack(">Q", header)
    return _recv_exact(sock, length)


def send_msg(sock, payload):
    sock.sendall(struct.pack(">Q", len(payload)) + payload)


def main(host="0.0.0.0", port=9999):
    cfg = setup_cfg()
    demo = VisualizationDemo(cfg, args=None)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen()
    print(f"APE service listening on {host}:{port} ...")

    while True:
        client, addr = server.accept()
        try:
            data = recv_msg(client)
            if data is None:
                client.close()
                continue
            frame_paths, text_prompt = pickle.loads(data)
            result = ape_inference(frame_paths, text_prompt, demo)
            send_msg(client, pickle.dumps(result))
        except Exception as e:
            try:
                send_msg(client, pickle.dumps([]))
            except Exception:
                pass
            print(f"[APE service] error handling {addr}: {e}")
        finally:
            client.close()


if __name__ == "__main__":
    main()
