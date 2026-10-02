# uds_bridge.py — host-side TCP -> (podman exec socat) -> UnrealCV UDS bridge.
# Run on the SERVER (host): python3 uds_bridge.py
# Then from the Mac:        ssh -N -L 9105:127.0.0.1:9105 user@server
import socket, subprocess, threading, os
LISTEN = ('127.0.0.1', 9105)
CMD = ["podman", "exec", "-i", "uz-scene", "socat", "-", "UNIX-CONNECT:/tmp/unrealcv_9104.socket"]


def handle(conn):
    p = subprocess.Popen(CMD, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def tcp_to_proc():
        try:
            while True:
                d = conn.recv(65536)
                if not d: break
                p.stdin.write(d); p.stdin.flush()
        except Exception:
            pass
        finally:
            try: p.stdin.close()
            except Exception: pass

    def proc_to_tcp():
        try:
            while True:
                d = os.read(p.stdout.fileno(), 65536)
                if not d: break
                conn.sendall(d)
        except Exception:
            pass
        finally:
            try: conn.close()
            except Exception: pass

    t1 = threading.Thread(target=tcp_to_proc, daemon=True)
    t2 = threading.Thread(target=proc_to_tcp, daemon=True)
    t1.start(); t2.start(); t1.join(); t2.join()
    try: p.terminate()
    except Exception: pass


s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(LISTEN); s.listen(5)
print("bridge listening on", LISTEN, "-> UDS via podman exec")
while True:
    conn, _ = s.accept()
    threading.Thread(target=handle, args=(conn,), daemon=True).start()