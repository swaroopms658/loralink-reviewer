import socket
import threading
import json
import dataclasses
import enum
import sys
import time

class MessageType(enum.Enum):
    HEARTBEAT = enum.auto()
    RUN_BENCHMARK = enum.auto()
    BENCHMARK_RESULT = enum.auto()
    SET_CONFIG = enum.auto()
    WORKER_READY = enum.auto()
    TENSOR = enum.auto()
    GRADIENT = enum.auto()
    ECHO_REQUEST = enum.auto()
    ECHO_RESPONSE = enum.auto()
    GET_LORA_WEIGHTS = enum.auto()
    LORA_WEIGHTS_RESPONSE = enum.auto()
    SHUTDOWN = enum.auto()  # Signal idle worker to shutdown gracefully

@dataclasses.dataclass
class Message:
    message_type: MessageType
    payload: bytes
    metadata: dict = dataclasses.field(default_factory=dict)

    def serialize(self) -> bytes:
        assert isinstance(self.message_type, MessageType)
        assert isinstance(self.payload, bytes)
        assert isinstance(self.metadata, dict)

        header = {"type": self.message_type.value, "meta_len": 0}
        metadata_bytes = b""

        if self.metadata:
            metadata_bytes = json.dumps(self.metadata).encode("utf-8")
            header["meta_len"] = len(metadata_bytes)

        header_bytes = json.dumps(header).encode("utf-8")
        header_len_bytes = len(header_bytes).to_bytes(2, "little")

        result = header_len_bytes + header_bytes + metadata_bytes + self.payload
        assert len(result) >= 2

        return result

    @classmethod
    def deserialize(cls, data: bytes) -> "Message":
        assert isinstance(data, bytes)
        assert len(data) >= 2

        header_len = int.from_bytes(data[:2], "little")
        assert header_len > 0
        assert len(data) >= 2 + header_len

        header_bytes = data[2:2 + header_len]
        header = json.loads(header_bytes.decode("utf-8"))

        assert "type" in header
        assert "meta_len" in header

        meta_len = header["meta_len"]
        assert meta_len >= 0
        assert len(data) >= 2 + header_len + meta_len

        metadata = {}
        if meta_len > 0:
            metadata_bytes = data[2 + header_len:2 + header_len + meta_len]
            metadata = json.loads(metadata_bytes.decode("utf-8"))

        payload = data[2 + header_len + meta_len:]
        message_type = MessageType(header["type"])

        return cls(message_type=message_type, payload=payload, metadata=metadata)

class NetworkManager:
    def __init__(self, host: str, port: int, message_callback: callable):
        assert isinstance(host, str)
        assert isinstance(port, int)
        assert 1 <= port <= 65535
        assert callable(message_callback)

        self.host = host
        self.port = port
        self.callback = message_callback
        self.server_thread = None
        self.running = False

    def start_server(self):
        assert not self.running
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self.server_thread.start()
        print(f"✅ Server started on {self.host}:{self.port}")

    def stop_server(self):
        if not self.running:
            return

        self.running = False

        try:
            dummy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dummy_sock.connect((self.host if self.host != "0.0.0.0" else "127.0.0.1", self.port))
            dummy_sock.close()
        except ConnectionRefusedError:
            pass
        except Exception:
            pass

        if self.server_thread:
            self.server_thread.join()

    def _server_loop(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)

        while self.running:
            try:
                conn, addr = server_socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_connection,
                    args=(conn, addr[0]),
                    daemon=True
                )
                client_thread.start()
            except OSError:
                if self.running:
                    break

        server_socket.close()

    def _handle_connection(self, conn: socket.socket, ip_addr: str):
        with conn:
            try:
                
                length_bytes = self._read_exact(conn, 4)
                if not length_bytes or len(length_bytes) != 4:
                    print(f"Failed to read complete length header from {ip_addr}")
                    return

                total_len = int.from_bytes(length_bytes, "little")
                if total_len <= 0 or total_len > 100 * 1024 * 1024:  
                    print(f"Invalid message length {total_len} from {ip_addr}")
                    return

                
                message_data = self._read_exact(conn, total_len)
                if len(message_data) != total_len:
                    print(f"Incomplete message data from {ip_addr}: expected {total_len}, got {len(message_data)}")
                    return
                    
                try:
                    message = Message.deserialize(message_data)
                except Exception as e:
                    print(f"Failed to deserialize message from {ip_addr}: {e}")
                    return

                
                try:
                    self.callback(ip_addr, message)
                except Exception as e:
                    print(f"Error in message callback for {ip_addr}: {e}")
                    
                    
            except Exception as e:
                print(f"Error handling connection from {ip_addr}: {e}")

    def _read_exact(self, conn: socket.socket, total_len: int) -> bytes:
        assert total_len > 0
        chunks = []
        bytes_received = 0
        
        
        conn.settimeout(30.0)  
        
        while bytes_received < total_len:
            try:
                remaining = total_len - bytes_received
                chunk = conn.recv(min(remaining, 8192))  
                if not chunk:
                    raise RuntimeError(f"Socket connection broken: received {bytes_received}/{total_len} bytes")
                chunks.append(chunk)
                bytes_received += len(chunk)
            except socket.timeout:
                print(f"Timeout reading from socket: received {bytes_received}/{total_len} bytes")
                raise RuntimeError("Socket read timeout")

        result = b''.join(chunks)
        assert len(result) == total_len, f"Length mismatch: expected {total_len}, got {len(result)}"
        return result

    @staticmethod
    def send_message(peer_ip: str, peer_port: int, message: Message) -> int:
        assert isinstance(peer_ip, str)
        assert isinstance(peer_port, int)
        assert isinstance(message, Message)
        assert 1 <= peer_port <= 65535

        try:
            serialized_data = message.serialize()
            length_bytes = len(serialized_data).to_bytes(4, "little")
            framed_data = length_bytes + serialized_data
            total_bytes = len(framed_data)

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30.0)  
            
            try:
                sock.connect((peer_ip, peer_port))
                
                
                bytes_sent = 0
                while bytes_sent < len(framed_data):
                    chunk_sent = sock.send(framed_data[bytes_sent:])
                    if chunk_sent == 0:
                        raise RuntimeError("Socket connection broken during send")
                    bytes_sent += chunk_sent
                    
            finally:
                sock.close()
                
            return total_bytes
            
        except socket.timeout:
            print(f"Timeout connecting to {peer_ip}:{peer_port}")
            raise
        except Exception as e:
            print(f"Failed to send message to {peer_ip}:{peer_port}: {e}")
            raise
