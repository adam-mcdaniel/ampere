import arkouda as ak
import os
import atexit

class AmpereSession:
    """
    Context manager for managing an Arkouda server connection.
    automatically connects on enter and disconnects on exit.
    """
    def __init__(self, server: str = "localhost", port: int = 5555, timeout: int = 0):
        """
        Initialize the session configuration.
        
        Args:
            server (str): Arkouda server hostname.
            port (int): Arkouda server port.
            timeout (int): Connection timeout in seconds.
        """
        self.server = server
        self.port = port
        self.timeout = timeout
        self.connected = False

    def __enter__(self):
        try:
            print(f"Connecting to Arkouda server at {self.server}:{self.port}...")
            ak.connect(server=self.server, port=self.port, timeout=self.timeout)
            self.connected = True
            print("Connected to Arkouda.")
        except Exception as e:
            print(f"Failed to connect to Arkouda: {e}")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.connected:
            try:
                print("Disconnecting from Arkouda...")
                ak.disconnect()
            except Exception as e:
                print(f"Error disconnecting: {e}")
            finally:
                self.connected = False

def connect(server="localhost", port=5555):
    """
    Helper function to connect to an Arkouda server without a context manager.
    Useful for interactive environments like Jupyter notebooks.
    
    Args:
        server (str): Arkouda server hostname.
        port (int): Arkouda server port.
        
    Side Effects:
        - Establishes a global Arkouda connection.
        - Registers `ak.disconnect` to run on interpreter exit.
    """
    print(f"Connecting to Arkouda server at {server}:{port}...")
    ak.connect(server=server, port=port)
    # Register disconnect on exit just in case
    atexit.register(ak.disconnect)
