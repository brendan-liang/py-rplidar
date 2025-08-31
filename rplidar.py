"""
File:           rplidar.py
Description:    RPLIDAR S1/S2/S3/C1 library for MicroPython.
                Tested and designed for the RPLidar C1M1

Author:         brendan-liang
"""

# Imports
import machine
import time
import _thread

# Constants (serial protocol)
_SERIAL_START = 0xA5

_CMD_STOP = 0x25
_CMD_RESET = 0x40
_CMD_SCAN = 0x20
_CMD_EXP_SCAN = 0x82
_CMD_FORCE_SCAN = 0x21

_SCAN_RESPONSE_DESCRIPTOR = bytes([0xA5, 0x5A, 0x05, 0x00, 0x00, 0x40, 0x81])

_CMD_GET_INFO = 0x50
_CMD_GET_HEALTH = 0x52
_CMD_GET_SAMPLERATE = 0x59
_CMD_GET_LIDAR_CONF = 0x84

# Helper functions
def _calculate_checksum(data_bytes:bytes):
    checksum = 0
    for b in data_bytes:
        checksum ^= b
    return checksum 

# Helper classes
class RPLidarResponseError(Exception):
    pass

class RPLidarInfoResponse:
    model: str
    firmware_version: str
    hardware_version: int
    serial_number: int

    def __init__(self, model:str, firmware_version:str, hardware_version:int, serial_number:int):
        self.model = model
        self.firmware_version = firmware_version
        self.hardware_version = hardware_version
        self.serial_number = serial_number

    def __repr__(self):
        return f"RPLidarInfoResponse(model={self.model}, firmware_version={self.firmware_version}, hardware_version={self.hardware_version}, serial_number={self.serial_number})"

class RPLidarHealthResponse:
    status: int
    error_code: int

    def __init__(self, status:int, error_code:int):
        self.status = status
        self.error_code = error_code

    def __repr__(self):
        return f"RPLidarHealthResponse(status={self.status}, error_code={self.error_code})"
    
class RPLidarSampleRateResponse:
    time_standard: int
    time_express: int

    def __init__(self, time_standard:int, time_express:int):
        self.time_standard = time_standard
        self.time_express = time_express
    
    def __repr__(self):
        return f"RPLidarSampleRateResponse(time_standard={self.time_standard}, time_express={self.time_express})"
    
class RPLidarConfResponse:
    conf_type: int
    config: int | str

    def __init__(self, conf_type:int, config:bytes):
        self.conf_type = conf_type
        self.config = config

    def __repr__(self):
        return f"RPLidarConfResponse(conf_type={hex(self.conf_type)}, config={self.config})"

# Main Class
class RPLidar(machine.UART):
    """
    RPLidar class for MicroPython.
    """
    def __init__(self, id=0, baudrate=460800, tx=machine.Pin(0), rx=machine.Pin(1), timeout=100):
        super().__init__(id, baudrate, tx=tx, rx=rx)
        self.init(baudrate=baudrate, bits=8, parity=None, stop=1, timeout=timeout)

        self._timeout = timeout
        self._last_request = int(time.time_ns() * 1e-6)  # Convert to milliseconds
        self._wait_time = 0

        self.scanning = False
        self.scan_thread = None

        # Test connection
        info = self.get_info()
        print(f"RPLidar {info.model} connected. Running firmware v{info.firmware_version}, hardware v{info.hardware_version}. S/N: {info.serial_number}")

    def _request(self, cmd: int, payload: bytes = None, wait_time: int = 0) -> bool:
        if self.scanning and cmd != _CMD_STOP:
            return False  # Only allow STOP command during scanning
        if cmd not in [_CMD_STOP, _CMD_RESET, _CMD_SCAN, _CMD_EXP_SCAN, _CMD_FORCE_SCAN,
                       _CMD_GET_INFO, _CMD_GET_HEALTH, _CMD_GET_SAMPLERATE, _CMD_GET_LIDAR_CONF]:
            raise ValueError("Invalid command")
        # Ensure minimum wait time between requests
        current_time = int(time.time_ns() * 1e-6) 
        elapsed_time = current_time - self._last_request
        if elapsed_time < self._wait_time:
            time.sleep((self._wait_time - elapsed_time) / 1000)
        # Construct command packet
        packet = bytes([_SERIAL_START, cmd])
        if payload:
            packet += bytes([len(payload)]) + payload
        # Add checksum
        checksum = _calculate_checksum(packet)
        packet += bytes([checksum])
        # Send command
        self.write(packet)
        print("Sent:", " ".join([hex(b) for b in packet]))
        # Update last request time and wait time
        self._last_request = int(time.time_ns() * 1e-6)
        self._wait_time = wait_time
        
        return True

    def _single_response(self, length:int=-1) -> bytes:
        if self.scanning:
            return b''  # No response during scanning mode
        last_byte_time = time.time_ns()
        received_data = bytearray()
        # Read descriptor (7 bytes)
        while len(received_data) < 7:
            if self.any():
                received_data += self.read(1)
                last_byte_time = time.time_ns()
                continue
            if int(time.time_ns() - last_byte_time) * 1e-6 > self._timeout:
                break
        if len(received_data) < 7:
            raise RPLidarResponseError(f"Timeout waiting for response descriptor. (Received {len(received_data)}/7 bytes)")
        
        # If length not specified, get from descriptor
        if length == -1:
            length = received_data[2]

        length += 7  # Add descriptor length

        # Read response, waiting for the full length or timeout
        while len(received_data) < length:
            if self.any():
                received_data += self.read(1)
                last_byte_time = time.time_ns()
                continue
            if int(time.time_ns() - last_byte_time) * 1e-6 > self._timeout:
                break
        if not received_data:
            raise RPLidarResponseError(f"Timeout waiting for response. (No data received)")
        if len(received_data) < length:
            raise RPLidarResponseError(f"Timeout waiting for full response. (Received {len(received_data)}/{length} bytes)")
        if received_data[0] != 0xA5 or received_data[1] != 0x5A:
            raise RPLidarResponseError("Invalid response header")
        print(" ".join([hex(b) for b in received_data]))
        return bytes(received_data[7:])  # Strip off the descriptor


    def get_info(self) -> RPLidarInfoResponse:
        self._request(_CMD_GET_INFO)
        res = self._single_response(20)

        model = {
            6: "S1",
            7: "S2",
            8: "S3",
            4: "C1",
        }.get(res[0] >> 4, "??") + f"M{res[0] & 0x0F}"
        
        return RPLidarInfoResponse(
            model = model,
            firmware_version = f"{res[2]}.{res[1]}",
            hardware_version = res[3],
            serial_number = int.from_bytes(res[4:19], 'little')
        )

    def get_health(self) -> RPLidarHealthResponse:
        self._request(_CMD_GET_HEALTH)
        res = self._single_response(3)
        return RPLidarHealthResponse(res[0], res[1] << 8 | res[2])
    
    def get_samplerate(self) -> RPLidarSampleRateResponse:
        self._request(_CMD_GET_SAMPLERATE)
        res = self._single_response(4)
        return RPLidarSampleRateResponse(
            time_standard = int.from_bytes(res[0:2], 'little'),
            time_express = int.from_bytes(res[2:4], 'little')
        )
    
    def get_lidar_conf(self, conf_type:int, payload:int=None) -> RPLidarConfResponse:
        # Check valid type
        if conf_type not in [0x70, 0x71, 0x74, 0x75, 0x7C, 0x7F]:
            raise ValueError("Invalid config type. Must be one of [0x70, 0x71, 0x74, 0x75, 0x7C, 0x7F]. See RPLIDAR communication protocol for details.")
        if conf_type in [0x71, 0x74, 0x75, 0x7F] and payload == None:
            raise ValueError(f"Config type {hex(conf_type)} requires a payload. See RPLIDAR communication protocol for details.")
        # Create payload, padding type to 4 bytes and payload to 2 bytes (u16)
        request_payload = bytes([conf_type]) + bytes(3)
        if payload != None:
            request_payload += bytes([payload]) + bytes(1)
        self._request(_CMD_GET_LIDAR_CONF, request_payload)
        res = self._single_response()
        # Check response has payload
        if len(res) <= 4:
            raise RPLidarResponseError("Incomplete response received. Ensure the config type and payload are valid.")
        # Convert response type depending on config type
        if conf_type == 0x7F:
            config = res[4:].decode('utf-8').rstrip('\x00')
        else:
            config = int.from_bytes(res[4:], 'little')

        return RPLidarConfResponse(conf_type, config)
    
    def _scan_response(self, test_check):
        if not self.scanning:
            return b''  # No response if not scanning
        
        # Read block of 5 bytes
        received_data = bytearray()
        last_byte_time = time.time_ns()
        while len(received_data) < 5:
            if self.any():
                received_data += self.read(1)
                last_byte_time = time.time_ns()
                continue
            if int(time.time_ns() - last_byte_time) * 1e-6 > self._timeout:
                break
        if len(received_data) < 5:
            raise RPLidarResponseError(f"Timeout waiting for scan response. (Received {len(received_data)}/5 bytes)")
        # Check start flag
        start_flag = received_data[0] & 0x01
        inverse_flag = received_data[0] & 0x02

        # Find check bit (test)
        test_check = bytes([b & test_check[i] for i, b in enumerate(received_data)])
        # if start_flag == inverse_flag:
        #     raise RPLidarResponseError(f"Invalid scan response start flag.")
        
        return test_check  # Strip off the descriptor
    
    def _scan_loop(self, scan_flag):
        # Find response descriptor
        received_descriptor = bytearray()
        last_byte_time = time.time_ns()
        while len(received_descriptor) < 7:
            if self.any():
                received_descriptor += self.read(1)
                last_byte_time = time.time_ns()
                continue
            if int(time.time_ns() - last_byte_time) * 1e-6 > self._timeout:
                break
        if len(received_descriptor) < 7:
            raise RPLidarResponseError(f"Timeout waiting for scan response descriptor. (Received {len(received_descriptor)}/7 bytes)")
        if received_descriptor != _SCAN_RESPONSE_DESCRIPTOR:
            raise RPLidarResponseError("Invalid scan response descriptor: " + " ".join([hex(b) for b in received_descriptor]))
        print("Scan response descriptor received: " + " ".join([hex(b) for b in received_descriptor]))
        # Main scan loop
        sample = b'0xff\0xff\0xff\0xff\0xff'  # Dummy initial value
        while scan_flag():
            sample = self._scan_response(sample)
            print(sample)
        print("Exiting scan loop.")

    def stop_scan(self):
        self.scanning = False
        # if self.scan_thread:
        #     self.scan_thread.join()
        self.scan_thread = None
        self._request(_CMD_STOP, wait_time=10)
    
    def start_scan(self):
        if self.scanning:
            return  # Already scanning
        self._request(_CMD_SCAN)
        self.scanning = True

        self.scan_thread = _thread.start_new_thread(self._scan_loop, (lambda: self.scanning,))


if __name__ == "__main__":
    # Simple example
    lidar = RPLidar(0, 460800, timeout=1500)

    print(f"\nHealth: {["OK", "Warning", "Error"][lidar.get_health().status]}\n")
    
    time.sleep(1)

    print("Starting scan...")
    time.sleep(1)
    lidar.start_scan()
    time.sleep(6)
    print("Stopping scan...")
    lidar.stop_scan()
    print("Scan stopped.")