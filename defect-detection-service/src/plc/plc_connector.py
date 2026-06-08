from typing import Dict, Any, Optional, List, Callable
from queue import Queue
import threading
import time
from collections import deque

from src.utils.schemas import (
    PLCProtocol, PLCCommandType, PLCCommand, PLCCommandResult,
    DefectType, AlertAction
)
from src.utils.logger import Logger

logger = Logger().logger

try:
    from pymodbus.client import ModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    logger.warning("pymodbus not available, Modbus TCP disabled")

try:
    from opcua import Client, ua
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False
    logger.warning("opcua not available, OPC UA disabled")


class PLCConnector:
    DEFECT_CODE_MAP = {
        DefectType.SCRATCH: 101,
        DefectType.DIRT: 102,
        DefectType.DENT: 103,
        DefectType.CRACK: 104,
        DefectType.MISSING: 105,
        DefectType.STAIN: 106,
        DefectType.DEFORMATION: 107,
        DefectType.BUBBLE: 108,
        DefectType.UNKNOWN: 199
    }

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._enabled = config.get("enable", False)
        self._protocol = PLCProtocol(config.get("type", "modbus_tcp"))
        self._host = config.get("host", "localhost")
        self._port = config.get("port", 502)
        self._slave_id = config.get("slave_id", 1)

        self._reject_coil = config.get("reject_coil_address", 100)
        self._stop_line_coil = config.get("stop_line_coil_address", 101)
        self._alarm_coil = config.get("alarm_coil_address", 102)
        self._reset_coil = config.get("reset_coil_address", 103)
        self._heartbeat_coil = config.get("heartbeat_coil_address", 104)

        self._defect_code_register = config.get("defect_code_register", 200)
        self._pulse_duration_ms = config.get("pulse_duration_ms", 500)
        self._command_timeout_ms = config.get("command_timeout_ms", 3000)
        self._max_retries = config.get("max_retries", 3)
        self._retry_delay_ms = config.get("retry_delay_ms", 1000)

        self._modbus_client: Optional[ModbusTcpClient] = None
        self._opcua_client: Optional[Client] = None
        self._is_connected = False
        self._lock = threading.RLock()

        self._command_queue: Queue = Queue(maxsize=1000)
        self._result_callbacks: List[Callable[[PLCCommand, PLCCommandResult], None]] = []
        self._command_history: deque = deque(maxlen=1000)

        self._worker_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._stats = {
            "total_commands": 0,
            "successful_commands": 0,
            "failed_commands": 0,
            "reject_commands": 0,
            "stop_line_commands": 0,
            "alarm_commands": 0,
            "connection_errors": 0
        }

        if self._enabled:
            self._start_worker()

    def connect(self) -> bool:
        if not self._enabled:
            logger.info("PLC communication disabled in config")
            return False

        with self._lock:
            try:
                if self._protocol == PLCProtocol.MODBUS_TCP:
                    return self._connect_modbus()
                elif self._protocol == PLCProtocol.OPC_UA:
                    return self._connect_opcua()
            except Exception as e:
                logger.error(f"PLC connection failed: {e}", exc_info=True)
                self._stats["connection_errors"] += 1
                self._is_connected = False
                return False

    def _connect_modbus(self) -> bool:
        if not MODBUS_AVAILABLE:
            logger.error("Modbus TCP not available, install pymodbus")
            return False

        try:
            if self._modbus_client:
                try:
                    self._modbus_client.close()
                except:
                    pass

            self._modbus_client = ModbusTcpClient(
                host=self._host,
                port=self._port,
                timeout=self._command_timeout_ms / 1000.0
            )

            connection = self._modbus_client.connect()
            if connection:
                self._is_connected = True
                logger.info(f"✅ Modbus TCP connected to {self._host}:{self._port}")
                return True
            else:
                logger.error(f"Failed to connect to Modbus TCP {self._host}:{self._port}")
                self._is_connected = False
                return False

        except Exception as e:
            logger.error(f"Modbus connection error: {e}", exc_info=True)
            self._is_connected = False
            return False

    def _connect_opcua(self) -> bool:
        if not OPCUA_AVAILABLE:
            logger.error("OPC UA not available, install opcua")
            return False

        try:
            if self._opcua_client:
                try:
                    self._opcua_client.disconnect()
                except:
                    pass

            url = f"opc.tcp://{self._host}:{self._port}"
            self._opcua_client = Client(url=url, timeout=self._command_timeout_ms / 1000.0)
            self._opcua_client.connect()

            self._is_connected = True
            logger.info(f"✅ OPC UA connected to {url}")
            return True

        except Exception as e:
            logger.error(f"OPC UA connection error: {e}", exc_info=True)
            self._is_connected = False
            return False

    def disconnect(self):
        with self._lock:
            self._stop_event.set()

            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=2)

            if self._heartbeat_thread and self._heartbeat_thread.is_alive():
                self._heartbeat_thread.join(timeout=2)

            if self._modbus_client:
                try:
                    self._modbus_client.close()
                except Exception as e:
                    logger.warning(f"Error closing Modbus client: {e}")
                self._modbus_client = None

            if self._opcua_client:
                try:
                    self._opcua_client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting OPC UA client: {e}")
                self._opcua_client = None

            self._is_connected = False
            logger.info("PLC connector disconnected")

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._command_worker, daemon=True)
        self._worker_thread.start()

        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

        logger.info("PLC command worker started")

    def _command_worker(self):
        while not self._stop_event.is_set():
            try:
                if not self._is_connected:
                    if not self.connect():
                        time.sleep(1)
                        continue

                try:
                    command = self._command_queue.get(timeout=0.1)
                    result = self._execute_command_with_retry(command)
                    self._command_history.append((command, result))
                    self._notify_callbacks(command, result)
                except:
                    continue

            except Exception as e:
                logger.error(f"PLC worker error: {e}", exc_info=True)
                time.sleep(1)

    def _execute_command_with_retry(self, command: PLCCommand) -> PLCCommandResult:
        start_time = time.time()
        last_error = ""

        for attempt in range(1, self._max_retries + 1):
            try:
                if not self._is_connected and not self.connect():
                    raise Exception("PLC not connected")

                result = self._execute_command(command)
                if result.success:
                    self._stats["successful_commands"] += 1
                    return result
                else:
                    last_error = result.error_message

            except Exception as e:
                last_error = str(e)
                logger.warning(f"PLC command attempt {attempt} failed: {e}")

            if attempt < self._max_retries:
                time.sleep(self._retry_delay_ms / 1000.0)

        self._stats["failed_commands"] += 1
        response_time = (time.time() - start_time) * 1000

        logger.error(f"❌ PLC command failed after {self._max_retries} attempts: {last_error}")

        return PLCCommandResult(
            command_id=command.command_id,
            success=False,
            timestamp=time.time(),
            response_time_ms=response_time,
            error_message=last_error
        )

    def _execute_command(self, command: PLCCommand) -> PLCCommandResult:
        start_time = time.time()

        try:
            if self._protocol == PLCProtocol.MODBUS_TCP:
                success = self._execute_modbus_command(command)
            elif self._protocol == PLCProtocol.OPC_UA:
                success = self._execute_opcua_command(command)
            else:
                raise Exception(f"Unsupported protocol: {self._protocol}")

            response_time = (time.time() - start_time) * 1000

            if success:
                if command.command_type == PLCCommandType.REJECT:
                    self._stats["reject_commands"] += 1
                elif command.command_type == PLCCommandType.STOP_LINE:
                    self._stats["stop_line_commands"] += 1
                elif command.command_type == PLCCommandType.ALARM:
                    self._stats["alarm_commands"] += 1

                logger.info(f"✅ PLC command success: {command.command_type.value} "
                           f"(coil={command.coil_address}, value={command.value})")

            return PLCCommandResult(
                command_id=command.command_id,
                success=success,
                timestamp=time.time(),
                response_time_ms=response_time,
                error_message="" if success else "Command failed"
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            logger.error(f"PLC command execution error: {e}", exc_info=True)
            return PLCCommandResult(
                command_id=command.command_id,
                success=False,
                timestamp=time.time(),
                response_time_ms=response_time,
                error_message=str(e)
            )

    def _execute_modbus_command(self, command: PLCCommand) -> bool:
        if not self._modbus_client or not self._is_connected:
            raise Exception("Modbus client not connected")

        if command.defect_codes:
            self._write_defect_codes(command.defect_codes)

        if command.command_type in [PLCCommandType.REJECT, PLCCommandType.STOP_LINE,
                                    PLCCommandType.ALARM, PLCCommandType.RESET]:
            return self._write_coil_with_pulse(command.coil_address, command.value)
        elif command.command_type == PLCCommandType.HEARTBEAT:
            return self._write_single_coil(command.coil_address, command.value)

        return True

    def _write_single_coil(self, address: int, value: bool) -> bool:
        result = self._modbus_client.write_coil(address, value, slave=self._slave_id)
        return not result.isError()

    def _write_coil_with_pulse(self, address: int, value: bool) -> bool:
        result = self._modbus_client.write_coil(address, value, slave=self._slave_id)
        if result.isError():
            return False

        if self._pulse_duration_ms > 0:
            time.sleep(self._pulse_duration_ms / 1000.0)
            result = self._modbus_client.write_coil(address, not value, slave=self._slave_id)
            return not result.isError()

        return True

    def _write_defect_codes(self, defect_codes: List[int]):
        try:
            for i, code in enumerate(defect_codes[:10]):
                self._modbus_client.write_register(
                    self._defect_code_register + i,
                    code,
                    slave=self._slave_id
                )
        except Exception as e:
            logger.warning(f"Failed to write defect codes: {e}")

    def _execute_opcua_command(self, command: PLCCommand) -> bool:
        if not self._opcua_client or not self._is_connected:
            raise Exception("OPC UA client not connected")

        node_id = f"ns=2;s=COIL_{command.coil_address}"
        node = self._opcua_client.get_node(node_id)
        node.set_value(ua.Variant(command.value, ua.VariantType.Boolean))

        if command.defect_codes:
            for i, code in enumerate(command.defect_codes[:10]):
                reg_node_id = f"ns=2;s=REGISTER_{self._defect_code_register + i}"
                reg_node = self._opcua_client.get_node(reg_node_id)
                reg_node.set_value(ua.Variant(code, ua.VariantType.Int16))

        if self._pulse_duration_ms > 0 and command.command_type in [
            PLCCommandType.REJECT, PLCCommandType.STOP_LINE, PLCCommandType.ALARM
        ]:
            time.sleep(self._pulse_duration_ms / 1000.0)
            node.set_value(ua.Variant(not command.value, ua.VariantType.Boolean))

        return True

    def _heartbeat_loop(self):
        heartbeat_state = False
        while not self._stop_event.is_set():
            try:
                if self._is_connected and self._enabled:
                    command = PLCCommand.create(
                        command_type=PLCCommandType.HEARTBEAT,
                        coil_address=self._heartbeat_coil,
                        value=heartbeat_state
                    )
                    self._execute_command(command)
                    heartbeat_state = not heartbeat_state
            except Exception as e:
                logger.debug(f"Heartbeat error: {e}")
                self._is_connected = False

            time.sleep(5)

    def send_reject_command(self, detection_id: str = "",
                            defect_types: Optional[List[DefectType]] = None,
                            alert_action: AlertAction = AlertAction.REJECT) -> Optional[str]:
        if not self._enabled:
            return None

        defect_codes = []
        if defect_types:
            defect_codes = [self.DEFECT_CODE_MAP.get(dt, 199) for dt in defect_types]

        coil_address = self._reject_coil
        if alert_action == AlertAction.STOP_LINE:
            coil_address = self._stop_line_coil

        command = PLCCommand.create(
            command_type=PLCCommandType.REJECT if alert_action == AlertAction.REJECT else PLCCommandType.STOP_LINE,
            detection_id=detection_id,
            defect_codes=defect_codes,
            coil_address=coil_address,
            value=True,
            details={"defect_types": [dt.value for dt in defect_types] if defect_types else []}
        )

        self._stats["total_commands"] += 1
        self._command_queue.put(command)

        logger.info(f"📤 PLC command queued: {command.command_type.value}, "
                   f"defect_codes={defect_codes}, detection_id={detection_id}")

        return command.command_id

    def send_stop_line_command(self, detection_id: str = "",
                               reason: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        command = PLCCommand.create(
            command_type=PLCCommandType.STOP_LINE,
            detection_id=detection_id,
            coil_address=self._stop_line_coil,
            value=True,
            details={"reason": reason}
        )

        self._stats["total_commands"] += 1
        self._command_queue.put(command)

        logger.warning(f"📤 STOP LINE command queued: {reason}")

        return command.command_id

    def send_alarm_command(self, detection_id: str = "",
                           alarm_type: str = "") -> Optional[str]:
        if not self._enabled:
            return None

        command = PLCCommand.create(
            command_type=PLCCommandType.ALARM,
            detection_id=detection_id,
            coil_address=self._alarm_coil,
            value=True,
            details={"alarm_type": alarm_type}
        )

        self._stats["total_commands"] += 1
        self._command_queue.put(command)

        return command.command_id

    def send_reset_command(self) -> Optional[str]:
        if not self._enabled:
            return None

        command = PLCCommand.create(
            command_type=PLCCommandType.RESET,
            coil_address=self._reset_coil,
            value=True
        )

        self._stats["total_commands"] += 1
        self._command_queue.put(command)

        logger.info("📤 RESET command queued")

        return command.command_id

    def register_result_callback(self, callback: Callable[[PLCCommand, PLCCommandResult], None]):
        with self._lock:
            self._result_callbacks.append(callback)

    def _notify_callbacks(self, command: PLCCommand, result: PLCCommandResult):
        for callback in self._result_callbacks:
            try:
                callback(command, result)
            except Exception as e:
                logger.error(f"Error in PLC result callback: {e}", exc_info=True)

    def get_defect_code(self, defect_type: DefectType) -> int:
        return self.DEFECT_CODE_MAP.get(defect_type, 199)

    def get_command_result(self, command_id: str) -> Optional[PLCCommandResult]:
        for cmd, result in reversed(self._command_history):
            if cmd.command_id == command_id:
                return result
        return None

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "enabled": self._enabled,
                "connected": self._is_connected,
                "protocol": self._protocol.value,
                "host": self._host,
                "port": self._port,
                "queue_size": self._command_queue.qsize(),
                "history_size": len(self._command_history)
            }

    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def enabled(self) -> bool:
        return self._enabled
