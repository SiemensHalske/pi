import json
import pathlib
from collections import deque
from dataclasses import dataclass, field as _dc_field
from datetime import datetime, timezone
from enum import IntEnum
from threading import Lock


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


_LEVEL_STYLES: dict = {
    LogLevel.DEBUG: "dim white",
    LogLevel.INFO: "cyan",
    LogLevel.SUCCESS: "bold green",
    LogLevel.WARNING: "bold yellow",
    LogLevel.ERROR: "bold red",
    LogLevel.CRITICAL: "bold white on red",
}
_LEVEL_LABELS: dict = {
    LogLevel.DEBUG: "DEBUG   ",
    LogLevel.INFO: "INFO    ",
    LogLevel.SUCCESS: "SUCCESS ",
    LogLevel.WARNING: "WARNING ",
    LogLevel.ERROR: "ERROR   ",
    LogLevel.CRITICAL: "CRITICAL",
}


@dataclass
class LogRecord:
    level: LogLevel
    subsystem: str
    message: str
    timestamp: datetime = _dc_field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict = _dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.name,
            "subsystem": self.subsystem,
            "message": self.message,
            **self.extra,
        }

    def to_line(self) -> str:
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        extra = ("  " + "  ".join(f"{k}={v}" for k, v in self.extra.items())) if self.extra else ""
        return f"[{ts}] [{self.level.name:<8}] [{self.subsystem:<14}] {self.message}{extra}"


class _FileHandler:
    MAX_BYTES = 5 * 1024 * 1024
    KEEP_BACKUPS = 3

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._fp = self._open()

    def _open(self):
        return open(self._path, "a", encoding="utf-8", buffering=1)

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            try:
                self._fp.write(record.to_line() + "\n")
                self._rotate_if_needed()
            except OSError:
                pass

    def _rotate_if_needed(self) -> None:
        try:
            if self._path.stat().st_size < self.MAX_BYTES:
                return
            self._fp.close()
            for i in range(self.KEEP_BACKUPS - 1, 0, -1):
                src = self._path.parent / f"{self._path.stem}.{i}{self._path.suffix}"
                dst = self._path.parent / f"{self._path.stem}.{i + 1}{self._path.suffix}"
                if src.exists():
                    src.replace(dst)
            self._path.replace(self._path.parent / f"{self._path.stem}.1{self._path.suffix}")
            self._fp = self._open()
        except OSError:
            pass

    def close(self) -> None:
        with self._lock:
            try:
                self._fp.close()
            except OSError:
                pass


class _PostgresHandler:
    _CREATE_SQL = """
        CREATE TABLE IF NOT EXISTS game_logs (
            id        BIGSERIAL    PRIMARY KEY,
            ts        TIMESTAMPTZ  NOT NULL DEFAULT now(),
            level     TEXT         NOT NULL,
            subsystem TEXT         NOT NULL,
            message   TEXT         NOT NULL,
            extra     JSONB        NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS game_logs_ts_idx  ON game_logs (ts  DESC);
        CREATE INDEX IF NOT EXISTS game_logs_lvl_idx ON game_logs (level);
        CREATE INDEX IF NOT EXISTS game_logs_sub_idx ON game_logs (subsystem);
    """
    _INSERT_SQL = (
        "INSERT INTO game_logs (ts, level, subsystem, message, extra) "
        "VALUES (%s, %s, %s, %s, %s)"
    )

    def __init__(self) -> None:
        self._conn = None
        self._dsn: str | None = None
        self._lock = Lock()
        self._queue: deque = deque(maxlen=2000)

    @property
    def connected(self) -> bool:
        return self._conn is not None

    def connect(self, dsn: str) -> bool:
        try:
            import psycopg as _pg
        except ImportError:
            try:
                import psycopg2 as _pg
            except ImportError:
                return False
        try:
            conn = _pg.connect(dsn, autocommit=True)
            with self._lock:
                self._conn = conn
                self._dsn = dsn
            self._ensure_table()
            self._flush_queue()
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

    def emit(self, record: LogRecord) -> None:
        with self._lock:
            if self._conn is None:
                self._queue.append(record)
                return
            self._write(record)

    def _write(self, record: LogRecord) -> None:
        try:
            cur = self._conn.cursor()
            cur.execute(
                self._INSERT_SQL,
                (
                    record.timestamp,
                    record.level.name,
                    record.subsystem,
                    record.message,
                    json.dumps(record.extra),
                ),
            )
        except Exception:
            self._conn = None
            self._queue.append(record)

    def _ensure_table(self) -> None:
        try:
            cur = self._conn.cursor()
            cur.execute(self._CREATE_SQL)
        except Exception:
            pass

    def _flush_queue(self) -> None:
        while self._queue:
            rec = self._queue.popleft()
            self._write(rec)


class _BoundLogger:
    __slots__ = ("_log", "_sub")

    def __init__(self, log: "GameLogger", sub: str) -> None:
        self._log = log
        self._sub = sub

    def debug(self, msg: str, **kw):
        return self._log.debug(self._sub, msg, **kw)

    def info(self, msg: str, **kw):
        return self._log.info(self._sub, msg, **kw)

    def success(self, msg: str, **kw):
        return self._log.success(self._sub, msg, **kw)

    def warning(self, msg: str, **kw):
        return self._log.warning(self._sub, msg, **kw)

    def error(self, msg: str, **kw):
        return self._log.error(self._sub, msg, **kw)

    def critical(self, msg: str, **kw):
        return self._log.critical(self._sub, msg, **kw)


class GameLogger:
    BUFFER_SIZE = 500

    def __init__(
        self,
        log_file: pathlib.Path | str = pathlib.Path("logs") / "game.log",
        console_level: LogLevel = LogLevel.DEBUG,
        file_level: LogLevel = LogLevel.DEBUG,
    ) -> None:
        from rich.console import Console as _RichConsole

        self._rich = _RichConsole(stderr=False, highlight=False)
        self._file_h = _FileHandler(pathlib.Path(log_file))
        self.pg = _PostgresHandler()
        self._buf: deque = deque(maxlen=self.BUFFER_SIZE)
        self._lock = Lock()
        self._console_level = console_level
        self._file_level = file_level

    def _emit(self, level: LogLevel, subsystem: str, msg: str, **extra) -> LogRecord:
        rec = LogRecord(level=level, subsystem=subsystem, message=str(msg), extra=extra)
        with self._lock:
            self._buf.append(rec)
        if level >= self._console_level:
            self._render(rec)
        if level >= self._file_level:
            self._file_h.emit(rec)
        self.pg.emit(rec)
        return rec

    def _render(self, rec: LogRecord) -> None:
        ts = rec.timestamp.strftime("%H:%M:%S")
        style = _LEVEL_STYLES[rec.level]
        label = _LEVEL_LABELS[rec.level]
        sub = f"[dim]{rec.subsystem:<14}[/dim]"
        extra_parts = (
            "  " + "  ".join(f"[dim]{k}[/dim]=[cyan]{v}[/cyan]" for k, v in rec.extra.items())
            if rec.extra
            else ""
        )
        self._rich.print(f"[dim]{ts}[/dim]  [{style}]{label}[/{style}]  {sub}  {rec.message}{extra_parts}")

    def debug(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.DEBUG, sub, msg, **kw)

    def info(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.INFO, sub, msg, **kw)

    def success(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.SUCCESS, sub, msg, **kw)

    def warning(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.WARNING, sub, msg, **kw)

    def error(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.ERROR, sub, msg, **kw)

    def critical(self, sub: str, msg: str, **kw) -> LogRecord:
        return self._emit(LogLevel.CRITICAL, sub, msg, **kw)

    def subsystem(self, name: str) -> _BoundLogger:
        return _BoundLogger(self, name)

    def set_level(self, console: LogLevel | None = None, file: LogLevel | None = None) -> None:
        if console is not None:
            self._console_level = console
        if file is not None:
            self._file_level = file

    def recent(self, n: int = 50, level: LogLevel | None = None, sub: str | None = None) -> list:
        with self._lock:
            records = list(self._buf)
        if level is not None:
            records = [r for r in records if r.level >= level]
        if sub is not None:
            records = [r for r in records if r.subsystem == sub]
        return records[-n:]

    def close(self) -> None:
        self._file_h.close()
        self.pg.disconnect()


log: GameLogger = GameLogger()
