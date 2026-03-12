"""Windows 콘솔 창 숨김 모듈

스크립트 최상단에서 import하면 콘솔 창이 즉시 사라진다.
python.exe로 실행해도, Task Scheduler에서 실행해도 동일하게 동작.

사용법:
    import hide_console  # 이것만으로 창 숨김
"""

import os
import sys

if os.name == "nt":
    try:
        import ctypes
        # 현재 프로세스의 콘솔 창 핸들 획득 후 숨김
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE = 0
    except Exception:
        pass  # 콘솔이 없는 환경(pythonw.exe)에서는 무시

    # stdout/stderr가 없는 경우(pythonw.exe) → devnull로 리다이렉트
    if sys.stdout is None or (hasattr(sys.stdout, "fileno") and not sys.stdout.writable()):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None or (hasattr(sys.stderr, "fileno") and not sys.stderr.writable()):
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
