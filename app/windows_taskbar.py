from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", GUID), ("pid", wintypes.DWORD)]


class _PROPVARIANT_VALUE(ctypes.Union):
    _fields_ = [("pwszVal", ctypes.c_wchar_p), ("pointer", ctypes.c_void_p)]


class PROPVARIANT(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("vt", wintypes.WORD),
        ("wReserved1", wintypes.WORD),
        ("wReserved2", wintypes.WORD),
        ("wReserved3", wintypes.WORD),
        ("value", _PROPVARIANT_VALUE),
    ]


def _guid(data1: int, data2: int, data3: int, tail: tuple[int, ...]) -> GUID:
    return GUID(data1, data2, data3, (ctypes.c_ubyte * 8)(*tail))


APP_MODEL_FORMAT = _guid(
    0x9F4C2855,
    0x9F79,
    0x4B39,
    (0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3),
)
IID_IPROPERTY_STORE = _guid(
    0x886D8EEB,
    0x8CF2,
    0x4446,
    (0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xCF, 0x99),
)

PKEY_RELAUNCH_COMMAND = PROPERTYKEY(APP_MODEL_FORMAT, 2)
PKEY_RELAUNCH_ICON_RESOURCE = PROPERTYKEY(APP_MODEL_FORMAT, 3)
PKEY_APP_USER_MODEL_ID = PROPERTYKEY(APP_MODEL_FORMAT, 5)
VT_LPWSTR = 31


def _string_variant(value: str) -> PROPVARIANT:
    variant = PROPVARIANT()
    variant.vt = VT_LPWSTR
    variant.pwszVal = value
    return variant


def set_taskbar_identity(
    hwnd: int,
    app_id: str,
    icon_resource: str,
    relaunch_command: str,
) -> bool:
    """Set the explicit identity and relaunch icon on one top-level window."""
    if sys.platform != "win32" or not hwnd:
        return False

    shell32 = ctypes.windll.shell32
    shell32.SHGetPropertyStoreForWindow.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    shell32.SHGetPropertyStoreForWindow.restype = ctypes.c_long
    store = ctypes.c_void_p()
    result = shell32.SHGetPropertyStoreForWindow(
        ctypes.c_void_p(hwnd), ctypes.byref(IID_IPROPERTY_STORE), ctypes.byref(store)
    )
    if result < 0 or not store.value:
        return False

    vtable = ctypes.cast(
        store, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    set_value_type = ctypes.WINFUNCTYPE(
        ctypes.c_long,
        ctypes.c_void_p,
        ctypes.POINTER(PROPERTYKEY),
        ctypes.POINTER(PROPVARIANT),
    )
    commit_type = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p)
    release_type = ctypes.WINFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)
    set_value = set_value_type(vtable[6])
    commit = commit_type(vtable[7])
    release = release_type(vtable[2])

    try:
        # Microsoft requires relaunch properties to be assigned before ID;
        # setting ID last tells the taskbar to refresh this window's identity.
        values = (
            (PKEY_RELAUNCH_COMMAND, relaunch_command),
            (PKEY_RELAUNCH_ICON_RESOURCE, icon_resource),
            (PKEY_APP_USER_MODEL_ID, app_id),
        )
        for key, text in values:
            variant = _string_variant(text)
            if set_value(store, ctypes.byref(key), ctypes.byref(variant)) < 0:
                return False
        return commit(store) >= 0
    finally:
        release(store)
