# Hikrobot camera for the live app: open, grab BGR frames, set exposure / gain, close.
# Taken from ~/label_defect/camera/capture.py and collect.py (same SDK calls), without the white balance
# and rectification parts. The SDK (MvImport) is looked for in <repo>/hik_sdk and ~/hik_sdk.
#
# IMPORTANT: close MVS first (exclusive access).
import sys
from ctypes import POINTER, byref, c_ubyte, cast, memmove, memset, sizeof
from pathlib import Path

import numpy as np

for sdk in (Path(__file__).resolve().parents[1] / "hik_sdk", Path.home() / "hik_sdk"):
    sys.path.append(str(sdk / "MvImport"))

from MvCameraControl_class import *  # noqa: E402,F403

GRAB_TIMEOUT_MS = 3000


def get_float(cam, name):
    value = MVCC_FLOATVALUE()
    cam.MV_CC_GetFloatValue(name, value)
    return value.fCurValue


def get_enum(cam, name):
    value = MVCC_ENUMVALUE()
    cam.MV_CC_GetEnumValue(name, value)
    return value.nCurValue


def open_camera(camera_index):
    # Enumerate, open exclusive, continuous grabbing. Remembers the settings to put them back on close.
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0:
        raise Exception("enum devices fail! ret[0x%x]" % ret)
    if device_list.nDeviceNum == 0:
        raise Exception("find no device!")
    if camera_index >= device_list.nDeviceNum:
        raise Exception("camera %d not found, only %d camera(s)" % (camera_index, device_list.nDeviceNum))
    info = cast(device_list.pDeviceInfo[camera_index], POINTER(MV_CC_DEVICE_INFO)).contents

    cam = MvCamera()
    ret = cam.MV_CC_CreateHandle(info)
    if ret != 0:
        raise Exception("create handle fail! ret[0x%x]" % ret)
    ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
    if ret != 0:
        cam.MV_CC_DestroyHandle()
        raise Exception("open device fail! ret[0x%x]" % ret)

    if info.nTLayerType == MV_GIGE_DEVICE:
        packet_size = cam.MV_CC_GetOptimalPacketSize()
        if int(packet_size) > 0:
            cam.MV_CC_SetIntValue("GevSCPSPacketSize", packet_size)

    cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
    original_settings = {"ExposureAuto": get_enum(cam, "ExposureAuto"), "GainAuto": get_enum(cam, "GainAuto"),
                         "ExposureTime": get_float(cam, "ExposureTime"), "Gain": get_float(cam, "Gain")}
    cam.MV_CC_SetBayerCvtQuality(1)
    ret = cam.MV_CC_StartGrabbing()
    if ret != 0:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
        raise Exception("start grabbing fail! ret[0x%x]" % ret)

    out_frame = MV_FRAME_OUT()
    memset(byref(out_frame), 0, sizeof(out_frame))
    return {"cam": cam, "out_frame": out_frame, "original_settings": original_settings}


def apply_camera_settings(camera, settings):
    # Auto must be Off first, or the camera overwrites the values right away.
    cam = camera["cam"]
    cam.MV_CC_SetEnumValue("ExposureAuto", MV_EXPOSURE_AUTO_MODE_OFF)
    cam.MV_CC_SetEnumValue("GainAuto", MV_GAIN_MODE_OFF)
    cam.MV_CC_SetFloatValue("ExposureTime", settings["ExposureTime"])
    cam.MV_CC_SetFloatValue("Gain", settings["Gain"])


def close_camera(camera):
    # Stop, put the camera back the way we found it, close.
    cam, original = camera["cam"], camera["original_settings"]
    cam.MV_CC_StopGrabbing()
    apply_camera_settings(camera, original)
    cam.MV_CC_SetEnumValue("ExposureAuto", original["ExposureAuto"])
    cam.MV_CC_SetEnumValue("GainAuto", original["GainAuto"])
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()


def grab_bgr(cam, out_frame):
    # One frame converted to BGR8 by the SDK.
    ret = cam.MV_CC_GetImageBuffer(out_frame, GRAB_TIMEOUT_MS)
    if not out_frame.pBufAddr or ret != 0:
        raise Exception("get one frame fail! ret[0x%x]" % ret)
    fi = out_frame.stFrameInfo
    bgr_size = fi.nWidth * fi.nHeight * 3
    convert = MV_CC_PIXEL_CONVERT_PARAM_EX()
    memset(byref(convert), 0, sizeof(convert))
    convert.nWidth, convert.nHeight = fi.nWidth, fi.nHeight
    convert.pSrcData, convert.nSrcDataLen, convert.enSrcPixelType = out_frame.pBufAddr, fi.nFrameLen, fi.enPixelType
    convert.enDstPixelType = PixelType_Gvsp_BGR8_Packed
    convert.pDstBuffer = (c_ubyte * bgr_size)()
    convert.nDstBufferSize = bgr_size
    ret = cam.MV_CC_ConvertPixelTypeEx(convert)
    cam.MV_CC_FreeImageBuffer(out_frame)  # always paired with GetImageBuffer
    if ret != 0:
        raise Exception("convert pixel fail! ret[0x%x]" % ret)
    bgr_buf = (c_ubyte * convert.nDstLen)()
    memmove(byref(bgr_buf), convert.pDstBuffer, convert.nDstLen)
    return np.frombuffer(bgr_buf, dtype=np.uint8).reshape(fi.nHeight, fi.nWidth, 3), fi.nFrameNum
