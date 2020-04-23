#!/usr/bin/env python

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) ifm electronic gmbh
#
# THE PROGRAM IS PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND.
#

# This is a sample code to provide an example
# how to read the reference profile and live profile of the OPD100.
# This code is reduced to it's bare minimum and does not cover all edge cases
# one would expect in production environments.
#
# There are some variables provided, that allow the user to
# change the basic behaviour of this script.
# If no command line arguments are given,
# the default variables are used.
#
# Usage example:
# opd100_loader.py --ip 192.168.0.34 --tcp-port 80 --device-port 4

import json
import atexit
import logging
import time
import binascii
import struct
import argparse

try:
    import httplib as http
except ImportError:
    import http.client as http


MASTER_IP = "192.168.0.99"
MASTER_TCP_PORT = 80
MASTER_DEVICE_PORT = 1  # physical port where the OPD100 is connected to

# Load the reference profile before reading the live profile
ENABLE_LOAD_REF_AT_START = False
# Only load the z values of the live profile, to speedup the readout duration
ENABLE_LOAD_LIVE_ONLY_Z = True
# Enables the plotting of loaded profiles to a file
ENABLE_PLOTTING = False
# Enables the dumping of loaded profiles to a json file
ENABLE_JSON_DUMP = True

PD_IN_SIZE = 32
MAX_FRAMES = 24
OPD_INT_TO_FLOAT = 0.00244
OPD_INT_TO_FLOAT_X_OFFSET = -79.93318

IOL_VENDOR_ID_IFM = 310
IOL_DEVICD_ID_OPD100 = 1260
IOL_DEVICD_ID_OPD100_TOOL = 1276
IOL_DEVICE_IDS = [IOL_DEVICD_ID_OPD100, IOL_DEVICD_ID_OPD100_TOOL]
IOL_DEVICE_MAPPING = {IOL_DEVICD_ID_OPD100: IOL_DEVICD_ID_OPD100_TOOL}

DATATYPE_LIVE_X = 1
DATATYPE_LIVE_Z = 2
DATATYPE_REF_X = 3
DATATYPE_REF_Z = 4
DATATYPE_FLOATING_X = 5
DATATYPE_FLOATING_Z = 6

VDSM_NO_CHECK = 0
VDSM_V10_DEVICE_CHECK = 1
VDSM_V11_DEVICE_CHECK = 2
VDSM_V11_DEVICE_CHECK_BACKUP_RESTORE = 3
VDSM_V11_DEVICE_CHECK_RESTORE = 4


class NextState:
    WAIT = 0
    TAKE_RESULT = 1
    MODIFY_DEVICE = 2
    FAIL = 4


class IoLinkEth:
    cid = 0
    cid_min = 1
    cid_max = 32767

    headers = {"Content-type": "application/json"}

    def __init__(self, master_ip, master_tcp_port, device_port):
        self.http_con = http.HTTPConnection(master_ip, master_tcp_port)
        self.device_port = device_port

    def getAndIncreaseCID(self):
        self.cid = max(self.cid_min, min(self.cid + 1, self.cid_max))
        return self.cid

    def readProcessInData(self):
        data = {
            "code": 10,
            "cid": self.getAndIncreaseCID(),
            "adr": "/iolinkmaster/port[{}]/iolinkdevice/pdin/getdata".format(self.device_port)
        }
        self.http_con.request("POST", "/", json.dumps(data), self.headers)
        response = self.http_con.getresponse()
        assert response.status == 200,\
            "{} returned {}".format(data["adr"], response.status)
        response_json = json.loads(response.read().decode())
        return response_json["data"]["value"]

    def writeProcessOutData(self, pdout):
        pdout_string = ""
        try:
            pdout_string = pdout.hex()
        except:
            pdout_string = binascii.hexlify(pdout)
        data = {
            "code": 10,
            "cid": self.getAndIncreaseCID(),
            "adr": "/iolinkmaster/port[{}]/iolinkdevice/pdout/setdata".format(self.device_port),
            "data": {
                "newvalue": pdout_string
            }
        }
        self.http_con.request("POST", "/", json.dumps(data), self.headers)
        response = self.http_con.getresponse()
        assert response.status == 200,\
            "{} http returned {}".format(data["adr"], response.status)
        response_json = json.loads(response.read().decode())
        json_code = response_json["code"]
        assert json_code == 200,\
            "{} json returned {}".format(data["adr"], json_code)
        return

    def getDataMulti(self, vars):
        data = {
            "code": 10,
            "cid": self.getAndIncreaseCID(),
            "adr": "/getdatamulti",
            "data": {
                "datatosend": [
                    "/iolinkmaster/port[{}]/{}".format(self.device_port, var) for var in vars
                ]
            }
        }
        self.http_con.request("POST", "/", json.dumps(data), self.headers)
        response = self.http_con.getresponse()
        assert response.status == 200,\
            "{} http returned {}".format(data["adr"], response.status)
        response_json = json.loads(response.read().decode())
        results = dict()
        for var in vars:
            path = "/iolinkmaster/port[{}]/{}".format(self.device_port, var)
            code = response_json["data"][path]["code"]
            assert path in response_json["data"],\
                "response_json['data'] doesn't contain {}".format(path)
            assert code == 200,\
                "{} json returned {}".format(data["adr"], code)
            results[var] = response_json["data"][path]["data"]
        return results

    def setData(self, uri, value):
        data = {
            "code": 10,
            "cid": self.getAndIncreaseCID(),
            "adr": uri.format(self.device_port),
            "data": {
                "newvalue": value
            }
        }
        self.http_con.request("POST", "/", json.dumps(data), self.headers)
        response = self.http_con.getresponse()
        assert response.status == 200,\
            "{} http returned {}".format(data["adr"], response.status)
        response_json = json.loads(response.read().decode())
        json_code = response_json["code"]
        assert json_code == 200,\
            "{} json returned {}".format(data["adr"], json_code)
        pass

    def setDeviceID(self, vendor_id, device_id, vdsm):
        self.setData(
            "/iolinkmaster/port[{}]/validation_vendorid/setdata", vendor_id)
        self.setData(
            "/iolinkmaster/port[{}]/validation_deviceid/setdata", device_id)
        self.setData(
            "/iolinkmaster/port[{}]/validation_datastorage_mode/setdata", vdsm)
        time.sleep(1)
        pass

    def resetDeviceID(self, vendor_id, device_id, vdsm):
        self.setData(
            "/iolinkmaster/port[{}]/validation_datastorage_mode/setdata", VDSM_NO_CHECK)
        self.setDeviceID(vendor_id, device_id, vdsm)
        self.setData(
            "/iolinkmaster/port[{}]/validation_datastorage_mode/setdata", VDSM_NO_CHECK)
        self.setData(
            "/iolinkmaster/port[{}]/validation_datastorage_mode/setdata", vdsm)
        time.sleep(1)
        pass


class ProfileLoader:
    pdout = bytearray()
    ref_mode = False
    live_mode_only_z = False

    # contains the current result
    framing_buffer_x = bytearray()
    framing_buffer_z = bytearray()
    # contains the last valid result
    last_matched = False
    last_match_value = 0
    last_framing_buffer_x = bytearray()
    last_framing_buffer_z = bytearray()

    def __init__(self, iolink):
        self.iolink = iolink

    def writePDout(self):
        self.iolink.writeProcessOutData(self.pdout)
        pass

    def getDtQueue(self):
        if self.ref_mode:
            return [DATATYPE_REF_X, DATATYPE_REF_Z]
        else:
            return [DATATYPE_LIVE_Z] if self.live_mode_only_z else [DATATYPE_LIVE_X, DATATYPE_LIVE_Z]

    def buildPDout(self):
        overall_frame_number = int(
            (len(self.framing_buffer_x) + len(self.framing_buffer_z))/PD_IN_SIZE)

        dtQueue = self.getDtQueue()
        dtIndex = int(overall_frame_number / MAX_FRAMES)
        dt = dtQueue[dtIndex] if dtIndex < len(dtQueue) else dtQueue[0]
        frame_number = int(overall_frame_number % MAX_FRAMES + 1)

        forceTrigger = False
        self.pdout = bytearray()
        self.pdout.append(1 << 3 if forceTrigger else 0)
        self.pdout.append(dt << 5 | frame_number)
        # print("dt: {} frameNo: {} bytes: {}".format(
        # dt, frame_number, self.pdout))
        pass

    def processPDIn(self, pdin_str):
        pdin = bytearray.fromhex(pdin_str)

        # assert len(pdin) == PD_IN_SIZE, "unexpected framing size"
        if len(pdin) != PD_IN_SIZE:
            return NextState.FAIL

        overall_frame_number = int(
            (len(self.framing_buffer_x) + len(self.framing_buffer_z)) / PD_IN_SIZE)
        requested_frame_number = self.pdout[1] & 0x1f if len(
            self.pdout) > 0 else 0xff
        requested_dt = self.pdout[1] >> 5 if len(self.pdout) > 0 else 0xff
        frame_number = pdin[31] & 0x1f
        dt = pdin[31] >> 5

        if requested_frame_number != frame_number or requested_dt != dt:
            self.buildPDout()
            return NextState.MODIFY_DEVICE

        # if ref_mode is requested, but received dt is not a ref, clear all and retry
        if self.ref_mode and not (dt == DATATYPE_REF_X or dt == DATATYPE_REF_Z):
            self.framing_buffer_x = bytearray()
            self.framing_buffer_z = bytearray()
            self.buildPDout()
            return NextState.MODIFY_DEVICE
        if not self.ref_mode and not (dt == DATATYPE_LIVE_X or dt == DATATYPE_LIVE_Z):
            self.framing_buffer_x = bytearray()
            self.framing_buffer_z = bytearray()
            self.buildPDout()
            return NextState.MODIFY_DEVICE

        if frame_number == 1:
            self.last_matched = bool(pdin[29] >> 7)
            self.last_match_value = pdin[29] & 0x7f
            # print("matched:{} value:{}".format(matched, match_value))

        if frame_number != 1 and (dt == DATATYPE_REF_X or dt == DATATYPE_LIVE_X) and len(self.framing_buffer_x) == 0:
            self.buildPDout()
            return NextState.MODIFY_DEVICE
        if frame_number != 1 and (dt == DATATYPE_REF_Z or dt == DATATYPE_LIVE_Z) and len(self.framing_buffer_z) == 0:
            self.buildPDout()
            return NextState.MODIFY_DEVICE

        if frame_number <= MAX_FRAMES and overall_frame_number <= MAX_FRAMES * len(self.getDtQueue()):
            if dt == DATATYPE_LIVE_X or dt == DATATYPE_REF_X:
                self.framing_buffer_x.extend(pdin)
            else:
                self.framing_buffer_z.extend(pdin)
            if overall_frame_number + 1 == MAX_FRAMES * len(self.getDtQueue()):
                self.last_framing_buffer_x = self.framing_buffer_x[:]
                self.last_framing_buffer_z = self.framing_buffer_z[:]
                self.framing_buffer_x = bytearray()
                self.framing_buffer_z = bytearray()
                self.buildPDout()
                return NextState.MODIFY_DEVICE | NextState.TAKE_RESULT

            self.buildPDout()
            return NextState.MODIFY_DEVICE

        iolink.buildPDout()
        return NextState.WAIT


def plot(live_x, live_z, ref_x, ref_z, matched, matching_value):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.style.use("ggplot")
        fig, ax = plt.subplots()

        if len(live_x) == 0:
            live_x = list(reversed(list(range(0, 352))))
        if len(live_x) == len(live_z):
            ax.plot(live_x, live_z, label="Live")

        if len(ref_x) == len(ref_z):
            ax.plot(ref_x, ref_z, label="Reference")
        ax.set_xlabel("x" if ENABLE_LOAD_LIVE_ONLY_Z else "x [mm]")
        ax.set_ylabel("z [mm]")
        ax.set_title("matched:{} \n match_value:{}%".format(
            matched, matching_value))
        ax.legend(loc="upper right")

        plt.show()
        plt.savefig("result.png")
        plt.close()
    except Exception as e:
        logging.exception("Can't plot the profile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ifm OPD100 profile loader')
    parser.add_argument("--ip", default=MASTER_IP,
                        help="IP address of the ifm ethernet master.")
    parser.add_argument("--tcp-port", default=MASTER_TCP_PORT,
                        help="TCP port of the ifm ethernet master.")
    parser.add_argument("--device-port", default=MASTER_TCP_PORT,
                        help="Physical port, the OPD100 is connected to.")
    args = parser.parse_args()
    if ("{}".format(args.ip) != ""):
        print('IP: {}'.format(args.ip))
        print('port: {}'.format(args.tcp_port))
        print('device port: {}'.format(args.device_port))
    else:
        print("No IP address specified to be set. Using default values")

    original_vendor_id = 0
    original_device_id = 0
    original_vdsm = 0
    changed_device_id = False

    reference_profile_x = list()
    reference_profile_z = list()

    def cleanup():
        print("cleanup")
        print(original_vendor_id, original_device_id, original_vdsm)
        if changed_device_id and original_vendor_id != 0 and original_device_id != 0 and original_vdsm != 0:
            iolink = IoLinkEth(args.ip, args.tcp_port, args.device_port)
            iolink.resetDeviceID(original_vendor_id,
                                 original_device_id, original_vdsm)

    atexit.register(cleanup)

    iolink = IoLinkEth(args.ip, args.tcp_port, args.device_port)
    loader = ProfileLoader(iolink)

    vendor_device_vdsm = iolink.getDataMulti([
        "iolinkdevice/vendorid",
        "iolinkdevice/deviceid",
        "validation_datastorage_mode"
    ])
    original_vendor_id = vendor_device_vdsm["iolinkdevice/vendorid"]
    original_device_id = vendor_device_vdsm["iolinkdevice/deviceid"]
    original_vdsm = vendor_device_vdsm["validation_datastorage_mode"]
    assert original_vendor_id == IOL_VENDOR_ID_IFM, "this device is not an ifm device"
    assert original_device_id in IOL_DEVICE_IDS, "this device is not an OPD"
    print(vendor_device_vdsm)

    # we have to change the device_id to _TOOL
    if vendor_device_vdsm["iolinkdevice/deviceid"] in IOL_DEVICE_MAPPING:
        try:
            iolink.setDeviceID(vendor_device_vdsm["iolinkdevice/vendorid"],
                               IOL_DEVICE_MAPPING[vendor_device_vdsm["iolinkdevice/deviceid"]],
                               vendor_device_vdsm["validation_datastorage_mode"])
            changed_device_id = True
        except Exception as e:
            logging.exception("error setting device_id")
            # calls cleanup() automatically
            exit

    loader.ref_mode = ENABLE_LOAD_REF_AT_START
    loader.live_mode_only_z = ENABLE_LOAD_LIVE_ONLY_Z

    start = time.time()
    end = time.time()
    try:
        while True:
            time.sleep(10 / 1000)
            nextState = loader.processPDIn(iolink.readProcessInData())
            if nextState & NextState.MODIFY_DEVICE:
                loader.writePDout()
            if nextState & NextState.TAKE_RESULT:
                end = time.time()
                print("read: {} ms".format(end - start))
                start = time.time()

                def extract_result(buffer):
                    final_result = bytearray()
                    for i in list(range(0, len(buffer)))[0::PD_IN_SIZE]:
                        start_offset = 14 if i / PD_IN_SIZE == MAX_FRAMES - 1 else 0
                        end_offset = 27 if i == 0 else 29
                        # this also reverts the byte-order, so we have little-endian
                        final_result.extend(
                            reversed(buffer[i + start_offset:i + end_offset + 1]))
                    return final_result

                final_result_x = extract_result(loader.last_framing_buffer_x)
                final_result_z = extract_result(loader.last_framing_buffer_z)

                def convert(buffer, toFloatLambda):
                    x = list()
                    for i in list(range(0, len(buffer)))[0::2]:
                        unpacked = struct.unpack("<H", buffer[i:i + 2])
                        value = float(unpacked[0])
                        value = float("nan") if value > 65519 else value
                        value = toFloatLambda(value)
                        x.append(value)
                    return x
                # convert to uint16
                x = convert(final_result_x, lambda x: x *
                            OPD_INT_TO_FLOAT + (OPD_INT_TO_FLOAT_X_OFFSET))
                z = convert(final_result_z, lambda z: z * OPD_INT_TO_FLOAT)

                if loader.ref_mode:
                    loader.ref_mode = False
                    reference_profile_x = x[:]
                    reference_profile_z = z[:]

                # dump as json
                if ENABLE_JSON_DUMP:
                    data = {
                        "matched": loader.last_matched,
                        "match_value": loader.last_match_value,
                        "ref_x": reference_profile_x,
                        "ref_z": reference_profile_z,
                        "live_x": x,
                        "live_z": z
                    }
                    with open("result.json", "w") as file:
                        file.write(json.dumps(data, indent=4, sort_keys=True))
                        file.close()

                # plot
                if ENABLE_PLOTTING:
                    plot(x, z, reference_profile_x, reference_profile_z,
                         loader.last_matched, loader.last_match_value)

    except Exception as e:
        logging.exception("error processing data")
        # calls cleanup() automatically
        exit
