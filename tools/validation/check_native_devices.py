"""Check all native haptics v4 packets and host device workflows against source.

Network checks use private loopback sockets. Serial checks use a pseudo-terminal;
firmware lifecycle checks use an isolated fake PlatformIO process. No phone or
wearable is opened, reset or flashed by this gate.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/"mir-desktop-app/src")]
from mir_desktop_app import haptics_protocol as p
from mir_desktop_app.firmware import parse_endpoint_line,flash_command
from mir_desktop_app.haptic_sender import ClockSyncExchange
from mir_desktop_app.input_devices import InputEventRecorder, EvdevInputSource
from evdev import ecodes
import evdev


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bin",type=Path,default=ROOT/"mir-desktop-app/native-workflows/target/debug/mir-native-workflows")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def native(request,*,reject=False,env=None):
        r=subprocess.run([str(args.native_bin)],input=json.dumps(request),capture_output=True,text=True,env=env,timeout=20)
        if reject:assert r.returncode!=0,(request,r.stdout);return r.stderr
        assert r.returncode==0,r.stderr
        return json.loads(r.stdout)
    rng=random.Random(72480);packets=[]
    for _ in range(40):
        seq=rng.randrange(65536);epoch=rng.randrange(65536);host=rng.randrange(2**63,2**64-10000)
        packets.extend([
            ("actuation",p.ActuationPacket(p.BeatEvent(rng.randrange(1,3)),seq,epoch,rng.randrange(2**32))),
            ("config",p.ActuationProfile(beat_motor_count=rng.randrange(5),beat_duration_ms=rng.randrange(1,2001),beat_intensity=rng.randrange(1,256),downbeat_motor_count=rng.randrange(5),downbeat_duration_ms=rng.randrange(1,2001),downbeat_intensity=rng.randrange(1,256),buzzer_events=p.BuzzerEvent(rng.randrange(4)),buzzer_tone_hz=rng.randrange(20,20001),telemetry_enabled=bool(seq%2)).configuration(sequence=seq,scope=p.ConfigScope(seq%4))),
            ("ack",p.DeviceAck(rng.choice(list(p.AckStatus)),seq,epoch)),
            ("discovery",p.DiscoveryPacket(p.DiscoveryMessage.PING,seq)),
            ("dance-config",p.DanceOneSettings(enabled=bool(seq%2),predictor_method=p.DancePredictorMethod(seq%3),communication_time_us=seq,actuation_time_us=epoch,cue_lead_time_us=seq).configuration(sequence=seq)),
            ("clock-sync",p.ClockSyncPacket(p.ClockSyncMessage.REQUEST,seq,host)),
            ("clock-sync",p.ClockSyncPacket(p.ClockSyncMessage.RESPONSE,seq,host,host+200,host+220)),
            ("event-timing",p.EventTimingPacket(rng.choice(list(p.EventTimingStatus)),seq,epoch,host,host+100,host+200)),
            ("dance-cue-timing",p.DanceCueTimingPacket(p.DanceCueStatus.DISPATCHED,seq,epoch,seq,host,host+100,host+200,500000,2000,32767)),
            ("dance-cue-timing",p.DanceCueTimingPacket(p.DanceCueStatus.DROPPED_LATE,seq,epoch,seq,host,host+100,0,500000,2000,0)),
        ])
    requests=[];expected=[]
    for name,packet in packets:
        raw={"type":name,**asdict(packet)};wire=packet.to_bytes()
        requests.extend([{"operation":"wire-encode","packet":raw},{"operation":"wire-decode","hex":wire.hex()}]);expected.extend([{"hex":wire.hex()},raw])
        for i in range(len(wire)):
            bad=bytearray(wire);bad[i]^=1
            requests.append({"operation":"wire-check","hex":bad.hex()});expected.append({"valid":False})
    # Invalid semantics with a valid envelope/CRC, not merely corrupt bytes.
    for name,packet in packets[0:10]:
        bad=bytearray(packet.to_bytes());bad[3]=255;bad[-2:]=p.crc16_ccitt(bad[:-2]).to_bytes(2,"big")
        try:p.decode_packet(bytes(bad));valid=True
        except p.ProtocolError:valid=False
        requests.append({"operation":"wire-check","hex":bad.hex()});expected.append({"valid":valid})
    for line in ["WiFi connected, IP 192.168.1.55", "AP started, SSID mir-haptics, IP 192.168.4.1", "\tIP 001.002.003.004 \r", "IP 256.1.1.1", "IP 1.2.3", "RIP 1.2.3.4", "IP 1.2.3.4foo", "other IP 0.0.0.0", "AP SSID with IP\t127.0.0.1"]:
        parsed=parse_endpoint_line(line);requests.append({"operation":"parse-endpoint","line":line});expected.append(None if parsed is None else asdict(parsed))
    for monitor in (False,True):
        for port in (None,"/dev/ttyACM7"):
            requests.append({"operation":"flash-command","project":"a project; $(literal)","environment":"test","port":port,"monitor":monitor})
            expected.append(flash_command("a project; $(literal)",environment="test",port=port,monitor=monitor))
    for enabled in (False,True):
        for method in list(p.DancePredictorMethod):
            settings=p.DanceOneSettings(enabled=enabled,predictor_method=method)
            requests.append({"operation":"dance-config","settings":asdict(settings),"sequence":42});expected.append({"type":"dance-config",**asdict(settings.configuration(sequence=42))})
    for telemetry in (False,True):
        profile=p.ActuationProfile(telemetry_enabled=telemetry)
        requests.append({"operation":"profile-config","profile":asdict(profile),"sequence":65535});expected.append({"type":"config",**asdict(profile.configuration(sequence=65535))})
    for _ in range(30):
        host=rng.randrange(1,2**63);rx=host+rng.randrange(-100000,100000);tx=rx+100;received=host+800
        ex=ClockSyncExchange(10,host,rx,tx,received)
        response=p.ClockSyncPacket(p.ClockSyncMessage.RESPONSE,10,host,rx,tx)
        requests.append({"operation":"clock-exchange","packet":{"type":"clock-sync",**asdict(response)},"received_us":received})
        expected.append({**asdict(ex),"receiver_turnaround_us":ex.receiver_turnaround_us,"network_rtt_us":ex.network_rtt_us,"receiver_minus_host_offset_us":ex.receiver_minus_host_offset_us,"offset_uncertainty_us":ex.offset_uncertainty_us})
    # Independently execute the source recorder and axis mapper, including arm,
    # retained debounce state, bounced controls, repeats, deadzones and direction changes.
    for debounce in (0.0,.030,.0300000005):
        recorder=InputEventRecorder(debounce_seconds=debounce);actions=[];outputs=[]
        def record(time,control="KEY_SPACE",source="  evdev:fixture  "):
            e={"source":source,"control":control,"monotonic_ns":time,"value":.5}
            actions.append({"kind":"record","event":e})
            out=recorder.record(source,control,monotonic_ns=time,value=.5)
            outputs.append(None if out is None else out.as_dict())
        record(1_000_000_000)
        actions.append({"kind":"arm"});outputs.append(None);recorder.arm()
        for offset in (0,1,29_999_999,30_000_000,30_000_001,100_000_000,99_000_000):record(1_000_000_000+offset)
        record(1_100_000_000,"KEY_ENTER")
        actions.append({"kind":"disarm"});outputs.append([e.as_dict() for e in recorder.disarm()])
        record(1_200_000_000)
        actions.append({"kind":"arm","clear":False});outputs.append(None);recorder.arm(clear=False)
        record(1_100_000_001)
        actions.append({"kind":"snapshot"});outputs.append([e.as_dict() for e in recorder.snapshot()])
        requests.append({"operation":"input-replay","actions":actions,"debounce_seconds":debounce});expected.append(outputs)
    source_events=[];source=EvdevInputSource("synthetic",source_events.append)
    source._device=SimpleNamespace(absinfo=lambda code:SimpleNamespace(min=-100,max=100))
    actions=[];outputs=[]
    for i,value in enumerate((0,49,50,100,99,0,-49,-50,-100,-100,100,0,50)):
        timestamp=1_000_000_000+i*100_000_000
        n=len(source_events);source._handle_axis(SimpleNamespace(code=ecodes.ABS_X,value=value),"evdev:fixture",timestamp,evdev)
        actions.append({"kind":"axis","source":"evdev:fixture","code":ecodes.ABS_X,"value":value,"minimum":-100,"maximum":100,"monotonic_ns":timestamp})
        outputs.append(source_events[-1].as_dict() if len(source_events)>n else None)
    for status,note,velocity in [(0x90,60,127),(0x99,30,64),(0x90,60,0),(0x80,60,100),(0xb0,60,100)]:
        actions.append({"kind":"midi","identifier":"fixture","bytes":[status,note,velocity],"monotonic_ns":2_000_000_000})
        outputs.append({"source":"midi:fixture","control":f"note_{note}","monotonic_ns":2_000_000_000,"value":velocity/127.0} if status&0xf0==0x90 and velocity>0 else None)
    requests.append({"operation":"input-replay","actions":actions});expected.append(outputs)
    actual=native({"operation":"batch","requests":requests})
    assert len(actual)==len(expected)
    for i,(a,e) in enumerate(zip(actual,expected)):assert a==e,(i,requests[i],a,e)

    # Both wrong sources and wrong sequences are ignored; retry uses identical bytes.
    network=[]
    def server(body):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);sock.bind(("127.0.0.1",0));sock.settimeout(5)
        errors=[]
        def run():
            try:body(sock)
            except BaseException as e:errors.append(e)
            finally:sock.close()
        thread=threading.Thread(target=run);thread.start()
        return sock.getsockname(),thread,errors
    def discovery(sock):
        data,addr=sock.recvfrom(256);request=p.DiscoveryPacket.from_bytes(data)
        sock.sendto(p.DeviceAck(p.AckStatus.OK,request.sequence,3).to_bytes(),addr)
        for _ in range(2):sock.sendto(p.DeviceAck(p.AckStatus.READY,request.sequence,3).to_bytes(),addr)
    addr,thread,errors=server(discovery)
    found=native({"operation":"device-discover","target":f"{addr[0]}:{addr[1]}","timeout_seconds":.15});thread.join();assert not errors,errors
    assert len(found)==1 and found[0]["config_revision"]==3 and found[0]["host"]==addr[0] and found[0]["port"]==addr[1]
    network.append("discovery rejects non-READY ACKs and deduplicates responders")
    def configuration(sock):
        data,addr=sock.recvfrom(256);pkt=p.ConfigurationPacket.from_bytes(data)
        with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as other:
            other.sendto(p.DeviceAck(p.AckStatus.OK,pkt.sequence,8).to_bytes(),addr)
        sock.sendto(p.DeviceAck(p.AckStatus.OK,(pkt.sequence+1)%65536,8).to_bytes(),addr)
        again,addr=sock.recvfrom(256);assert again==data
        sock.sendto(p.DeviceAck(p.AckStatus.DUPLICATE,pkt.sequence,9).to_bytes(),addr)
    addr,thread,errors=server(configuration)
    result=native({"operation":"device-configure","target":f"{addr[0]}:{addr[1]}","timeout_seconds":.1,"retries":3,"packet":{"type":"config",**asdict(p.ActuationProfile().configuration(sequence=17))}});thread.join();assert not errors,errors
    assert result["attempts"]==2 and result["acknowledgement"]["config_revision"]==9
    network.append("configuration authenticates source/sequence and retries identical bytes")
    def sync(sock):
        for i in range(3):
            data,addr=sock.recvfrom(256);pkt=p.ClockSyncPacket.from_bytes(data)
            response=p.ClockSyncPacket(p.ClockSyncMessage.RESPONSE,pkt.sequence,pkt.host_send_time_us,pkt.host_send_time_us+2000,pkt.host_send_time_us+2020)
            sock.sendto(response.to_bytes(),addr)
    addr,thread,errors=server(sync)
    result=native({"operation":"device-sync","target":f"{addr[0]}:{addr[1]}","timeout_seconds":.2,"samples":3});thread.join();assert not errors,errors
    assert len(result["exchanges"])==3 and result["selected"]==min(result["exchanges"],key=lambda e:e["network_rtt_us"])
    network.append("clock exchange selects minimum network RTT without changing scheduling")

    # A serial pseudo-terminal exercises the real native serial driver, with no board.
    master,slave=os.openpty();slave_name=os.ttyname(slave)
    def banner():time.sleep(.5);os.write(master,b"booting\nWiFi connected, IP 192.168.1.55\n")
    thread=threading.Thread(target=banner);thread.start()
    serial=native({"operation":"serial-detect","port":slave_name,"timeout_seconds":2});thread.join();os.close(master);os.close(slave)
    assert serial==asdict(parse_endpoint_line("WiFi connected, IP 192.168.1.55"))
    # Real child process lifecycle, isolated executable in PATH; shell metacharacters
    # in project paths remain literal argv. No real upload is issued.
    project=args.output.resolve()/"firmware project; literal";project.mkdir();(project/"platformio.ini").write_text("[env:test]\n")
    bindir=args.output.resolve()/"bin";bindir.mkdir();pio=bindir/"pio"
    pio.write_text("#!/bin/sh\nprintf 'WiFi connected, IP 192.168.1.55\\n'\nsleep 30\n");pio.chmod(0o755)
    env={**os.environ,"PATH":str(bindir)+os.pathsep+os.environ["PATH"]}
    result=native({"operation":"firmware-flash","project":str(project),"environment":"test","timeout_seconds":3},env=env)
    assert result["return_code"]==0 and result["stopped_early"] and result["endpoint"]==serial
    pio.write_text("#!/bin/sh\nexit 7\n")
    result=native({"operation":"firmware-flash","project":str(project),"environment":"test","timeout_seconds":3},env=env)
    assert result["return_code"]==7 and not result["stopped_early"]
    pio.write_text("#!/bin/sh\nsleep 30\n")
    native({"operation":"firmware-flash","project":str(project),"timeout_seconds":.1},reject=True,env=env)
    # A child that exits while its descendant retains pipes must not leak that
    # descendant or leave the caller blocked in a reader thread.
    pio.write_text("#!/bin/sh\nsleep 30 &\nexit 0\n")
    started=time.monotonic()
    native({"operation":"firmware-flash","project":str(project),"timeout_seconds":.1},reject=True,env=env)
    assert time.monotonic()-started<5
    # Cancel a long network wait through the same SIGTERM used by native UI jobs.
    proc=subprocess.Popen([str(args.native_bin)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    proc.stdin.write(json.dumps({"operation":"device-discover","target":"127.0.0.1:59999","timeout_seconds":30}));proc.stdin.close();proc.stdin=None
    time.sleep(.1);started=time.monotonic();proc.terminate();proc.communicate(timeout=3)
    assert proc.returncode!=0 and time.monotonic()-started<1
    report={"passed":True,"packet_families":8,"packets":len(packets),"source_comparisons":len(requests),"network":network,"input":"original recorder/debounce/axis mapping and MIDI note semantics passed","serial":"pseudo-terminal driver/banner passed","firmware":"isolated subprocess success, failure, timeout, descendant cleanup and monitor shutdown passed","cancellation":"30-second UDP wait stopped through SIGTERM in under one second","physical_devices_accessed":False,"source_sha256":hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
