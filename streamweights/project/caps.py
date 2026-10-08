"""Backend capability checks, done before any run state is published.

A backend must provide the atomic operations the ownership protocol is built on:

  local disk  an exclusive OS lock that excludes a second opener, and an atomic replace.
              Only documented local filesystems are accepted. Known network filesystems
              (NFS, SMB/CIFS, AFP, sshfs, 9p, cephfs, gluster, lustre, WebDAV, ...) and
              unknown types are refused: a probe on one machine proves nothing about
              flock across machines, and the Linux manual says as much for NFS.
  S3          PutObject with If-None-Match: * (create only) and If-Match <etag> (replace only
              if unchanged), each rejecting a violated condition with HTTP 412.
              Checked on disposable probe objects that are deleted afterwards.

Nothing here writes run state. A refused backend raises SpillError."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path

from ..errors import SpillError

LOCAL_OK = {"apfs", "hfs", "hfs+", "ext2", "ext3", "ext4", "xfs", "btrfs", "tmpfs", "zfs",
            "f2fs", "jfs", "overlay", "overlayfs", "ramfs", "reiserfs", "ufs"}
NETWORK = {"nfs", "nfs4", "nfs3", "cifs", "smb", "smb2", "smb3", "smbfs", "afpfs", "afp", "webdav",
           "9p", "ceph", "cephfs", "glusterfs", "lustre", "gpfs", "beegfs", "sshfs",
           "fuse.sshfs", "fuse.glusterfs", "ncpfs", "autofs", "osxfuse", "macfuse", "fuseblk",
           "fuse"}
_CACHE: dict = {}


def fs_type(path: str | Path) -> str:
    """The filesystem type of the mount holding `path`, lowercase; 'unknown' if not found."""
    p = os.path.realpath(str(path))
    try:
        if sys.platform.startswith("linux"):
            best, typ = "", "unknown"
            for line in Path("/proc/mounts").read_text().splitlines():
                parts = line.split()
                if len(parts) < 3:
                    continue
                mnt = parts[1].replace("\\040", " ")
                if (p == mnt or p.startswith(mnt.rstrip("/") + "/") or mnt == "/") \
                        and len(mnt) >= len(best):
                    best, typ = mnt, parts[2]
            return typ.lower()
        if sys.platform == "darwin":
            out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
            best, typ = "", "unknown"
            for line in out.splitlines():
                if " on " not in line or "(" not in line:
                    continue
                mnt = line.split(" on ", 1)[1].rsplit(" (", 1)[0]
                kind = line.rsplit("(", 1)[1].split(",")[0].strip(") ")
                if (p == mnt or p.startswith(mnt.rstrip("/") + "/") or mnt == "/") \
                        and len(mnt) >= len(best):
                    best, typ = mnt, kind
            return typ.lower()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def check_local(path: str | Path, fstype=fs_type) -> dict:
    """Refuse a network or unverified filesystem; then probe flock exclusion and atomic
    replace in a disposable file next to the run state."""
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    typ = fstype(d)
    if typ in NETWORK or typ.startswith("fuse"):
        raise SpillError(f"{d} is on a network or FUSE filesystem ({typ}); the run lock needs "
                         f"local flock semantics that cannot be verified there",
                         "keep the project on a local disk, or move it with `spill move "
                         "<project> s3://bucket/prefix`")
    if typ not in LOCAL_OK:
        raise SpillError(f"{d} is on a filesystem this version has not verified ({typ}); "
                         f"verified: {', '.join(sorted(LOCAL_OK))}",
                         "keep the project on a verified local disk")
    import fcntl
    probe = d / f".probe-{uuid.uuid4().hex[:8]}"
    try:
        probe.write_text("probe")
        fd1 = os.open(probe, os.O_RDWR)
        fd2 = os.open(probe, os.O_RDWR)
        try:
            fcntl.flock(fd1, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                fcntl.flock(fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
                raise SpillError(f"{d}: an exclusive lock did not exclude a second opener",
                                 "use a different local filesystem")
            except BlockingIOError:
                pass
        finally:
            os.close(fd1)
            os.close(fd2)
        tmp = d / f".probe-{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text("2")
        os.replace(tmp, probe)
        if probe.read_text() != "2":
            raise SpillError(f"{d}: an atomic replace did not replace", "use a different disk")
    finally:
        probe.unlink(missing_ok=True)
    return {"backend": "local", "fstype": typ, "platform": platform.system()}


def check_s3(backend) -> dict:
    """Probe conditional writes on disposable keys next to the run's control object."""
    from botocore.exceptions import ClientError
    c = backend.client
    base = backend.prefix + "/" if backend.prefix else ""
    key = f"{base}_probe/{uuid.uuid4().hex}"
    made = []

    def put(body=b"x", **kw):
        return c.put_object(Bucket=backend.bucket, Key=key, Body=body, **kw)

    def refused(fn):
        try:
            fn()
        except ClientError as e:
            return e.response.get("ResponseMetadata", {}).get("HTTPStatusCode") in (409, 412)
        return False

    try:
        r = put(IfNoneMatch="*")
        made.append(key)
        etag = r["ETag"]
        if not refused(lambda: put(IfNoneMatch="*")):
            raise SpillError(f"{backend.uri}: this store accepted a second create with "
                             f"If-None-Match: *; it has no conditional create",
                             "use a store with S3 conditional writes (AWS S3, MinIO)")
        if not refused(lambda: put(IfMatch='"0000000000000000"')):
            raise SpillError(f"{backend.uri}: this store accepted a replace with a wrong "
                             f"If-Match ETag; it has no compare-and-swap",
                             "use a store with S3 conditional writes (AWS S3, MinIO)")
        put(b"y", IfMatch=etag)                    # a different body: the ETag changes
        if not refused(lambda: put(b"z", IfMatch=etag)):
            raise SpillError(f"{backend.uri}: this store accepted a replace with a stale ETag",
                             "use a store with S3 conditional writes (AWS S3, MinIO)")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        raise SpillError(f"{backend.uri}: the conditional-write probe failed ({code})",
                         "check the endpoint, credentials and bucket (AWS_ENDPOINT_URL, "
                         "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)") from e
    except TypeError as e:
        raise SpillError(f"this boto3 cannot send conditional writes ({e})",
                         'pip install -U "streamweights[cloud]"') from e
    finally:
        for k in made:
            try:
                c.delete_object(Bucket=backend.bucket, Key=k)
            except Exception:
                pass
    return {"backend": "s3", "endpoint": os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_ENDPOINT_URL") or "aws"}


def open_backend(state_uri: str, probe: bool = True):
    """The control backend for a run's state location, after the capability probe passes
    (once per process per location)."""
    from ..portable.store import is_uri, scheme_of
    from .control import LocalBackend, S3Backend
    if is_uri(state_uri) and scheme_of(state_uri) in ("s3", "s3a"):
        b = S3Backend(state_uri.replace("s3a://", "s3://"))
        if probe and state_uri not in _CACHE:
            _CACHE[state_uri] = check_s3(b)
        return b
    if is_uri(state_uri) and scheme_of(state_uri) != "file":
        raise SpillError(f"{scheme_of(state_uri)}:// cannot provide the atomic operations a "
                         f"live run needs (only local disk and s3:// are supported)",
                         "use a local path or s3://bucket/prefix")
    path = state_uri[len("file://"):] if state_uri.startswith("file://") else state_uri
    if probe and path not in _CACHE:
        _CACHE[path] = check_local(path)
    return LocalBackend(path)
