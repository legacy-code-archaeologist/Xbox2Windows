"""Publish a synthetic Xbox envelope, then load the source-built desktop stages."""

import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'tools' / 'startup'))
import mesh86_cache


def source_envelope(stage):
    # The envelope exercises cache transport only. Desktop rendering uses
    # the separate shader sources below.
    image = b'FSSM' + bytes(192)
    compressed = zlib.compress(image)
    native = b'XBCH' + struct.pack('<IQQ', 3, len(compressed), len(image)) + compressed
    program = (struct.pack('<I', (stage << 16) | 0x65) + bytes(20) +
               b'XDAT' + struct.pack('<IIIII', 1, 36, 13, 48, len(native)) + native)
    part = b'DXIL' + struct.pack('<I', len(program)) + program
    return b'DXBC' + bytes(16) + struct.pack('<IIII', 1, 36 + len(part), 1, 36) + part


def trace_paired_submission(cmd, environment):
    # The dual fixture records the missing-writer control, the unsubmitted
    # argument writer, the ordered draw, then each remap and its recovery.
    for ordinal, changed_input in ((2, None), (3, None), (4, 'root-srv'), (6, 'arguments')):
        traced = dict(environment, X2W_MESH86_WRITER_TRACE=str(ordinal))
        result = subprocess.run(cmd, env=traced, stderr=subprocess.PIPE, encoding='utf-8', errors='replace')
        sys.stderr.write(result.stderr)
        if result.returncode:
            return result.returncode
        prefix = f'x2w-mesh86-trace call={ordinal} '
        rows = [line[len(prefix):] for line in result.stderr.splitlines() if line.startswith(prefix)]
        forwarded = [row for row in rows if row.startswith('stage=submit outcome=forwarded')]
        if ordinal == 2:
            assert any(row.startswith('input=root-srv range=fixed-span bytes=4 placement=copy lease=matched')
                       for row in rows), rows
            assert 'input=arguments range=fixed-span bytes=12 placement=copy lease=absent writer-submit=0' in rows, rows
            assert 'stage=record outcome=refused reason=mesh XMem read has no submitted writer for the required size' in rows, rows
            assert not forwarded, rows
            continue
        assert 'stage=record outcome=recorded' in rows, rows
        if changed_input:
            assert f'stage=submit outcome=refused reason={changed_input}-identity-changed' in rows, rows
            assert not forwarded, rows
            continue

        writers = set()
        for name, width, source, queue in (('root-srv', 4, 'upload', 'copy'),
                                           ('arguments', 12, 'default-uav', 'compute')):
            input_row = f'input={name}'
            publication = re.compile(
                rf'{input_row} range=fixed-span bytes={width} placement=copy lease=matched writer-submit=([1-9]\d*)')
            matches = [match for row in rows if (match := publication.fullmatch(row))]
            assert len(matches) == 1, rows
            writer = matches[0][1]
            writers.add(writer)
            assert f'{input_row} copy-slot=15 source={source} queue={queue} submit={writer} range=covers-placement' in rows, rows
            assert f'{input_row} order=wait writer-submit={writer} fence-submit={writer}' in rows, rows
        assert len(writers) == 2, rows
        assert len(forwarded) == 1, rows
        assert re.fullmatch(r'stage=submit outcome=forwarded submit=[1-9]\d* queue=direct signal=enqueued',
                            forwarded[0]), rows
    return 0


def check_trace_disabled(cmd, environment):
    # Both invalid values would select call 3 if the parser accepted a prefix or truncated to UINT.
    for selector in (None, '3x', '4294967299'):
        traced = dict(environment)
        if selector is None:
            traced.pop('X2W_MESH86_WRITER_TRACE', None)
        else:
            traced['X2W_MESH86_WRITER_TRACE'] = selector
        result = subprocess.run(cmd, env=traced, stderr=subprocess.PIPE, encoding='utf-8', errors='replace')
        sys.stderr.write(result.stderr)
        if result.returncode:
            return result.returncode
        trace = [line for line in result.stderr.splitlines() if line.startswith('x2w-mesh86-trace ')]
        assert not trace, f'trace selector {selector!r} emitted unexpected rows: {trace!r}'
        print(f'trace selector {selector!r}: paired readback passed without trace rows')
    return 0


def run(dxc, fixture, backend, work, depth_only, lease_xmem=None, mode=None, depth_pixel=None):
    for suffix, stage in (('ms', 13), ('ps', 0)):
        (work / f'source.{suffix}.dxbc').write_bytes(source_envelope(stage))
    environment = dict(os.environ, X2W_MESH86_FIXTURE_DIR=str(work), X2W_SHADER_CACHE=str(work))
    for flag in ('READY', 'DEPTH', 'DEPTH_PIXEL', 'PADDING', 'SAMPLE_MASK'):
        environment.pop(f'X2W_MESH86_FIXTURE_{flag}', None)
    if depth_only:
        environment['X2W_MESH86_FIXTURE_DEPTH'] = '1'
    if depth_pixel:
        environment['X2W_MESH86_FIXTURE_DEPTH_PIXEL'] = depth_pixel
    has_pixel = not depth_only or depth_pixel is not None
    if lease_xmem:
        environment['X2W_MESH86_FIXTURE_XMEM'] = str(lease_xmem)
        for flag in ('X2W_TEST_MESH_XMEM_SKIP_WAIT', 'X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT',
                     'X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT', 'X2W_MESH86_FIXTURE_DUAL',
                     'X2W_MESH86_FIXTURE_INIT_ONLY', 'X2W_MESH_INDIRECT_XMEM',
                     'X2W_MESH86_FIXTURE_LOOSE_ROOT'):
            environment.pop(flag, None)
        if mode == 'loose-root':
            environment['X2W_MESH86_FIXTURE_LOOSE_ROOT'] = '1'
        elif mode == 'skip-wait':
            environment['X2W_TEST_MESH_XMEM_SKIP_WAIT'] = '1'
        elif mode in ('dual', 'dual-skip-srv', 'dual-skip-argument', 'dual-init-only', 'dual-trace', 'dual-trace-disabled'):
            environment.update(X2W_MESH86_FIXTURE_DUAL='1', X2W_MESH_INDIRECT_XMEM='1')
            if mode == 'dual-skip-srv':
                environment['X2W_TEST_MESH_XMEM_SKIP_SRV_WAIT'] = '1'
            elif mode == 'dual-skip-argument':
                environment['X2W_TEST_MESH_XMEM_SKIP_ARGUMENT_WAIT'] = '1'
            elif mode == 'dual-init-only':
                environment['X2W_MESH86_FIXTURE_INIT_ONLY'] = '1'
    cmd = [fixture, backend, 'mesh']
    capture = subprocess.run(cmd, env=environment)
    if capture.returncode:
        return capture.returncode
    marker, = work.glob('req_*.m86')
    key = int(marker.stem.removeprefix('req_'), 16)
    source_ms, source_ps, root, stream = mesh86_cache.load_request(work, key)
    assert source_ms == source_envelope(13) and source_ps == (source_envelope(0) if has_pixel else b'')
    assert root.startswith(b'DXBC') and stream[:16] == bytes(16)
    ms, ps = (work / f'desktop.{suffix}.cso' for suffix in ('ms', 'ps'))
    mesh_source = HERE / ('xmem_mesh_root_srv.hlsl' if lease_xmem else 'startup_indirect_mesh.hlsl')
    defines = ['-D', 'X2W_MESH_DUAL=1'] if environment.get('X2W_MESH86_FIXTURE_DUAL') else []
    subprocess.run([dxc, '-T', 'ms_6_5', '-E', 'mesh', *defines, '-Fo', ms, mesh_source], check=True)
    if lease_xmem:
        subprocess.run([dxc, '-T', 'ps_6_0', '-E', 'pixel', '-D', 'X2W_MESH_LEASE_COLOR=1',
                        '-Fo', ps, mesh_source], check=True)
    elif has_pixel:
        entry = 'depth_only' if depth_pixel else 'main'
        subprocess.run([dxc, '-T', 'ps_6_0', '-E', entry, '-Fo', ps,
                        HERE / 'startup_indirect_mesh_cache.hlsl'], check=True)
    mesh86_cache.publish(work, key, ms, ps if has_pixel else None, srv_dxc=dxc if lease_xmem else None)
    assert (work / f'{key:016x}.ps.cso').is_file() == has_pixel
    ready = work / f'{key:016x}.mesh86.txt'
    correct = ready.read_bytes()
    ready.write_bytes(correct.replace(b'mesh86 2' if lease_xmem else b'mesh86 1', b'mesh86 0', 1))
    stale = subprocess.run(cmd, env=dict(environment, X2W_MESH86_FIXTURE_REJECT_MARKER='1') if lease_xmem else environment)
    if stale.returncode:
        return stale.returncode
    ready.write_bytes(correct)
    # WARP keeps the existing MS-only depth refusal.
    if has_pixel or backend == 'hardware':
        environment['X2W_MESH86_FIXTURE_READY'] = '1'
    if mode == 'dual-trace':
        return trace_paired_submission(cmd, environment)
    if mode == 'dual-trace-disabled':
        return check_trace_disabled(cmd, environment)
    if lease_xmem or depth_only:
        return subprocess.run(cmd, env=environment).returncode

    # Change only the blend and depth-stencil object padding between capture
    # and replay. A cache miss leaves the fixture's color attachments clear.
    padded = dict(environment, X2W_MESH86_FIXTURE_PADDING='1')
    replay = subprocess.run(cmd, env=padded)
    if replay.returncode:
        return replay.returncode
    assert list(work.glob('req_*.m86')) == [marker]

    # A real sample-mask change must instead publish a different request and
    # refuse the previously cached PSO, leaving all attachments clear.
    padded.pop('X2W_MESH86_FIXTURE_READY')
    changed = subprocess.run(cmd, env=dict(padded, X2W_MESH86_FIXTURE_SAMPLE_MASK='1'))
    if changed.returncode:
        return changed.returncode
    changed_marker, = set(work.glob('req_*.m86')) - {marker}
    changed_key = int(changed_marker.stem.removeprefix('req_'), 16)
    changed_ms, changed_ps, changed_root, changed_stream = mesh86_cache.load_request(work, changed_key)
    assert (changed_ms, changed_ps, changed_root) == (source_ms, source_ps, root)
    assert stream[-8:] == struct.pack('<II', 9, 0xffffffff)
    assert changed_stream == stream[:-4] + struct.pack('<I', 0xfffffffe)
    print('mesh cache padding replay reused the PSO; sample-mask change generated a new key')
    return 0


if __name__ == '__main__':
    dxc, fixture, backend, *lease = sys.argv[1:]
    cases = ((False, None),) if lease else (
        (False, None), (True, None), (True, 'write'), (True, 'no-dsv'), (True, 'read-only'),
    )
    for depth_only, depth_pixel in cases:
        with tempfile.TemporaryDirectory(prefix='mesh86-', dir=Path.cwd()) as directory:
            result = run(dxc, fixture, backend, Path(directory), depth_only, Path(lease[0]) if lease else None,
                         lease[1] if len(lease) > 1 else None, depth_pixel)
            if result:
                sys.exit(result)
