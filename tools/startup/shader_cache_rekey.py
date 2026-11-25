#!/usr/bin/env python3
"""Copy the shader requests of an older cache under the keys of the current cache revision.

usage: shader_cache_rekey.py <old cache dir> <new request dir>

The key of a graphics request (req_<key>.a), of an RTV variant (req_<key>.rtv) and of a device slot 67 compute
request (req_<key>.c67) hashes the translation revision string (pipeline_key and compute_key in gfx2hlsl.py).
After a revision change the runtime asks for new keys, so a kept cache is cold again. This tool recomputes the
key of every such request with the gfx2hlsl.py next to it and copies the request files under the new names into
<new request dir>. `gfx2hlsl.py --pending <new request dir>` can then translate them without a running title.

The old cache is only read. Only request files (req_*) are written; results and .fail files are not, and a file
that already exists in <new request dir> is left as it is. Each file is written to a temporary name and renamed;
the ready marker (.a, .c67) goes last, as the runtime writes it. Slot 57 and slot 86 mesh requests have their own
key strings and are only counted.

All requests are read before anything is written. When a request part is missing or unreadable, or an RTV
variant is malformed, does not match its own name or names a base request that is not in the cache, the tool
prints how many, writes nothing and exits with 1. Usage errors exit with 2. X2W_TRANSLATE_JOBS sets the
number of hashing processes (default: the processor count, at most 61 on Windows).
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfx2hlsl as g  # noqa: E402

REQUEST = re.compile(r'req_([0-9a-f]{16})\.(a|rtv|c67|m57|m86)')
# Request files of each kind; the ready marker is last.
PARTS = {'g': ('.b', '.root', '.ia', '.a'), 'c': ('.desc', '.sa', '.sb', '.data', '.root', '.c67')}
OPTIONAL = {'.ia'}  # a graphics request without an input layout has no .ia


class Incomplete(Exception):
    pass


def request_name(directory, key, ext=''):
    return os.path.join(directory, 'req_%016x%s' % (key, ext))


def read_parts(src, kind, key):
    parts = {}
    for ext in PARTS[kind]:
        path = request_name(src, key, ext)
        if ext in OPTIONAL and not os.path.isfile(path):
            continue
        with open(path, 'rb') as file:
            parts[ext] = file.read()
    return parts


def current_key(job):
    """(kind, old key, new key, None) or (kind, old key, None, reason)."""
    src, kind, key = job
    try:
        p = read_parts(src, kind, key)
    except OSError as error:
        return kind, key, None, 'unreadable %s: %s' % (os.path.basename(error.filename or ''), error.strerror)
    if kind == 'g':
        return kind, key, g.pipeline_key(p['.a'], p['.b'], p['.root'], p.get('.ia', b'')), None
    return kind, key, g.compute_key(p['.c67'], p['.data'], p['.root'], p['.desc'], p['.sa'], p['.sb']), None


def rtv_request(src, key, new_base):
    """The re-keyed text of an RTV variant "rtv1 <base key> <format>...", or raise Incomplete."""
    try:
        with open(request_name(src, key, '.rtv'), 'rb') as file:
            text = file.read().decode('ascii')
    except (OSError, UnicodeError) as error:
        raise Incomplete('unreadable .rtv: %s' % error) from error
    fields = text.split(' ')
    # Looser than the RTV pattern in gfx2hlsl.py (no limit on formats or digits); --pending refuses such a variant.
    if len(fields) < 3 or fields[0] != 'rtv1' or not all(f.isdigit() for f in fields[1:]):
        raise Incomplete('malformed .rtv')
    if g.fnv(text.encode('ascii')) != key:
        raise Incomplete('.rtv text does not hash to its name')
    base = new_base.get(int(fields[1]))
    if base is None:
        raise Incomplete('.rtv base request %016x is not a complete graphics request' % int(fields[1]))
    return ' '.join(['rtv1', str(base)] + fields[2:])


def publish(path, data, stats):
    if os.path.exists(path):
        stats['kept'] += 1
        return
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as file:
            file.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    stats['written'] += 1


def rekey(src, dst, jobs=None, out=sys.stdout):
    """Re-key the requests of src into dst; return the exit code."""
    if not os.path.isdir(src):
        print('error: %s is not a directory' % src, file=sys.stderr)
        return 2
    if os.path.exists(dst) and os.path.samefile(src, dst):
        print('error: the new request dir must not be the old cache', file=sys.stderr)
        return 2
    found = {'g': [], 'c': [], 'rtv': [], 'm57': 0, 'm86': 0}
    for name in sorted(os.listdir(src)):
        m = REQUEST.fullmatch(name)
        if not m:
            continue
        key, ext = int(m.group(1), 16), m.group(2)
        if ext in ('m57', 'm86'):
            found[ext] += 1
        else:
            found[{'a': 'g', 'c67': 'c', 'rtv': 'rtv'}[ext]].append(key)

    work = [(src, kind, key) for kind in ('g', 'c') for key in found[kind]]
    jobs = min(jobs or g.translation_jobs(), len(work))
    if jobs < 2:
        results = [current_key(job) for job in work]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(current_key, work, chunksize=max(1, len(work) // (jobs * 4))))

    new = {'g': {}, 'c': {}}
    bad = []
    for kind, old, key, reason in results:
        if reason:
            bad.append('req_%016x: %s' % (old, reason))
        else:
            new[kind][old] = key
    rtv = {}
    for old in found['rtv']:
        try:
            text = rtv_request(src, old, new['g'])
        except Incomplete as reason:
            bad.append('req_%016x.rtv: %s' % (old, reason))
            continue
        rtv[g.fnv(text.encode('ascii'))] = text
    if bad:
        print('error: %d of %d requests are incomplete or unreadable; nothing written' % (
            len(bad), len(work) + len(found['rtv'])), file=sys.stderr)
        for line in bad[:10]:
            print('  ' + line, file=sys.stderr)
        return 1

    os.makedirs(dst, exist_ok=True)
    stats = {'written': 0, 'kept': 0}
    for kind, label in (('g', 'graphics'), ('c', 'compute')):
        done = set()
        for old, key in sorted(new[kind].items()):
            if key in done:  # requests of the same pipeline from several revisions
                continue
            done.add(key)
            # Read again, not checked against the hashing pass: nothing may write the old cache meanwhile.
            for ext, data in read_parts(src, kind, old).items():
                publish(request_name(dst, key, ext), data, stats)
        current = sum(old == key for old, key in new[kind].items())
        print('%s: %d requests, %d already current -> %d keys' % (label, len(new[kind]), current, len(done)), file=out)
    for key, text in sorted(rtv.items()):
        publish(request_name(dst, key, '.rtv'), text.encode('ascii'), stats)
    print('rtv variants: %d requests -> %d keys' % (len(found['rtv']), len(rtv)), file=out)
    print('not re-keyed (own key strings): m57 %d, m86 %d' % (found['m57'], found['m86']), file=out)
    print('files: %d written, %d already present in %s' % (stats['written'], stats['kept'], dst), file=out)
    return 0


def main(argv):
    if len(argv) != 2 or argv[0].startswith('-'):
        print(__doc__.split('\n\n')[1], file=sys.stderr)
        return 2
    return rekey(argv[0], argv[1])


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
