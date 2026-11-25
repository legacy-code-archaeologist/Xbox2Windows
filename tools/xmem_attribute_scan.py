"""Report the attribute word at each xmem call site in a PE image.

The names covered are the exports XMemAlloc, XMemFree and the Default aliases XMemAllocDefault
and XMemFreeDefault, which released images import by name as well. Each site records the name it
imported, so a site through XMemAllocDefault is never reported as XMemAlloc.

Usage: xmem_attribute_scan.py <image> [--output report.json] [--fail-on-null] [--fail-on-unresolved]

The image is read, never loaded or run.

Sites. A site is a call or tail jump into one of those xmem.dll exports, from a normal or
delay-load import:
- through the IAT slot;
- through an import thunk (a jmp through the slot that starts a block of code, as the linker
  emits);
- through a register loaded from the slot.

Code is decoded one function at a time using the exception directory (.pdata). Code outside
.pdata is decoded linearly between functions.

Sources. From each site the scanner walks backwards over the function's control flow to every
definition of RDX, the attribute word argument:
- Conditional and unconditional jumps inside the function are followed. A join with several
  definitions reports each one; a loop that does not change RDX adds nothing.
- A call between the definition and the site may change RDX (Microsoft x64 convention), so it
  ends that path as unknown.
- An immediate (including xor edx, edx) is a constant.
- A 32- or 64-bit load from RIP-relative image data is a constant when the data lies in a
  read-only section. In a writable section it is data, never a constant: a write through a
  pointer can change the word before the call, and writes through pointers are not tracked. The
  report keeps the initial value and the RIP-relative writes found (written_at, possibly empty)
  as evidence only.
- A register copy is followed. A 32-bit copy keeps only the low 32 bits.
- Reaching the function entry means the caller supplies the register being tracked. The entry
  register and whether a 32-bit copy truncated it on the way are recorded, and direct calls and
  jumps to the entry are resolved one level deep with that register. Exported functions may also
  have callers outside the image.
- Anything else (partial-register writes, arithmetic, loads through pointers, blocks reached only
  through jump tables, chained unwind fragments) is computed or unknown, never a constant.

Constant words are decoded with the public Xbox One ERA XALLOC_ATTRIBUTES layout used by
src/era_heap.h and classified with era_heap::allocate's desktop rules for a non-zero size.
--fail-on-null exits with 1 when any constant word would make era_heap return null.
--fail-on-unresolved exits with 1 when any site has a path that does not end in a constant.
"""
import argparse
import bisect
import json
import os
import re
import struct
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import pefile
from iced_x86 import (Decoder, FlowControl, InstructionInfoFactory, Mnemonic as M, OpAccess,
                      OpKind as K, Register as R, RegisterExt)

# Full-register names for the report and back, from iced-x86's Register constant module.
REGISTER_NAMES = {value: name for name, value in vars(R).items()
                  if isinstance(value, int) and not name.startswith("_")}
REGISTERS = {name: value for value, name in REGISTER_NAMES.items()}

FORMAT = 2
# Released images import the Default aliases by name, so a scan that only knows the first two
# names misses those call sites. The alias exports forward to the same two implementations, but
# they have their own IAT slots, thunks and import names.
TARGETS = ("XMemAlloc", "XMemAllocDefault", "XMemFree", "XMemFreeDefault")
GRANULARITY = 64 << 10  # x64 Windows allocation granularity
PAGE = 4 << 10
SCN_EXECUTE = 0x20000000
SCN_WRITE = 0x80000000
UNW_FLAG_CHAININFO = 0x4
WRITES = (OpAccess.WRITE, OpAccess.READ_WRITE, OpAccess.COND_WRITE, OpAccess.READ_COND_WRITE)
STOPS = (FlowControl.RETURN, FlowControl.UNCONDITIONAL_BRANCH, FlowControl.INDIRECT_BRANCH,
         FlowControl.INTERRUPT, FlowControl.EXCEPTION)
CALLS = (FlowControl.CALL, FlowControl.INDIRECT_CALL)
JUMPS = (FlowControl.CONDITIONAL_BRANCH, FlowControl.UNCONDITIONAL_BRANCH)
# Registers a callee may change under the Microsoft x64 calling convention.
VOLATILE = {R.RAX, R.RCX, R.RDX, R.R8, R.R9, R.R10, R.R11}
IMMEDIATES = (K.IMMEDIATE8, K.IMMEDIATE16, K.IMMEDIATE32, K.IMMEDIATE64, K.IMMEDIATE32TO64,
              K.IMMEDIATE8TO32, K.IMMEDIATE8TO64)
MAX_PATHS = 64
MASK64 = (1 << 64) - 1


def decode(word):
    """Fields of era_heap::decode."""
    return {
        "object_type": word & 0x3fff,
        "page_size": (word >> 14) & 0x3,
        "allocator_id": (word >> 16) & 0xff,
        "alignment": (word >> 24) & 0x1f,
        "memory_type": (word >> 29) & 0xf,
        "reserved": word >> 33,
    }


def desktop(word):
    """What era_heap::allocate does with the word for a non-zero size."""
    f = decode(word)
    if f["reserved"] or f["page_size"] == 3:
        return "null: reserved bits or undefined page size"
    alignment = 1 << max(4, f["alignment"])
    if f["memory_type"] == 0:
        if alignment > PAGE:
            return f"heap: page-backed, {alignment}-byte alignment"
        return f"heap: aligned allocation, {alignment}-byte alignment"
    if 1 <= f["memory_type"] <= 6:
        page_bytes = (4 << 10, 64 << 10, 4 << 20)[f["page_size"]]
        if max(page_bytes, alignment) > GRANULARITY:
            return "null: graphics page size or alignment above the allocation granularity"
        return "graphics: VirtualAlloc on an allocation-granularity boundary"
    if 7 <= f["memory_type"] <= 9:
        return "null: physical memory type not modeled"
    return "null: undefined memory type"


def full(register):
    return RegisterExt.full_register(register)


class Region:
    """Instructions of one function, chained fragment or stretch of code outside .pdata."""

    def __init__(self, lo, hi, kind, data, start):
        self.lo = lo
        self.kind = kind
        self.ins = list(Decoder(64, data[lo - start:hi - start], ip=lo))
        self.index = {ins.ip: n for n, ins in enumerate(self.ins)}
        self.jumps_to = {}
        for n, ins in enumerate(self.ins):
            if ins.flow_control in JUMPS and ins.op0_kind == K.NEAR_BRANCH64 and ins.near_branch_target in self.index:
                self.jumps_to.setdefault(ins.near_branch_target, []).append(n)


class Image:
    def __init__(self, path):
        self.pe = pefile.PE(str(path), fast_load=True)
        try:
            d = pefile.DIRECTORY_ENTRY
            self.pe.parse_data_directories(directories=[
                d["IMAGE_DIRECTORY_ENTRY_IMPORT"], d["IMAGE_DIRECTORY_ENTRY_EXPORT"],
                d["IMAGE_DIRECTORY_ENTRY_EXCEPTION"], d["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"]])
            if self.pe.FILE_HEADER.Machine != pefile.MACHINE_TYPE["IMAGE_FILE_MACHINE_AMD64"]:
                raise ValueError("Expected an x64 image")
            self.base = self.pe.OPTIONAL_HEADER.ImageBase
            self.info = InstructionInfoFactory()
            self.slots = {}  # IAT slot RVA -> (name, "import" or "delay")
            self.ordinals = []  # xmem.dll imports by ordinal, which have no name to match
            for desc in getattr(self.pe, "DIRECTORY_ENTRY_IMPORT", []):
                self._add_imports(desc, "import")
            for desc in getattr(self.pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []):
                self._add_imports(desc, "delay")
            self.sections = [(s.VirtualAddress, s.VirtualAddress + max(s.Misc_VirtualSize, s.SizeOfRawData),
                              s.Characteristics) for s in self.pe.sections]
            self.code = []
            for s in self.pe.sections:
                if s.Characteristics & SCN_EXECUTE:
                    data = s.get_data()
                    if s.Misc_VirtualSize:
                        data = data[:s.Misc_VirtualSize]
                    self.code.append((s.VirtualAddress, data))
            self.functions = []  # (begin, end, chained)
            for entry in getattr(self.pe, "DIRECTORY_ENTRY_EXCEPTION", []):
                unwind = self.pe.get_data(entry.struct.UnwindData, 1)
                chained = bool(unwind and (unwind[0] >> 3) & UNW_FLAG_CHAININFO)
                self.functions.append((entry.struct.BeginAddress, entry.struct.EndAddress, chained))
            self.functions.sort()
            self.begins = [f[0] for f in self.functions]
            self.exports = {sym.address for sym in
                            getattr(getattr(self.pe, "DIRECTORY_ENTRY_EXPORT", None), "symbols", []) if sym.address}
            # Function entries: exports, unchained .pdata starts, and (added while streaming) call targets.
            self.entries = set(self.exports) | {b for b, _, chained in self.functions if not chained}
            self.thunks = self._find_thunks()
            self.regions = {}
            self.decoded = 0
            self.passes = 0
        except Exception:
            self.pe.close()
            raise

    def _add_imports(self, desc, kind):
        name = desc.dll.decode(errors="replace").lower()
        if name.endswith("."):
            name = name[:-1]
        elif "." not in name:
            name += ".dll"
        if name != "xmem.dll":
            return
        for imp in desc.imports:
            if imp.name is None:
                self.ordinals.append(imp.ordinal)
                continue
            symbol = imp.name.decode(errors="replace")
            if symbol in TARGETS:
                self.slots[imp.address - self.base] = (symbol, kind)

    def section_of(self, rva):
        for lo, hi, characteristics in self.sections:
            if lo <= rva < hi:
                return lo, hi, characteristics
        return None

    def code_of(self, rva):
        for start, data in self.code:
            if start <= rva < start + len(data):
                return start, data
        return None

    def function_at(self, rva):
        i = bisect.bisect_right(self.begins, rva) - 1
        if i >= 0 and self.functions[i][0] <= rva < self.functions[i][1]:
            return self.functions[i]
        return None

    def _find_thunks(self):
        """jmp through an xmem slot at the start of a block: after padding, a return or another thunk."""
        thunks = {}
        for start, data in self.code:
            for match in re.finditer(rb"\xff\x25", data):
                off = match.start()
                if off + 6 > len(data):
                    continue
                target = start + off + 6 + struct.unpack_from("<i", data, off + 2)[0]
                if target not in self.slots:
                    continue
                if off and data[off - 1] == 0x48:  # REX.W form
                    off -= 1
                ip = start + off
                instruction = Decoder(64, data[off:off + 7], ip=ip).decode()
                if instruction.mnemonic != M.JMP or instruction.ip_rel_memory_address != target:
                    continue
                if self.function_at(ip):
                    continue
                before = data[off - 1] if off else None
                after_thunk = off >= 6 and data[off - 6:off - 4] == b"\xff\x25"
                if ip in self.entries or before in (None, 0xCC, 0x90, 0xC3) or after_thunk:
                    thunks[ip] = self.slots[target]
        return thunks

    def stream(self):
        """Every instruction, one .pdata function (or gap between functions) at a time."""
        self.passes += 1
        for start, data in self.code:
            end = start + len(data)
            pos = start
            i = bisect.bisect_left(self.begins, start)
            while i < len(self.functions) and self.functions[i][0] < end:
                lo, hi, _ = self.functions[i]
                hi = min(hi, end)
                if lo >= pos:
                    if lo > pos:
                        yield from self._decode(start, data, pos, lo)
                    yield from self._decode(start, data, lo, hi)
                    pos = hi
                i += 1
            if pos < end:
                yield from self._decode(start, data, pos, end)

    def _decode(self, start, data, lo, hi):
        for instruction in Decoder(64, data[lo - start:hi - start], ip=lo):
            self.decoded += 1
            yield instruction

    def region(self, rva):
        start, data = self.code_of(rva)
        end = start + len(data)
        function = self.function_at(rva)
        if function:
            lo, hi = function[0], min(function[1], end)
            kind = "fragment" if function[2] else "function"
        else:
            i = bisect.bisect_right(self.begins, rva)
            lo = max(start, self.functions[i - 1][1]) if i > 0 else start
            hi = min(end, self.functions[i][0]) if i < len(self.functions) else end
            kind = "gap"
        if lo not in self.regions:
            self.regions[lo] = Region(lo, hi, kind, data, start)
        return self.regions[lo]

    def writes(self, instruction, register):
        return any(used.access in WRITES and full(used.register) == register
                   for used in self.info.info(instruction).used_registers())


class Scan:
    def __init__(self, image):
        self.image = image
        self.data_queries = set()
        self.caller_queries = set()
        self.data_writers = {}
        self.callers = {}
        self.scanned_data = set()
        self.scanned_callers = set()

    # Walking backwards -----------------------------------------------------------------------

    def resolve(self, region, n, register=R.RDX, depth=0):
        """Source of `register` just before instruction n of `region`, merged over all paths."""
        results = self.walk(region, n, register, set(), [MAX_PATHS], depth)
        if not results:
            return {"source": "unknown", "detail": "no definition on any path"}
        return merge(results)

    def walk(self, region, n, register, seen, budget, depth):
        results = []
        k = n
        while True:
            key = (region.lo, k, register)
            if key in seen:
                return results  # already explored, or a loop that does not define the register
            seen.add(key)
            here = region.ins[k].ip
            for j in region.jumps_to.get(here, []):
                if budget[0] <= 0:
                    results.append({"source": "unknown", "detail": f"more than {MAX_PATHS} paths"})
                    return results
                budget[0] -= 1
                results.extend(self.walk(region, j, register, seen, budget, depth))
            if k == 0:
                results.append(self.block_start(region, 0, register, depth))
                return results
            previous = region.ins[k - 1]
            if previous.flow_control in STOPS:
                if not region.jumps_to.get(here):
                    results.append(self.block_start(region, k, register, depth))
                return results
            if self.image.writes(previous, register):
                results.append(self.describe(region, k - 1, register, seen, budget, depth))
                return results
            if previous.flow_control in CALLS and register in VOLATILE:
                results.append({"source": "unknown", "detail": "a call in between may change the register",
                                "at": hex(previous.ip)})
                return results
            k -= 1

    def block_start(self, region, k, register, depth):
        ip = region.ins[k].ip
        if region.kind == "fragment" and k == 0:
            return {"source": "unknown", "detail": "start of a chained unwind fragment", "at": hex(ip)}
        if ip in self.image.entries or (k == 0 and region.kind == "function"):
            # The caller supplies the register still being tracked, not always RDX: a copy such as
            # `mov rdx, r8` is followed to the entry, and the callers must be read from R8.
            result = {"source": "argument", "function": hex(ip), "register": REGISTER_NAMES[register]}
            if ip in self.image.exports:
                result["exported"] = True
            if depth == 0:
                self.caller_queries.add(ip)
                result["callers"] = None  # filled after the caller pass
            return result
        return {"source": "unknown", "detail": "block reached only through an indirect jump or from outside",
                "at": hex(ip)}

    def describe(self, region, k, register, seen, budget, depth):
        instruction = region.ins[k]
        at = hex(instruction.ip)
        op = instruction.mnemonic
        if op == M.XOR and instruction.op0_kind == K.REGISTER and instruction.op1_kind == K.REGISTER \
                and instruction.op0_register == instruction.op1_register \
                and RegisterExt.size(instruction.op0_register) in (4, 8):
            # An 8- or 16-bit xor leaves the upper bits in place, so it is not a constant.
            return {"source": "constant", "word": 0, "at": at}
        if op == M.MOV and instruction.op0_kind == K.REGISTER:
            width = RegisterExt.size(instruction.op0_register)
            if width in (4, 8) and instruction.op1_kind in IMMEDIATES:
                # A 32-bit destination zero-extends; a 64-bit one takes the sign-extended immediate.
                return {"source": "constant", "word": instruction.immediate(1) & MASK64, "at": at}
            if width in (4, 8) and instruction.op1_kind == K.MEMORY and instruction.is_ip_rel_memory_operand:
                return self.data_source(instruction.ip_rel_memory_address, width, at)
            if width in (4, 8) and instruction.op1_kind == K.REGISTER:
                followed = self.walk(region, k, full(instruction.op1_register), seen, budget, depth)
                if not followed:
                    followed = [{"source": "unknown", "detail": "no definition on any path"}]
                if width == 4:
                    followed = [truncate(result) for result in followed]
                return dict(merge(followed), via=at)
        return {"source": "computed", "at": at, "instruction": str(instruction)}

    def data_source(self, rva, width, at):
        raw = self.image.pe.get_data(rva, width)
        section = self.image.section_of(rva)
        result = {"data": hex(rva), "at": at}
        if len(raw) == width:
            result["word"] = int.from_bytes(raw, "little")
        else:
            result["word"] = 0
            result["zero_filled"] = True  # beyond the section's raw data
        if section is None:
            return dict(result, source="unknown", detail="address outside every section")
        if not section[2] & SCN_WRITE:
            return dict(result, source="constant", certainty="read-only section")
        self.data_queries.add(rva)
        return dict(result, source="data", writers=None)  # filled after the writer pass

    # Whole-image queries ---------------------------------------------------------------------

    def run_queries(self):
        data = self.data_queries - self.scanned_data
        entries = self.caller_queries - self.scanned_callers
        if not data and not entries:
            return False
        for instruction in self.image.stream():
            if instruction.is_ip_rel_memory_operand and instruction.ip_rel_memory_address in data:
                if any(mem.access in WRITES for mem in self.image.info.info(instruction).used_memory()):
                    self.data_writers.setdefault(instruction.ip_rel_memory_address, []).append(instruction.ip)
            if instruction.mnemonic in (M.CALL, M.JMP) and instruction.op0_kind == K.NEAR_BRANCH64 \
                    and instruction.near_branch_target in entries:
                self.callers.setdefault(instruction.near_branch_target, []).append(instruction.ip)
        self.scanned_data |= data
        self.scanned_callers |= entries
        return True

    def finish(self, result):
        """Fill in writers and callers for queries that have run."""
        source = result.get("source")
        if source == "data" and "writers" in result and int(result["data"], 16) in self.scanned_data:
            # Evidence only: with no RIP-relative writer the word can still be written through a
            # pointer before the call, so the path stays data (unresolved).
            result.pop("writers")
            result["written_at"] = [hex(w) for w in self.data_writers.get(int(result["data"], 16), [])]
            result["detail"] = "writable section: initial value only, writes through pointers are not tracked"
        if source == "argument" and result.get("callers", []) is None \
                and int(result["function"], 16) in self.scanned_callers:
            register = REGISTERS[result["register"]]
            truncated = result.get("truncated", False)
            callers = []
            for ip in self.callers.get(int(result["function"], 16), []):
                region = self.image.region(ip)
                if ip not in region.index:
                    callers.append({"source": "unknown", "detail": "caller not on an instruction boundary",
                                    "call": hex(ip)})
                    continue
                resolved = self.resolve(region, region.index[ip], register=register, depth=1)
                if truncated:
                    resolved = truncate(resolved)
                callers.append(dict(resolved, call=hex(ip)))
            result["callers"] = callers
        for nested in result.get("paths", []) + (result.get("callers") or []):
            self.finish(nested)
        return result


def truncate(result):
    """A 32-bit register copy keeps only the low 32 bits.

    Constants are masked, and an entry register is marked so that Scan.finish truncates each
    caller's resolved value the same way.
    """
    source = result.get("source")
    if source == "constant":
        return dict(result, word=result["word"] & 0xffffffff)
    if source == "multiple":
        return dict(result, paths=[truncate(p) for p in result["paths"]])
    if source == "argument":
        # Scan.finish applies this to each caller's resolved value.
        return dict(result, truncated=True)
    return result


def merge(results):
    """One result per distinct definition; equal constants from several paths count once."""
    distinct = []
    for result in results:
        for candidate in result["paths"] if result.get("source") == "multiple" else [result]:
            key = json.dumps({k: v for k, v in candidate.items() if k not in ("at", "via", "also_at")},
                             sort_keys=True)
            for existing_key, existing in distinct:
                if existing_key == key:
                    if candidate.get("at"):
                        existing.setdefault("also_at", []).append(candidate["at"])
                    break
            else:
                distinct.append((key, dict(candidate)))
    if len(distinct) == 1:
        return distinct[0][1]
    return {"source": "multiple", "paths": [d[1] for d in distinct]}


def annotate(result):
    if result.get("source") == "constant":
        result["word_hex"] = f"0x{result['word']:x}"
        result["fields"] = decode(result["word"])
        result["desktop"] = desktop(result["word"])
    elif "word" in result:
        result["word_hex"] = f"0x{result['word']:x}"
    for nested in result.get("paths", []) + (result.get("callers") or []):
        annotate(nested)
    return result


def constants(result):
    """Constant words a site can receive, and whether every path ends in a constant."""
    source = result.get("source")
    if source == "constant":
        return [result["word"]], True
    nested = result.get("paths") if source == "multiple" else result.get("callers") if source == "argument" else None
    if not nested:
        return [], False
    words, complete = [], not result.get("exported")
    for path in nested:
        w, c = constants(path)
        words += w
        complete = complete and c
    return words, complete


def scan(path):
    started = time.perf_counter()
    image = Image(path)
    try:
        work = Scan(image)
        found = []
        loads = []
        for instruction in image.stream() if image.slots else ():
            op = instruction.mnemonic
            if op == M.CALL and instruction.op0_kind == K.NEAR_BRANCH64:
                image.entries.add(instruction.near_branch_target)
            if op in (M.CALL, M.JMP):
                if instruction.is_ip_rel_memory_operand and instruction.ip_rel_memory_address in image.slots \
                        and instruction.ip not in image.thunks:
                    found.append((instruction.ip, image.slots[instruction.ip_rel_memory_address], "slot"))
                elif instruction.op0_kind == K.NEAR_BRANCH64 and instruction.near_branch_target in image.thunks:
                    found.append((instruction.ip, image.thunks[instruction.near_branch_target], "thunk"))
            elif op == M.MOV and instruction.op0_kind == K.REGISTER and instruction.is_ip_rel_memory_operand \
                    and instruction.ip_rel_memory_address in image.slots:
                loads.append((instruction.ip, image.slots[instruction.ip_rel_memory_address]))
        # Calls through a register loaded from a slot, before the register changes.
        for ip, target in loads:
            region = image.region(ip)
            n = region.index.get(ip)
            if n is None:
                continue
            register = full(region.ins[n].op0_register)
            for instruction in region.ins[n + 1:]:
                if instruction.mnemonic in (M.CALL, M.JMP) and instruction.op0_kind == K.REGISTER \
                        and full(instruction.op0_register) == register:
                    found.append((instruction.ip, target, "register"))
                    break
                if instruction.flow_control in STOPS or image.writes(instruction, register):
                    break
        sites = []
        for ip, (name, kind), route in sorted(found):
            region = image.region(ip)
            n = region.index.get(ip)
            attributes = work.resolve(region, n) if n is not None else {
                "source": "unknown", "detail": "site not on an instruction boundary of its region"}
            transfer = "jmp" if n is not None and region.ins[n].mnemonic == M.JMP else "call"
            sites.append({"site": hex(ip), "import": name, "import_kind": kind, "route": route, "transfer": transfer,
                          "region": f"{region.kind} {hex(region.lo)}", "attributes": attributes})
        # Writer and caller passes; resolving callers can add data queries of their own.
        while work.run_queries():
            for site in sites:
                work.finish(site["attributes"])
        words = {}
        by_source = Counter()
        unresolved = []
        for site in sites:
            annotate(site["attributes"])
            by_source[site["attributes"]["source"]] += 1
            site_words, complete = constants(site["attributes"])
            if not complete:
                unresolved.append(site["site"])
            for word in site_words:
                entry = words.setdefault(word, {"word_hex": f"0x{word:x}", "desktop": desktop(word),
                                                "fields": decode(word), "sites": []})
                if site["site"] not in entry["sites"]:
                    entry["sites"].append(site["site"])
        word_list = [words[w] for w in sorted(words)]
        return {
            "format": FORMAT,
            "image": Path(path).name,
            "imports": [{"import": n, "kind": k} for n, k in sorted(set(image.slots.values()))],
            "unmatched_xmem_ordinals": image.ordinals,
            "thunks": [{"thunk": hex(ip), "import": name, "exported": ip in image.exports}
                       for ip, (name, _) in sorted(image.thunks.items())],
            "sites": sites,
            "summary": {
                "sites": len(sites),
                "by_source": dict(sorted(by_source.items())),
                "words": word_list,
                "null_words": [w["word_hex"] for w in word_list if w["desktop"].startswith("null")],
                "sites_not_fully_constant": unresolved,
            },
            "stats": {"pdata_functions": len(image.functions), "instructions_decoded": image.decoded,
                      "passes": image.passes, "seconds": round(time.perf_counter() - started, 3)},
        }
    finally:
        image.pe.close()


def _same_path(left: Path, right: Path) -> bool:
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def _atomic_write_text(destination: Path, payload: str) -> None:
    """Stage JSON output in a sibling temp dir and atomically replace destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging_dir:
        staged = Path(staging_dir) / destination.name
        staged.write_text(payload, encoding="utf-8")
        os.replace(staged, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fail-on-null", action="store_true",
                        help="exit 1 when a constant word would make era_heap return null")
    parser.add_argument("--fail-on-unresolved", action="store_true",
                        help="exit 1 when a site has a path that does not end in a constant")
    args = parser.parse_args(argv)
    try:
        if args.output is not None:
            if args.output.exists() and args.output.is_dir():
                raise ValueError(f"Output path is an existing directory: {args.output}")
            if _same_path(args.image, args.output):
                raise ValueError(f"Output path collides with image: {args.output}")
        if args.image.is_dir():
            raise ValueError(f"Image path is a directory: {args.image}")
        report = scan(args.image)
        text = json.dumps(report, indent=2)
        if args.output is not None:
            _atomic_write_text(args.output, text + "\n")
        else:
            print(text)
    except (OSError, ValueError, pefile.PEFormatError) as exc:
        # Status 1 is reserved for the --fail-on-* gates.
        parser.exit(2, f"{type(exc).__name__}: {args.image}: {exc}\n")
    summary = report["summary"]
    print(f"{report['image']}: {summary['sites']} sites, {len(summary['words'])} distinct words, "
          f"{len(summary['null_words'])} null, {len(summary['sites_not_fully_constant'])} not fully constant, "
          f"{report['stats']['instructions_decoded']} instructions in {report['stats']['seconds']} s",
          file=sys.stderr)
    if args.fail_on_null and summary["null_words"]:
        return 1
    if args.fail_on_unresolved and summary["sites_not_fully_constant"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
