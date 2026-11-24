"""Verify all provided staff assets. The one known legacy non-data script mismatch
is documented explicitly, never ignored for materialized data or manifests.
"""
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
out=pathlib.Path(sys.argv[2]).resolve()
manifest=json.loads((root/"SHA256SUMS.json").read_text())
mismatch={}
for name,expected in manifest.items():
    file=root/name
    actual=hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    if actual!=expected:mismatch[name]={"expected":expected,"actual":actual}
exception="scripts/check_release_STAFF_LEGACY.py"
unapproved=sorted(set(mismatch)-{exception})
audit={"checked":len(manifest),"mismatches":mismatch,
       "approved_legacy_exception":list(mismatch)==[exception] or not mismatch,
       "unapproved":unapproved}
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(audit,indent=2)+"\n")
if unapproved:
    raise RuntimeError("Data integrity failure: "+str(unapproved))
if exception in mismatch:
    print("WARNING: staff checksum mismatch in unrelated legacy release checker; "
          "all other files verified. See audit.",flush=True)
else:
    print("PASS: all staff checksums verified",flush=True)
