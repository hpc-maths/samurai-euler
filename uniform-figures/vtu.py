import re, struct, numpy as np
def read_vtu(f):
    b=open(f,'rb').read(); i=b.index(b'<AppendedData'); head=b[:i].decode(); data=b[b.index(b'_',i)+1:]
    arrs={}; k=0
    for m in re.finditer(r'<DataArray type="(\w+)"(?: Name="(\w+)")?(?: NumberOfComponents="(\d)")?[^>]*offset="(\d+)"',head):
        t,name,nc,off=m.groups(); off=int(off)
        n=struct.unpack('<Q',data[off:off+8])[0]
        dt={'Float32':np.float32,'Int32':np.int32,'UInt8':np.uint8}[t]
        a=np.frombuffer(data[off+8:off+8+n],dtype=dt)
        if name is None: name='points'; a=a.reshape(-1,3)
        arrs[name]=a
    ends=arrs['offsets']; starts=np.concatenate([[0],ends[:-1]])
    arrs['polys']=[arrs['points'][s:e,:2] for s,e in zip(starts,ends)]
    return arrs
