"""Read a bounded public archive member through verified HTTP byte ranges."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import urllib.request


class BoundedHTTPFile(io.RawIOBase):
    def __init__(self,url,offset,size,budget=64*1024**2,block_size=1024**2,cache_dir=None):
        super().__init__()
        if offset<0 or size<=0 or budget<=0 or block_size<=0:raise ValueError('Invalid byte limits')
        self.url,self.offset,self.size=url,int(offset),int(size)
        self.budget,self.block_size=int(budget),int(block_size)
        self.position=0;self.fetched_bytes=0;self.requests=0;self.blocks=OrderedDict()
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
            identity={'url':url,'offset':self.offset,'size':self.size,'block_size':self.block_size}
            path=self.cache_dir/'identity.json'
            if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Range cache identity changed')
            path.write_text(json.dumps(identity,indent=2)+'\n')

    def readable(self):return True
    def seekable(self):return True
    def writable(self):return False
    def tell(self):return self.position

    def seek(self,offset,whence=io.SEEK_SET):
        position=offset if whence==io.SEEK_SET else self.position+offset if whence==io.SEEK_CUR else self.size+offset if whence==io.SEEK_END else None
        if position is None or position<0:raise ValueError('Invalid seek')
        self.position=int(position);return self.position

    def _block(self,index):
        if index in self.blocks:
            self.blocks.move_to_end(index);return self.blocks[index]
        start=index*self.block_size;end=min(start+self.block_size,self.size)
        path=self.cache_dir/f'block_{index:07d}.bin' if self.cache_dir else None
        if path and path.exists():
            body=path.read_bytes()
            if len(body)!=end-start:raise ValueError('Incomplete cached range')
        else:
            if self.fetched_bytes+end-start>self.budget:raise OSError('Remote member byte budget exceeded')
            left,right=self.offset+start,self.offset+end-1
            request=urllib.request.Request(self.url,headers={'Range':f'bytes={left}-{right}'})
            with urllib.request.urlopen(request,timeout=60) as response:
                if response.status!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {left}-{right}/'):
                    raise OSError('Server did not honor exact member byte range')
                body=response.read(end-start+1)
            if len(body)!=end-start:raise OSError('Incorrect range payload length')
            self.fetched_bytes+=len(body);self.requests+=1
            if path:path.write_bytes(body)
        self.blocks[index]=body
        if len(self.blocks)>8:self.blocks.popitem(last=False)
        return body

    def read(self,size=-1):
        if self.closed:raise ValueError('Closed byte-range stream')
        end=self.size if size is None or size<0 else min(self.size,self.position+size)
        if self.position>=end:return b''
        pieces=[]
        while self.position<end:
            index=self.position//self.block_size
            block=self._block(index);inside=self.position-index*self.block_size
            count=min(end-self.position,len(block)-inside)
            pieces.append(block[inside:inside+count]);self.position+=count
        return b''.join(pieces)

    def readinto(self,target):
        view=memoryview(target).cast('B');body=self.read(len(view));view[:len(body)]=body;return len(body)
