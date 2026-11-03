import sys,os,glob,numpy as np,soundfile as sf
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from shale import core,eclq
K=os.urandom(16)
fs=sorted(glob.glob('data/CREMA-D/AudioWAV/*.wav'))[:20]
ok=True
for f in fs:
    x,sr=sf.read(f,dtype='int16')
    for F in (320,1600):
        top=core.frame_top(F,1,16)
        b=eclq.uniform_buckets(top,4)
        for a in ('ascon','aesgcm'):
            h,recs,idx=core.encrypt_stream(K,x,sr,16,F,b,aead=a)
            y=core.decrypt_stream(K,h,recs); ok&=np.array_equal(y[:,0],x)
            assert all(len(r)==b[j]+16 for r,j in zip(recs,idx))
    blob,j=core.encrypt_file(K,x,sr,16,4096,[0.4,0.6,0.8,1.0])
    ok&=np.array_equal(core.decrypt_file(K,blob,[0.4,0.6,0.8,1.0])[:,0],x)
print('roundtrip ok',ok)
# tamper tests
x,sr=sf.read(fs[0],dtype='int16')
h,recs,idx=core.encrypt_stream(K,x,sr,16,320,eclq.uniform_buckets(core.frame_top(320,1,16),4))
def bad(fn):
    try: fn(); return False
    except core.AuthError: return True
r2=list(recs); r2[3]=bytes([r2[3][0]^1])+r2[3][1:]
r3=list(recs); r3[2],r3[3]=r3[3],r3[2]
r4=list(recs[:-1])
h5=bytearray(h); h5[10]^=1
print('bitflip',bad(lambda:core.decrypt_stream(K,h,r2)),'swap',bad(lambda:core.decrypt_stream(K,h,r3)),'truncate',bad(lambda:core.decrypt_stream(K,h,r4)),'header',bad(lambda:core.decrypt_stream(K,bytes(h5),recs)),'wrongkey',bad(lambda:core.decrypt_stream(os.urandom(16),h,recs)))
blob,j=core.encrypt_file(K,x,sr,16,4096,[0.4,0.6,0.8,1.0]); b2=bytearray(blob); b2[200]^=4
print('file bitflip',bad(lambda:core.decrypt_file(K,bytes(b2),[0.4,0.6,0.8,1.0])))
# stereo 24 bit
a=np.random.default_rng(0).integers(-2**23,2**23,size=(9000,2))
h,recs,idx=core.encrypt_stream(K,a,48000,24,1000,eclq.uniform_buckets(core.frame_top(1000,2,24),3))
print('24bit stereo stream',np.array_equal(core.decrypt_stream(K,h,recs),a))
blob,j=core.encrypt_file(K,a,48000,24,1000,[0.5,1.0]); print('24bit stereo file',np.array_equal(core.decrypt_file(K,blob,[0.5,1.0]),a), j)
