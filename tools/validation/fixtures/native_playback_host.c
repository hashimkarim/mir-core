/* Private host-only PortAudio/evdev test double. Never opens a real audio or
 * input device. Signatures are compiled against public platform headers. */
#define _GNU_SOURCE
#include <portaudio.h>
#include <linux/input.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static void pause_seconds(double s) {if(s>0){struct timespec t={(time_t)s,(long)((s-(time_t)s)*1e9)};nanosleep(&t,0);}}
static int fixture_fd(int fd) {const char *p=getenv("MIR_TEST_EVDEV");struct stat a,b;return p && fstat(fd,&a)==0 && stat(p,&b)==0 && a.st_dev==b.st_dev && a.st_ino==b.st_ino;}
int ioctl(int fd,unsigned long request,...) {
    va_list ap;va_start(ap,request);void *p=va_arg(ap,void*);va_end(ap);
    if(fixture_fd(fd) && _IOC_TYPE(request)=='E') {
        size_t n=_IOC_SIZE(request);memset(p,0,n);
        unsigned nr=_IOC_NR(request);
        if(nr==0x06){const char *name="MIR isolated input fixture";size_t count=strlen(name)+1;if(count>n)count=n;memcpy(p,name,count);return (int)count;}
        if(nr==0x20 && n>0)((unsigned char*)p)[0]=1<<EV_KEY;
        if(nr==0x21 && n>KEY_SPACE/8)((unsigned char*)p)[KEY_SPACE/8]|=1<<(KEY_SPACE%8);
        if(nr==0x01 && n>=4)*(int*)p=0x10001;
        return 0;
    }
    int(*original)(int,unsigned long,...)=dlsym(RTLD_NEXT,"ioctl");return original(fd,request,p);
}
ssize_t read(int fd,void *out,size_t n) {
    if(fixture_fd(fd)) {
        static double start=0;static size_t index=0;const double taps[]={0.20,0.21,0.70,1.20};
        double t=now();if(!start)start=t;
        if(index<4 && t-start>=taps[index] && n>=sizeof(struct input_event)) {
            struct input_event event={0};event.type=EV_KEY;event.code=KEY_SPACE;event.value=1;memcpy(out,&event,sizeof(event));index++;return sizeof(event);
        }
        errno=EAGAIN;return -1;
    }
    ssize_t(*original)(int,void*,size_t)=dlsym(RTLD_NEXT,"read");return original(fd,out,n);
}
typedef struct {
    PaStreamCallback *callback;PaStreamFinishedCallback *finished;void *data;
    double rate;int channels;atomic_int active,stop;pthread_t thread;int joined;
} Fixture;
static const PaDeviceInfo info={1,"MIR isolated output",0,0,2,0,0.02,0,0.1,48000};
PaError Pa_Initialize(void){return paNoError;}
PaError Pa_Terminate(void){return paNoError;}
PaDeviceIndex Pa_GetDeviceCount(void){return 1;}
PaDeviceIndex Pa_GetDefaultOutputDevice(void){return 0;}
const PaDeviceInfo *Pa_GetDeviceInfo(PaDeviceIndex i){return i==0?&info:NULL;}
const char *Pa_GetErrorText(PaError e){(void)e;return "isolated output error";}
PaError Pa_IsFormatSupported(const PaStreamParameters *in,const PaStreamParameters *out,double rate) {
    if(in || !out || out->device!=0 || out->channelCount<1 || out->channelCount>2 || out->sampleFormat!=paFloat32)return paInvalidChannelCount;
    if(getenv("MIR_TEST_RESAMPLE") && rate!=48000)return paInvalidSampleRate;
    return paFormatIsSupported;
}
PaError Pa_OpenStream(PaStream **stream,const PaStreamParameters *in,const PaStreamParameters *out,double rate,unsigned long frames,PaStreamFlags flags,PaStreamCallback *cb,void *data) {
    if(Pa_IsFormatSupported(in,out,rate)!=0 || frames!=0 || flags!=0 || out->suggestedLatency!=0.1)return paInvalidFlag;
    Fixture *f=calloc(1,sizeof(*f));if(!f)return paInsufficientMemory;
    f->callback=cb;f->data=data;f->rate=rate;f->channels=out->channelCount;*stream=(PaStream*)f;return paNoError;
}
PaError Pa_SetStreamFinishedCallback(PaStream *stream,PaStreamFinishedCallback *cb){((Fixture*)stream)->finished=cb;return paNoError;}
static void *run(void *data) {
    Fixture *f=data;const char *path=getenv("MIR_TEST_PLAYED_PCM");FILE *out=path?fopen(path,"wb"):NULL;
    unsigned long frames=441;float samples[882];double due=now();int first=1;
    while(!atomic_load(&f->stop)) {
        double current=now();PaStreamCallbackTimeInfo time={0,current,current+0.02};
        if(getenv("MIR_TEST_BAD_CLOCK"))time.outputBufferDacTime=0.0/0.0;
        PaStreamCallbackFlags status=first && getenv("MIR_TEST_UNDERFLOW")?paOutputUnderflow:0;
        int result=f->callback(NULL,samples,frames,&time,status,f->data);first=0;
        if(result==paAbort)break;
        if(out){fwrite(samples,sizeof(float),frames*f->channels,out);fflush(out);}
        due+=frames/f->rate;pause_seconds(due-now());
        if(result==paComplete){pause_seconds(0.02);break;}
    }
    if(out)fclose(out);
    if(f->finished)f->finished(f->data);
    atomic_store(&f->active,0);return NULL;
}
PaError Pa_StartStream(PaStream *stream){Fixture *f=(Fixture*)stream;atomic_store(&f->active,1);return pthread_create(&f->thread,NULL,run,f)==0?paNoError:paInternalError;}
PaError Pa_IsStreamActive(PaStream *stream){return atomic_load(&((Fixture*)stream)->active);}
PaError Pa_AbortStream(PaStream *stream){Fixture *f=(Fixture*)stream;atomic_store(&f->stop,1);if(!f->joined){pthread_join(f->thread,NULL);f->joined=1;}return paNoError;}
PaError Pa_CloseStream(PaStream *stream){Pa_AbortStream(stream);free(stream);return paNoError;}
