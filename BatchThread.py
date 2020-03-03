
import numpy as np
# A simple generator wrapper, not sure if it's good for anything at all.
# With basic python threading
import threading
# from threading import Thread
import multiprocessing as mp
import time
try:
    from queue import Queue

except ImportError:
    from Queue import Queue

# ... or use multiprocessing versions
# WARNING: use sentinel based on value, not identity
from multiprocessing import Process, Queue as MpQueue
from utils import *

class ThreadedGenerator(object):
    """
    Generator that runs on a separate thread, returning values to calling
    thread. Care must be taken that the iterator does not mutate any shared
    variables referenced in the calling thread.
    """
   
    def __init__(self, iterator_train,
                 # iterator_label,
                 batch_size,
                 sentinel=object(),
                 queue_maxsize=128,
                 daemon=False,
                 Thread=threading.Thread,
                 Queue=Queue):
        self.iterator_train = iterator_train
        # self.iterator_label = iterator_label
        self.batch_size = batch_size
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        #self._thread = mp.Process(target=self._run)
        self._thread = Thread(
            name=repr(iterator_train),
            target=self._run
        )
        self.__iter_time = 0
        
        #self._thread.daemon = daemon

    def __repr__(self):
        return 'ThreadedGenerator({!r})'.format(self.iterator_train)

    def _run(self):
        try:
            # for value in self._iterator:
            #     self._queue.put(value)
            batch_gen = self._gen_batches()
            # loop over generator and put each batch into the queue
            #print("__batch_gen_shape__%s"%batch_gen.shape)\
            print("__batch_gen")
            print(batch_gen)
            for data in batch_gen:
                
                self._queue.put(data, block=True)
                #print("__queue__size__:%d "%(self._queue.qsize()))
            # once the generator gets through all data issue the terminating command and close it
            self._queue.put(None)
        finally:
            pass
            # self._queue.put(self._sentinel)


    def _gen_batches(self):
        num_samples = len(self.iterator_train)
        idx = np.random.permutation(num_samples)
        batches = range(0, num_samples - self.batch_size + 1, self.batch_size)
        print("_gen")
        i=0
        e_i =0
        for batch in batches:
            tmp_ = []
            
            X_batch = self.iterator_train[batch:batch + self.batch_size]
            for data in X_batch:
                try:
                    img = process_sub_image(data,random_crop=True)
                    #checkimage(img)
                    tmp_.append(img)
                    #print(tmp_)
                except:
                    e_i+=1
                    print("__except__time:%d data:%s"%(e_i,data))
                    continue
                #print(img.shape)
            # y_batch = self.iterator_label[batch:batch + self.batch_size]
            i+=1
            print("__iter__data__batchs: %s"%i)
            self.__iter_time += 1
            # do some stuff to the batches like augment images or load from folders
            
            print("__except__time:%d"%(e_i))
            tmp_ = np.asarray(tmp_)
            print(tmp_.shape)
            yield [tmp_]
    
    def get_iter_time(self):
        return self.__iter_time
    
    def __iter__(self):
        c = 0
        print("thread start...")
        self._thread.setDaemon(True)
        self._thread.start()
        # load the batch generator as a python generator
        print(threading.get_ident())
        print(self._thread.ident)

        print("_iter_")
        # self._queue.close()
        for value in iter(self._queue.get, None):
            
            c += 1
            print("__DEBUG__iter__%s" % c)
            print("pading.......")
            yield value[0]

        self._thread.join()

