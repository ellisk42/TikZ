from utilities import *
import signal
import os
import time
from multiprocessing import Process, Queue

def executeInProcessGroup(task):
    os.setpgid(0,0)
    task.q.put(task.command(*task.arguments))


class TimeshareTask():
    def __init__(self, command, arguments, logScore = 0, timeout = None):
        self.timeout = timeout
        self.logScore = logScore
        self.running = False
        self.finished = False
        self.command = command
        self.arguments = arguments
        self.q = Queue()
        self.process = None

    def execute(self,dt):
        if self.finished: return "finished"
        if not self.running:
            self.process = Process(target = executeInProcessGroup, args = (self,))
            self.process.start()
            print("timeshare child PID:",self.process.pid)
            os.setpgid(self.process.pid,self.process.pid)
            print("timeshare process group",os.getpgid(self.process.pid))
            assert os.getpgid(self.process.pid) == self.process.pid
            print("my process group",os.getpgrp(),"which should be",os.getpgid(0))
            assert os.getpgid(self.process.pid) != os.getpgid(0)
            self.running = True
        else:
            os.killpg(self.process.pid, signal.SIGCONT)
        
        self.process.join(dt)
        if self.process.is_alive():
            os.killpg(self.process.pid, signal.SIGSTOP)
            return "still running"
        else:
            self.finished = True
            return self.q.get()

    def cleanup(self):
        if self.process != None and self.process.is_alive():
            os.killpg(self.process.pid, signal.SIGKILL)


def executeTimeshareTasks(tasks, dt = 1, exponent = 1, minimumSlice = 0.05, globalTimeout = None):
    startTime = time.time()
    while len(tasks) > 0:
        sliceStartTime = time.time()
        
        if globalTimeout != None and time.time() - startTime > globalTimeout: break

        # Normalize the log scores and don't sign anything to anyone with less than minimum slice time
        bestScore = max([ t.logScore for t in tasks ])
        denominator = sum([ math.exp(t.logScore - bestScore) for t in tasks ])
        shares = [ dt*math.exp(t.logScore - bestScore)/denominator for t in tasks ]
        shares = [ int(s > minimumSlice) * s for s in shares ]
        z = sum(shares)
        if z < minimumSlice:
            # failure case: model predicts nothing should be allocated at least minimum slice
            # allocate equal time to everything which has maximal score
            shares = [ float(int(t.logScore == bestScore)) for t in tasks ]
            z = sum(shares)
        shares = [ dt*s/z for s in shares ]
        
        numberOfActiveSlices = len([ s for s in shares if s >= minimumSlice ])
        
        print("Time-sharing between %d tasks with weights: %s"%(len(tasks),shares))
        for share,task in zip(shares,tasks):
            if share < minimumSlice: continue
            # This can happen if the caller decides to explicitly mark something is finished
            if task.finished: continue
            
            # Setting the share to None will cause us to block until the process finishes
            # This is the same as saying that if we are spending everything on one process that just run it until the end
            if numberOfActiveSlices == 1: share = None
            print("Executing task:",task.arguments[0],"for",share,"sec")
            if share == None: print("Blocking until process finishes...")
            result = task.execute(share)
            if result == "still running": continue
            elif result == "finished": assert False
            else: yield result
        tasks = [ t for t in tasks if not t.finished ]

        sliceTotalTime = time.time() - sliceStartTime
        print("Finished giving all of the tasks a slice. Took %f sec, efficiency = %d"%(sliceTotalTime,int(100*dt/sliceTotalTime)))
        if exponent > 1:
            dt = dt*exponent
            print("Grew dt to",dt)

def executeTimeshareTasksFairly(tasks, dt = 1, minimumSlice = 0.05, globalTimeout = None):
    startTime = time.time()
    progress = np.zeros(len(tasks))
    totalRunTime = np.zeros(len(tasks))
    
    i = 0
    while True:
        sliceStartTime = time.time()

        i += 1
        
        if len(tasks) == 0: break
        if globalTimeout != None and time.time() - startTime > globalTimeout: break

        # weight vector
        W = np.exp(normalizeLogs(np.array([t.logScore for t in tasks ])))
        
        desiredSlices = dt * W * i - progress
        desiredSlices[desiredSlices < minimumSlice] = 0
        # failure case: nothing allocated at least minimum slice
        if not np.any(desiredSlices > 0):
            desiredSlices = (W >= W.max())*1.0
        shares = desiredSlices * (dt/np.sum(desiredSlices))

        progress = progress + shares
        totalRunTime = totalRunTime + shares

        # print "Time-sharing between %d tasks with weights: %s"%(len(tasks),shares)
        for share,task, in zip(shares,tasks):
            if share < minimumSlice: continue
            # This can happen if the caller decides to explicitly mark something is finished
            if task.finished: continue
            
            print("Executing task:",[str(a) for a in task.arguments],"for",share,"sec")
            result = task.execute(share)
            if result == "still running": continue
            elif result == "finished": assert False
            else: yield result

        for task,totalTime in zip(tasks,totalRunTime):
            if task.timeout != None and totalTime >= task.timeout:
                task.finished = True
                task.cleanup()
                print("(task %s timed out)"%(task.arguments))

        if any(t.finished for t in tasks):
            tasksAndRuntimes = [ (t,r) for t,r in zip(tasks,totalRunTime) if not t.finished ]
            tasks = [t for t,r in tasksAndRuntimes ]
            totalRunTime = np.array([r for t,r in tasksAndRuntimes ])
            # Reset sharing
            progress = np.zeros(len(tasks))
            i = 0

        sliceTotalTime = time.time() - sliceStartTime
        print("Finished giving all of the tasks a slice. Took %f sec, efficiency = %d%%"%(sliceTotalTime,int(100*dt/sliceTotalTime)))
        
    
if __name__ == "__main__":
    def callback(dummy):
        os.system("sleep 10")
        return "returning a value!"
    from math import log
    for result in executeTimeshareTasksFairly([TimeshareTask(callback,["high-priority"],logScore = log(0.9)),
                                               TimeshareTask(callback,["low-priority"],logScore = log(0.1),
                                                             timeout = 2)],
                                              dt = 1,
                                              minimumSlice = 0.15):
        print(result)
