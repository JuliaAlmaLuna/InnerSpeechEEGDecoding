import multiprocessing
from multiprocessing import Value
import time


def printProcess(processName, printText):
    with open(f"processOutputs/processId{processName}Output.txt", "a") as f:
        print(printText, file=f)


def hiSleep(x, start):
    # global start

    time.sleep(0.2)
    # print(q.get())
    print(start.value)
    printProcess(f"hi{x}", "hi")
    # print("hi", flush=True, file="juliaOutput.txt")
    time.sleep(x)
    print(start.value)
    printProcess(f"hi{x}", "goodBye")
    # print("goodbye", flush=True, file="juliaOutput.txt")


def test():
    # print(multiprocessing.active_children())
    # for child in multiprocessing.active_children():
    #    child.terminate()
    # q = multiprocessing.Queue()
    # q.put(True)

    start = Value("i", 0)
    #start = 0

    processList = []

    for x in range(10):
        p = multiprocessing.Process(target=hiSleep, args=(x, start))
        processList.append(p)

    for process in processList:
        print(process, process.is_alive())
        print(process)
        process.start()

    print(multiprocessing.active_children())
    while True:
        time.sleep(1)
        start.value = start.value + 5
        # if len(multiprocessing.active_children()) > 9:

        print(len(multiprocessing.active_children()))

        if len(multiprocessing.active_children()) < 1:
            break
    # p.set_start_method("")

    # print(p, p.is_alive())
    for process in processList:
        process.join()


if __name__ == '__main__':
    test()
