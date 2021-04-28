# import socket programming library
import socket

# import thread module
from _thread import *
import threading
import time


class AsyncServer:
    def __init__(self, host, port, timeout, batchsize):
        self.timeout = timeout
        self.batchsize = batchsize

        self.pending_req_counter = 0
        self.first_pending_req_arrival = 0.0
        self.pending_time = 0.0

        self.task_lock = threading.Lock()
        self.task_done_cv = threading.Condition()
        self.pending_tasks = dict()
        self.finished_tasks = dict()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind((host, port))
        self.sock.listen(5)
        print("async server is on")

    def update_pending_time(self):
        if self.pending_req_counter == 0:
            self.pending_req_counter += 1
            self.first_pending_req_arrival = time.time()
            self.pending_time = 0.0
        elif self.pending_req_counter >= 1:
            self.pending_req_counter += 1
            self.pending_time = time.time() - self.first_pending_req_arrival

    def do_batch(self):
        while True:
            thereiswaiting = 0
            self.task_lock.acquire()
            for k, v in self.pending_tasks.items():
                thereiswaiting += 1
                self.finished_tasks[k] = v[::-1]
            self.pending_tasks.clear()
            self.task_lock.release()
            if thereiswaiting >= 1:
                self.task_done_cv.acquire()
                self.task_done_cv.notify_all()
                self.task_done_cv.release()

    def on_new_request(self, id, connection):
        # data received from client
        data = connection.recv(1024)
        self.task_lock.acquire()
        self.pending_tasks[id] = data
        # # reverse the given string from client
        # data = data[::-1]
        self.task_lock.release()
        self.task_done_cv.acquire()
        self.task_done_cv.wait()
        self.task_done_cv.release()
        self.task_lock.acquire()
        data = self.finished_tasks[id]
        del self.finished_tasks[id]
        self.task_lock.release()
        # send back reversed string to client
        connection.send(data)
        # connection closed
        connection.close()
        return

    def run(self):
        start_new_thread(self.do_batch, ())
        # a forever loop until client wants to exit
        id = 1
        while True:
            # establish connection with client
            c, addr = self.sock.accept()
            print('connection from :', addr[0], ':', addr[1])
            self.update_pending_time()
            # Start a new thread and return its identifier
            start_new_thread(self.on_new_request, (id, c,))
            id += 1
        self.sock.close()


if __name__ == '__main__':
    server = AsyncServer("localhost", 12345, 0, 0)
    server.run()