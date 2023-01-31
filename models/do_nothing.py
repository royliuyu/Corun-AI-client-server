def work(config, pipe, queue):
    queue.put(dict(process= 'yolo_v5s'))  # for queue exception of this process
    con_a,con_b = pipe
    con_a.close()
    print('Infer: do nothing')

    while True:
        pass
        if con_b.poll():
            msg = con_b.recv()  # receive msg from train
            if msg == 'stop':  # signal from training process to end, ROy added on 12092022
                print('Do noting: get "stop" notice from main. ')
                con_b.close()
                queue.put(dict(latency=0))
                print('Do noting: Quiting...')
                break
