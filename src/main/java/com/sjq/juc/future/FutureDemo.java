package com.sjq.juc.future;

import java.util.concurrent.*;

/**
 * @Author Kemp
 * @create 2023/12/25 22:53
 */
public class FutureDemo {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        demo1();
    }

    public static void demo1() throws ExecutionException, InterruptedException {
        ExecutorService executorService = Executors.newFixedThreadPool(3);

        FutureTask<String> task = new FutureTask<>(() -> {
            System.out.println("futureTask running~~");
            TimeUnit.SECONDS.sleep(3);
            return "end";
        });

        executorService.submit(task);

        while (!task.isDone()){
            System.out.println("asyn task is not done");

            if(task.isDone()){
                String futureTaskRes = task.get();
                System.out.println("futureTask done: " + futureTaskRes);
            }
        }

        executorService.shutdown();
    }
}
