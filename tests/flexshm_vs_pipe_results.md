# Flexible SHM vs Pipe Benchmark Results

**Configuration:**

- Flexible SHM: slots=1 (latest-value)

- Pipe: multiprocessing.Pipe

- Iterations: 500 per configuration

- Timing: Write = copy to mechanism + finalize/send

- Latency: finalize/send done → blocking read/recv done



## SensorData: 100 float64 + 40 bool (140 fields)


| Change Rate | Flex SHM Write µ±σ (µs) | Pipe Write µ±σ (µs) | Flex SHM Latency µ±σ (µs) | Pipe Latency µ±σ (µs) |
|-------------|------------------------|---------------------|--------------------------|----------------------|
| 5% | 21.4±1.7 | 27.6±3.4 | 282.2±423.1 | 21.5±235.0 |
| 10% | 28.2±1.9 | 56.7±13.2 | 280.9±425.7 | 41131018.7±56548335.8 |
| 20% | 42.1±2.6 | 80.3±4.5 | 280.3±426.7 | 26.3±11.3 |
| 50% | 79.9±2.8 | 189.2±8.6 | 293.8±674.2 | 89.7±73.4 |
| 100% | 140.4±3.0 | 363.6±36.6 | 282.3±417.7 | 283.9±869.8 |


## RobotData: 40 float32 + 60 bool + 2× 2MP images

*(All joints/status updated every iteration, images every 4th iteration - 25fps @ 100Hz)*


| Test | Flex SHM Write µ±σ (µs) | Pipe Write µ±σ (µs) | Flex SHM Latency µ±σ (µs) | Pipe Latency µ±σ (µs) |
|------|------------------------|---------------------|--------------------------|----------------------|
| Full Update | 550.5±805.0 | 3389.2±6029.7 | 514.2±227.0 | 523.6±685.8 |

