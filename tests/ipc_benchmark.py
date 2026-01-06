"""
Flexible Shared Memory vs Pipe Benchmark
=========================================

Vergleicht Flexible SHM (slots=1) mit multiprocessing.Pipe
für partial updates bei verschiedenen Change Rates.

Szenarien:
1. SensorData: 100 float64 + 40 bool (140 Felder)
2. RobotData: 40 float32 + 60 bool + 2× 1920×1080 uint8 Bilder
   - Bilder: alle 4 Iterationen neu (25fps @ 100Hz)
   - Andere Werte: jede Iteration neu

Change Rates: 5%, 10%, 20%, 50%, 100%
Iterationen: 500 pro Konfiguration

Messung:
- Write-Zeit: Kopieren in Mechanismus bis finalize()/send()
- Latenz: finalize()/send() Ende bis read()/recv() Ende (blockierend)
"""

import time
import multiprocessing as mp
from multiprocessing import Pipe, Process
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any
import sys

# Flexible SHM Import
try:
    from flexible_shared_memory import SharedMemory as FlexibleSHM
    FLEXIBLE_SHM_AVAILABLE = True
except ImportError:
    FLEXIBLE_SHM_AVAILABLE = False
    print("ERROR: flexible_shared_memory nicht verfügbar!")
    sys.exit(1)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensorData:
    """140 einzelne Felder: 100 float64 Sensoren + 40 bool Status"""
    sensor_00: float
    sensor_01: float
    sensor_02: float
    sensor_03: float
    sensor_04: float
    sensor_05: float
    sensor_06: float
    sensor_07: float
    sensor_08: float
    sensor_09: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float
    sensor_22: float
    sensor_23: float
    sensor_24: float
    sensor_25: float
    sensor_26: float
    sensor_27: float
    sensor_28: float
    sensor_29: float
    sensor_30: float
    sensor_31: float
    sensor_32: float
    sensor_33: float
    sensor_34: float
    sensor_35: float
    sensor_36: float
    sensor_37: float
    sensor_38: float
    sensor_39: float
    sensor_40: float
    sensor_41: float
    sensor_42: float
    sensor_43: float
    sensor_44: float
    sensor_45: float
    sensor_46: float
    sensor_47: float
    sensor_48: float
    sensor_49: float
    sensor_50: float
    sensor_51: float
    sensor_52: float
    sensor_53: float
    sensor_54: float
    sensor_55: float
    sensor_56: float
    sensor_57: float
    sensor_58: float
    sensor_59: float
    sensor_60: float
    sensor_61: float
    sensor_62: float
    sensor_63: float
    sensor_64: float
    sensor_65: float
    sensor_66: float
    sensor_67: float
    sensor_68: float
    sensor_69: float
    sensor_70: float
    sensor_71: float
    sensor_72: float
    sensor_73: float
    sensor_74: float
    sensor_75: float
    sensor_76: float
    sensor_77: float
    sensor_78: float
    sensor_79: float
    sensor_80: float
    sensor_81: float
    sensor_82: float
    sensor_83: float
    sensor_84: float
    sensor_85: float
    sensor_86: float
    sensor_87: float
    sensor_88: float
    sensor_89: float
    sensor_90: float
    sensor_91: float
    sensor_92: float
    sensor_93: float
    sensor_94: float
    sensor_95: float
    sensor_96: float
    sensor_97: float
    sensor_98: float
    sensor_99: float
    status_00: bool
    status_01: bool
    status_02: bool
    status_03: bool
    status_04: bool
    status_05: bool
    status_06: bool
    status_07: bool
    status_08: bool
    status_09: bool
    status_10: bool
    status_11: bool
    status_12: bool
    status_13: bool
    status_14: bool
    status_15: bool
    status_16: bool
    status_17: bool
    status_18: bool
    status_19: bool
    status_20: bool
    status_21: bool
    status_22: bool
    status_23: bool
    status_24: bool
    status_25: bool
    status_26: bool
    status_27: bool
    status_28: bool
    status_29: bool
    status_30: bool
    status_31: bool
    status_32: bool
    status_33: bool
    status_34: bool
    status_35: bool
    status_36: bool
    status_37: bool
    status_38: bool
    status_39: bool


@dataclass
class RobotData:
    """Robot: 40 float32 + 60 bool + 2× 1920×1080 uint8 Bilder"""
    joint_00: float
    joint_01: float
    joint_02: float
    joint_03: float
    joint_04: float
    joint_05: float
    joint_06: float
    joint_07: float
    joint_08: float
    joint_09: float
    joint_10: float
    joint_11: float
    joint_12: float
    joint_13: float
    joint_14: float
    joint_15: float
    joint_16: float
    joint_17: float
    joint_18: float
    joint_19: float
    joint_20: float
    joint_21: float
    joint_22: float
    joint_23: float
    joint_24: float
    joint_25: float
    joint_26: float
    joint_27: float
    joint_28: float
    joint_29: float
    joint_30: float
    joint_31: float
    joint_32: float
    joint_33: float
    joint_34: float
    joint_35: float
    joint_36: float
    joint_37: float
    joint_38: float
    joint_39: float
    status_00: bool
    status_01: bool
    status_02: bool
    status_03: bool
    status_04: bool
    status_05: bool
    status_06: bool
    status_07: bool
    status_08: bool
    status_09: bool
    status_10: bool
    status_11: bool
    status_12: bool
    status_13: bool
    status_14: bool
    status_15: bool
    status_16: bool
    status_17: bool
    status_18: bool
    status_19: bool
    status_20: bool
    status_21: bool
    status_22: bool
    status_23: bool
    status_24: bool
    status_25: bool
    status_26: bool
    status_27: bool
    status_28: bool
    status_29: bool
    status_30: bool
    status_31: bool
    status_32: bool
    status_33: bool
    status_34: bool
    status_35: bool
    status_36: bool
    status_37: bool
    status_38: bool
    status_39: bool
    status_40: bool
    status_41: bool
    status_42: bool
    status_43: bool
    status_44: bool
    status_45: bool
    status_46: bool
    status_47: bool
    status_48: bool
    status_49: bool
    status_50: bool
    status_51: bool
    status_52: bool
    status_53: bool
    status_54: bool
    status_55: bool
    status_56: bool
    status_57: bool
    status_58: bool
    status_59: bool
    camera_1: "uint8[1920,1080]"  # 2 Megapixel (Full-HD)
    camera_2: "uint8[1920,1080]"  # 2 Megapixel (Full-HD)


# =============================================================================
# DATA GENERATORS
# =============================================================================

def create_initial_sensor_data() -> SensorData:
    """Create initial SensorData with random values"""
    sensor_vals = np.random.rand(100)
    status_vals = np.random.rand(40) > 0.5
    
    return SensorData(**{
        **{f'sensor_{i:02d}': sensor_vals[i] for i in range(100)},
        **{f'status_{i:02d}': bool(status_vals[i]) for i in range(40)}
    })


def create_initial_robot_data() -> RobotData:
    """Create initial RobotData with random values"""
    joint_vals = np.random.rand(40).astype(np.float32)
    status_vals = np.random.rand(60) > 0.5
    cam1 = np.random.randint(0, 256, (1920, 1080), dtype=np.uint8)
    cam2 = np.random.randint(0, 256, (1920, 1080), dtype=np.uint8)
    
    return RobotData(
        **{f'joint_{i:02d}': float(joint_vals[i]) for i in range(40)},
        **{f'status_{i:02d}': bool(status_vals[i]) for i in range(60)},
        camera_1=cam1,
        camera_2=cam2
    )


def generate_sensor_updates(change_rate: float, iteration: int) -> Dict[str, Any]:
    """Generate random field updates for SensorData"""
    all_fields = [f'sensor_{i:02d}' for i in range(100)] + \
                 [f'status_{i:02d}' for i in range(40)]
    
    num_changes = max(1, int(len(all_fields) * change_rate))
    changed_fields = np.random.choice(all_fields, num_changes, replace=False)
    
    updates = {}
    for field in changed_fields:
        if field.startswith('sensor_'):
            updates[field] = np.random.rand()
        else:  # status
            updates[field] = bool(np.random.rand() > 0.5)
    
    return updates



def generate_robot_updates(iteration: int) -> Dict[str, Any]:
    """Generate updates for RobotData (alle Felder immer, Bilder nur alle 4 Iterationen)"""
    updates = {}
    
    # Alle joints und status IMMER ändern
    for i in range(40):
        updates[f'joint_{i:02d}'] = float(np.random.rand())
    for i in range(60):
        updates[f'status_{i:02d}'] = bool(np.random.rand() > 0.5)
    
    # Bilder alle 4 Iterationen
    if iteration % 4 == 0:
        updates['camera_1'] = np.random.randint(0, 256, (1920, 1080), dtype=np.uint8)
        updates['camera_2'] = np.random.randint(0, 256, (1920, 1080), dtype=np.uint8)
    
    return updates



# =============================================================================
# WORKER PROCESSES - FLEXIBLE SHM
# =============================================================================

def writer_flexshm(data_class, scenario_name: str, change_rate: float, 
                   num_runs: int, ready_queue: mp.Queue, 
                   result_queue: mp.Queue, timestamp_array, connection_queue: mp.Queue,
                   done_queue: mp.Queue, reader_done_array):
    """Writer using Flexible SHM (slots=1) - Ping-Pong Pattern"""
    
    try:
        from flexible_shared_memory import SharedMemory as FlexibleSHM
        
        # Create initial data template
        if scenario_name == "sensor":
            data_template = create_initial_sensor_data()
        else:  # robot
            data_template = create_initial_robot_data()
        
        # Setup Flexible SHM with 1 slot (latest-value)
        shm = FlexibleSHM(data_class, slots=1)
        
        # Send SHM name to reader
        connection_queue.put(shm.name)
        
        # Signal ready
        ready_queue.put("ready")
        
        # Benchmark
        write_times = []
        
        for run in range(num_runs):
            # Generate updates (NOT MEASURED)
            if scenario_name == "sensor":
                updates = generate_sensor_updates(change_rate, run)
            else:  # robot
                updates = generate_robot_updates(run)
            
            # MEASURE: Write + finalize
            start = time.perf_counter()
            shm.write(**updates)
            shm.finalize()
            end = time.perf_counter()
            
            write_time = (end - start) * 1_000_000  # µs
            write_times.append(write_time)
            
            # Signal Reader: Data ready!
            timestamp_array[run] = end
            
            # PING-PONG: Wait for Reader to finish THIS iteration
            timeout_start = time.perf_counter()
            while reader_done_array[run] == 0:
                if time.perf_counter() - timeout_start > 10.0:
                    raise TimeoutError(f"Reader didn't finish iteration {run}")
                # Busy-wait (kein sleep für genaue Messung)
        
        # Send results FIRST
        result_queue.put({'write_times': write_times})
        
        # Wait for reader to finish completely
        done_queue.get(timeout=30)
        
        # Small delay before cleanup
        time.sleep(0.2)
        
        # Cleanup
        shm.unlink()
        
    except Exception as e:
        import traceback
        result_queue.put({'error': f"{e}\n{traceback.format_exc()}"})


def reader_flexshm(data_class, scenario_name: str, num_runs: int,
                   ready_queue: mp.Queue, result_queue: mp.Queue, 
                   timestamp_array, connection_queue: mp.Queue,
                   done_queue: mp.Queue, reader_done_array):
    """Reader using Flexible SHM - Ping-Pong Pattern"""
    
    try:
        from flexible_shared_memory import SharedMemory as FlexibleSHM
        
        # Get SHM name from writer
        shm_name = connection_queue.get(timeout=30)
        
        # Wait for writer ready
        ready_queue.get(timeout=30)
        
        # Connect to SHM
        shm = FlexibleSHM(shm_name, expected_type=data_class)
        
        # Benchmark
        latencies = []
        
        for run in range(num_runs):
            # Wait for Writer to signal data ready
            timeout_start = time.perf_counter()
            while timestamp_array[run] < 0.0:
                if time.perf_counter() - timeout_start > 10.0:
                    raise TimeoutError(f"Timeout waiting for writer at iteration {run}")
                # Busy-wait (kein sleep!)
            
            write_done_ts = timestamp_array[run]
            
            # Read data (sollte sofort verfügbar sein!)
            data = shm.read(timeout=5.0)
            read_done_ts = time.perf_counter()
            
            if data is None:
                raise ValueError(f"Read returned None at iteration {run}")
            
            # Calculate latency
            latency = (read_done_ts - write_done_ts) * 1_000_000  # µs
            latencies.append(latency)
            
            # PING-PONG: Signal Writer we're done with THIS iteration
            reader_done_array[run] = 1
        
        # Send results FIRST
        result_queue.put({'latencies': latencies})
        
        # Signal writer we're done
        done_queue.put("done")
        
        # Small delay before cleanup
        time.sleep(0.1)
        
        # Cleanup
        shm.close()
        
    except Exception as e:
        import traceback
        # Signal even on error
        try:
            done_queue.put("error")
        except:
            pass
        result_queue.put({'error': f"{e}\n{traceback.format_exc()}"})





# =============================================================================
# WORKER PROCESSES - PIPE
# =============================================================================

def writer_pipe(scenario_name: str, change_rate: float, num_runs: int,
                ready_queue: mp.Queue, result_queue: mp.Queue, 
                timestamp_array, pipe_conn, reader_done_array):
    """Writer using Pipe - Ping-Pong Pattern"""
    
    try:
        # Signal ready
        ready_queue.put("ready")
        
        # Benchmark
        write_times = []
        
        for run in range(num_runs):
            # Generate updates (NOT MEASURED)
            if scenario_name == "sensor":
                updates = generate_sensor_updates(change_rate, run)
            else:  # robot
                updates = generate_robot_updates(run)
            
            # Convert to list for pipe
            update_list = list(updates.items())
            
            # MEASURE: Send
            start = time.perf_counter()
            pipe_conn.send(update_list)
            end = time.perf_counter()
            
            write_time = (end - start) * 1_000_000  # µs
            write_times.append(write_time)
            
            # Signal Reader: Data sent!
            timestamp_array[run] = end
            
            # PING-PONG: Wait for Reader to finish THIS iteration
            timeout_start = time.perf_counter()
            while reader_done_array[run] == 0:
                if time.perf_counter() - timeout_start > 10.0:
                    raise TimeoutError(f"Reader didn't finish iteration {run}")
                # Busy-wait
        
        # Send results
        result_queue.put({'write_times': write_times})
        
    except Exception as e:
        import traceback
        result_queue.put({'error': f"{e}\n{traceback.format_exc()}"})





def reader_pipe(num_runs: int, ready_queue: mp.Queue, 
                result_queue: mp.Queue, timestamp_array, pipe_conn, reader_done_array):
    """Reader using Pipe - Ping-Pong Pattern"""
    
    try:
        # Wait for writer ready
        ready_queue.get(timeout=30)
        
        # Benchmark
        latencies = []
        
        for run in range(num_runs):
            # Bei Pipe: KEIN Wait auf Timestamp!
            # recv() blockiert bis Daten da sind (Pipe synchronisiert selbst)
            
            update_list = pipe_conn.recv()  # Blockiert bis send() fertig
            read_done_ts = time.perf_counter()
            
            # JETZT erst Timestamp holen (Writer hat ihn inzwischen gesetzt)
            write_done_ts = timestamp_array[run]
            
            # Calculate latency
            latency = (read_done_ts - write_done_ts) * 1_000_000  # µs
            latencies.append(latency)
            
            # PING-PONG: Signal Writer we're done
            reader_done_array[run] = 1
        
        # Send results
        result_queue.put({'latencies': latencies})
        
    except Exception as e:
        import traceback
        result_queue.put({'error': f"{e}\n{traceback.format_exc()}"})





# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_single_test(mechanism: str, data_class, scenario_name: str, 
                   change_rate: float, num_runs: int = 500) -> Dict:
    """Run a single benchmark configuration"""
    
    # Create communication channels
    ready_queue = mp.Queue()
    writer_result_queue = mp.Queue()
    reader_result_queue = mp.Queue()
    connection_queue = mp.Queue()
    done_queue = mp.Queue()
    
    # Shared timestamp array
    timestamp_array = mp.Array('d', num_runs)
    for i in range(num_runs):
        timestamp_array[i] = -1.0
    
    # Shared reader_done array (Ping-Pong Synchronisation!)
    reader_done_array = mp.Array('i', num_runs)
    for i in range(num_runs):
        reader_done_array[i] = 0
    
    if mechanism == "flexshm":
        writer = Process(target=writer_flexshm,
                        args=(data_class, scenario_name, change_rate, num_runs,
                              ready_queue, writer_result_queue, timestamp_array, 
                              connection_queue, done_queue, reader_done_array))
        reader = Process(target=reader_flexshm,
                        args=(data_class, scenario_name, num_runs,
                              ready_queue, reader_result_queue, timestamp_array, 
                              connection_queue, done_queue, reader_done_array))
    else:  # pipe
        reader_conn, writer_conn = Pipe(duplex=False)
        
        writer = Process(target=writer_pipe,
                        args=(scenario_name, change_rate, num_runs,
                              ready_queue, writer_result_queue, timestamp_array, 
                              writer_conn, reader_done_array))
        reader = Process(target=reader_pipe,
                        args=(num_runs, ready_queue, reader_result_queue, 
                              timestamp_array, reader_conn, reader_done_array))
    
    # Run
    start_time = time.time()
    writer.start()
    reader.start()
    
    writer.join(timeout=300)
    reader.join(timeout=300)
    
    if writer.is_alive() or reader.is_alive():
        writer.terminate()
        reader.terminate()
        return {'error': 'Timeout'}
    
    # Get results
    try:
        writer_result = writer_result_queue.get(timeout=5)
        reader_result = reader_result_queue.get(timeout=5)
        
        if 'error' in writer_result or 'error' in reader_result:
            return {'error': writer_result.get('error', reader_result.get('error'))}
        
        write_times = np.array(writer_result['write_times'])
        latencies = np.array(reader_result['latencies'])
        
        return {
            'write_mean': np.mean(write_times),
            'write_std': np.std(write_times),
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'elapsed': time.time() - start_time
        }
        
    except Exception as e:
        return {'error': str(e)}




def run_all_benchmarks(num_runs: int = 500):
    """Run complete benchmark suite"""
    
    scenarios = [
        ('sensor', SensorData, 'SensorData: 100 float64 + 40 bool'),
        ('robot', RobotData, 'RobotData: 40 float32 + 60 bool + 2× 2MP images'),
    ]
    
    results = {}
    
    print("=" * 80)
    print("Flexible SHM vs Pipe Benchmark")
    print("=" * 80)
    print(f"Runs per test: {num_runs}")
    print(f"Flexible SHM: slots=1 (latest-value)")
    print("=" * 80)
    print()
    
    for scenario_id, data_class, scenario_desc in scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_desc}")
        print(f"{'='*80}\n")
        
        results[scenario_id] = {}
        
        # Sensor: verschiedene Change-Rates
        if scenario_id == 'sensor':
            change_rates = [0.05, 0.10, 0.20, 0.50, 1.00]
            
            for change_rate in change_rates:
                rate_pct = int(change_rate * 100)
                results[scenario_id][rate_pct] = {}
                
                print(f"Change rate: {rate_pct}%")
                
                # Flexible SHM
                print(f"  Testing Flexible SHM... ", end='', flush=True)
                result = run_single_test("flexshm", data_class, scenario_id, 
                                        change_rate, num_runs)
                if 'error' in result:
                    print(f"ERROR: {result['error']}")
                    results[scenario_id][rate_pct]['flexshm'] = result
                else:
                    print(f"✓ ({result['elapsed']:.1f}s)")
                    results[scenario_id][rate_pct]['flexshm'] = result
                
                # Pipe
                print(f"  Testing Pipe... ", end='', flush=True)
                result = run_single_test("pipe", data_class, scenario_id, 
                                        change_rate, num_runs)
                if 'error' in result:
                    print(f"ERROR: {result['error']}")
                    results[scenario_id][rate_pct]['pipe'] = result
                else:
                    print(f"✓ ({result['elapsed']:.1f}s)")
                    results[scenario_id][rate_pct]['pipe'] = result
        
        # Robot: nur ein Test (100% non-image Felder, Bilder alle 4 Iterationen)
        else:
            results[scenario_id]['full'] = {}
            
            print("All joints/status every iteration, images every 4th iteration (25fps @ 100Hz)")
            
            # Flexible SHM
            print(f"  Testing Flexible SHM... ", end='', flush=True)
            result = run_single_test("flexshm", data_class, scenario_id, 
                                    1.0, num_runs)  # change_rate ignoriert
            if 'error' in result:
                print(f"ERROR: {result['error']}")
                results[scenario_id]['full']['flexshm'] = result
            else:
                print(f"✓ ({result['elapsed']:.1f}s)")
                results[scenario_id]['full']['flexshm'] = result
            
            # Pipe
            print(f"  Testing Pipe... ", end='', flush=True)
            result = run_single_test("pipe", data_class, scenario_id, 
                                    1.0, num_runs)  # change_rate ignoriert
            if 'error' in result:
                print(f"ERROR: {result['error']}")
                results[scenario_id]['full']['pipe'] = result
            else:
                print(f"✓ ({result['elapsed']:.1f}s)")
                results[scenario_id]['full']['pipe'] = result
    
    return results, scenarios





def format_results_markdown(results: Dict) -> str:
    """Format results as markdown tables"""
    
    md = ["# Flexible SHM vs Pipe Benchmark Results\n"]
    md.append("**Configuration:**\n")
    md.append("- Flexible SHM: slots=1 (latest-value)\n")
    md.append("- Pipe: multiprocessing.Pipe\n")
    md.append("- Iterations: 500 per configuration\n")
    md.append("- Timing: Write = copy to mechanism + finalize/send\n")
    md.append("- Latency: finalize/send done → blocking read/recv done\n")
    md.append("\n")
    
    # SensorData
    if 'sensor' in results:
        md.append("## SensorData: 100 float64 + 40 bool (140 fields)\n\n")
        md.append("| Change Rate | Flex SHM Write µ±σ (µs) | Pipe Write µ±σ (µs) | Flex SHM Latency µ±σ (µs) | Pipe Latency µ±σ (µs) |")
        md.append("|-------------|------------------------|---------------------|--------------------------|----------------------|")
        
        for rate_pct in sorted(results['sensor'].keys()):
            row_data = results['sensor'][rate_pct]
            
            flex_result = row_data.get('flexshm', {})
            pipe_result = row_data.get('pipe', {})
            
            if 'error' in flex_result:
                flex_write = "ERROR"
                flex_latency = "ERROR"
            else:
                flex_write = f"{flex_result['write_mean']:.1f}±{flex_result['write_std']:.1f}"
                flex_latency = f"{flex_result['latency_mean']:.1f}±{flex_result['latency_std']:.1f}"
            
            if 'error' in pipe_result:
                pipe_write = "ERROR"
                pipe_latency = "ERROR"
            else:
                pipe_write = f"{pipe_result['write_mean']:.1f}±{pipe_result['write_std']:.1f}"
                pipe_latency = f"{pipe_result['latency_mean']:.1f}±{pipe_result['latency_std']:.1f}"
            
            md.append(f"| {rate_pct}% | {flex_write} | {pipe_write} | {flex_latency} | {pipe_latency} |")
        
        md.append("\n")
    
    # RobotData
    if 'robot' in results:
        md.append("## RobotData: 40 float32 + 60 bool + 2× 2MP images\n")
        md.append("*(All joints/status updated every iteration, images every 4th iteration - 25fps @ 100Hz)*\n\n")
        md.append("| Test | Flex SHM Write µ±σ (µs) | Pipe Write µ±σ (µs) | Flex SHM Latency µ±σ (µs) | Pipe Latency µ±σ (µs) |")
        md.append("|------|------------------------|---------------------|--------------------------|----------------------|")
        
        row_data = results['robot']['full']
        
        flex_result = row_data.get('flexshm', {})
        pipe_result = row_data.get('pipe', {})
        
        if 'error' in flex_result:
            flex_write = "ERROR"
            flex_latency = "ERROR"
        else:
            flex_write = f"{flex_result['write_mean']:.1f}±{flex_result['write_std']:.1f}"
            flex_latency = f"{flex_result['latency_mean']:.1f}±{flex_result['latency_std']:.1f}"
        
        if 'error' in pipe_result:
            pipe_write = "ERROR"
            pipe_latency = "ERROR"
        else:
            pipe_write = f"{pipe_result['write_mean']:.1f}±{pipe_result['write_std']:.1f}"
            pipe_latency = f"{pipe_result['latency_mean']:.1f}±{pipe_result['latency_std']:.1f}"
        
        md.append(f"| Full Update | {flex_write} | {pipe_write} | {flex_latency} | {pipe_latency} |")
        
        md.append("\n")
    
    return "\n".join(md)




# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    mp.set_start_method('spawn')
    
    if not FLEXIBLE_SHM_AVAILABLE:
        print("ERROR: Flexible SHM not available!")
        sys.exit(1)
    
    # Run benchmarks
    results, scenarios = run_all_benchmarks(num_runs=500)
    
    # Format results
    markdown = format_results_markdown(results)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80 + "\n")
    print(markdown)
    
    # Save to file
    with open("flexshm_vs_pipe_results.md", "w") as f:
        f.write(markdown)
    
    print("\nResults saved to: flexshm_vs_pipe_results.md")