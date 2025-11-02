import time, json, os
from aes_handler import AESHandler
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import psutil

# Uncomment if you want to use accuracy/r2/roc:
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, confusion_matrix, roc_curve

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

conf = yaml.safe_load(open('config.yaml'))
dataset = pd.read_csv(conf['dataset']['file'])
features = conf['dataset']['features']
label_col = conf['dataset'].get('label_col', None)
sample_size = conf['dataset'].get('sample_size', len(dataset))
dataset = dataset[features + ([label_col] if label_col else [])]
dataset = dataset.iloc[:sample_size]  # No random sampling, just head

def resource_stats():
    process = psutil.Process(os.getpid())
    cpu = psutil.cpu_percent(interval=None)
    mem = process.memory_info().rss / (1024 * 1024)  # MB
    return cpu, mem

def benchmark(mode, key_size):
    aes = AESHandler()
    enc_times, dec_times, size_overheads = [], [], []
    success_count, error_count = 0, 0
    resource_samples = []

    for _, row in dataset.iterrows():
        # ... (processing code unchanged) ...
        # append to resource_samples, etc.
        pass

    # handle possible empty results
    avg_enc_ms = np.mean(enc_times) if enc_times else 0
    std_enc_ms = np.std(enc_times) if enc_times else 0
    avg_dec_ms = np.mean(dec_times) if dec_times else 0
    std_dec_ms = np.std(dec_times) if dec_times else 0
    avg_size_overhead = np.mean(size_overheads) if size_overheads else 0
    throughput_ops_per_sec = len(dataset) / (np.sum(enc_times) / 1000) if enc_times else 0
    avg_plaintext_size = np.mean([len(json.dumps(row[features].to_dict()).encode('utf-8')) for _, row in dataset.iterrows()]) if len(dataset) > 0 else 0

    if resource_samples:
        avg_cpu = np.mean([x[0] for x in resource_samples])
        max_mem = np.max([x[1] for x in resource_samples])
    else:
        avg_cpu = 0
        max_mem = 0

    success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0

    return {
        'mode': mode,
        'key_size': key_size,
        'avg_enc_ms': avg_enc_ms,
        'std_enc_ms': std_enc_ms,
        'avg_dec_ms': avg_dec_ms,
        'std_dec_ms': std_dec_ms,
        'avg_size_overhead_bytes': avg_size_overhead,
        'avg_plaintext_size': avg_plaintext_size,
        'throughput_ops_per_sec': throughput_ops_per_sec,
        'success_rate': success_rate,
        'avg_cpu': avg_cpu,
        'max_memory_mb': max_mem,
        'enc_times_raw': enc_times,
        'dec_times_raw': dec_times,
        'size_overheads_raw': size_overheads
    }

def create_visualizations(results):
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('AES-GCM Performance Benchmarking (MQTT IoT)', fontsize=16, fontweight='bold')

    # Latency Comparison
    ax1 = plt.subplot(2, 3, 1)
    x = [r['key_size'] for r in results]
    y_enc = [r['avg_enc_ms'] for r in results]
    y_dec = [r['avg_dec_ms'] for r in results]
    ax1.plot(x, y_enc, 'o-', label='Encrypt', linewidth=2, markersize=8)
    ax1.plot(x, y_dec, 's--', label='Decrypt', linewidth=2, markersize=8)
    ax1.set_title('Encryption/Decryption Latency')
    ax1.set_xlabel('Key Size (bits)')
    ax1.set_ylabel('Time (ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Throughput
    ax2 = plt.subplot(2, 3, 2)
    throughput = [r['throughput_ops_per_sec'] for r in results]
    ax2.bar(x, throughput, color='teal')
    ax2.set_title('System Throughput')
    ax2.set_xlabel('Key Size (bits)')
    ax2.set_ylabel('Ops/sec')

    # Size Overhead
    ax3 = plt.subplot(2, 3, 3)
    overheads = [r['avg_size_overhead_bytes'] for r in results]
    ax3.bar(x, overheads, color='orange')
    ax3.set_title('Size Overhead (Bytes)')
    ax3.set_xlabel('Key Size (bits)')
    ax3.set_ylabel('Overhead (bytes)')

    # Boxplot for latency distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.boxplot([r['enc_times_raw'] for r in results], labels=[f"{r['key_size']}-{r['mode']}" for r in results])
    ax4.set_title('Encryption Time Distribution')
    ax4.set_xlabel('AES-GCM Config')
    ax4.set_ylabel('Encrypt time (ms)')
    ax4.grid(True, alpha=0.3)

    # Success Rate, CPU, Memory
    ax5 = plt.subplot(2, 3, 5)
    success_rates = [r['success_rate'] for r in results]
    ax5.bar(x, success_rates, color='lightgreen')
    ax5.set_title('Decryption Success Rate')
    ax5.set_xlabel('Key Size (bits)')
    ax5.set_ylabel('Success Rate')

    ax6 = plt.subplot(2, 3, 6)
    cpu = [r['avg_cpu'] for r in results]
    mem = [r['max_memory_mb'] for r in results]
    ax6.plot(x, cpu, 'o-', label='CPU (%)')
    ax6.plot(x, mem, 's--', label='Max Memory (MB)')
    ax6.set_title('Resource Use')
    ax6.set_xlabel('Key Size (bits)')
    ax6.set_ylabel('Resource')
    ax6.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("aes_performance_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def report(results):
    print("\n" + "="*80)
    print("AES-GCM PERFORMANCE REPORT (IoT Healthcare Dataset)")
    print("="*80)
    for r in results:
        print(f"\nConfig: AES-GCM-{r['key_size']}")
        print(f"Enc Time: {r['avg_enc_ms']:.2f} ± {r['std_enc_ms']:.2f} ms")
        print(f"Dec Time: {r['avg_dec_ms']:.2f} ± {r['std_dec_ms']:.2f} ms")
        print(f"Size Overhead: {r['avg_size_overhead_bytes']:.0f} B ({(r['avg_size_overhead_bytes']/r['avg_plaintext_size']*100):.1f}%)")
        print(f"Throughput: {r['throughput_ops_per_sec']:.0f} ops/sec")
        print(f"Success Rate: {r['success_rate']:.2%}")
        print(f"Avg CPU: {r['avg_cpu']:.2f}, Max Mem: {r['max_memory_mb']:.2f} MB")
        if r['accuracy'] is not None:
            print(f"Accuracy: {r['accuracy']:.2f}")
        if r['r2'] is not None:
            print(f"R² Score: {r['r2']:.2f}")
        if r['roc_auc'] is not None:
            print(f"ROC AUC: {r['roc_auc']:.2f}")

if __name__ == "__main__":
    print("Starting AES-GCM benchmarking...")
    results = []
    for key_size in conf['encryption']['key_sizes']:
        for mode in conf['encryption']['modes']:
            print(f"Benchmarking AES-GCM-{key_size}...")
            r = benchmark(mode, key_size)
            results.append(r)
    create_visualizations(results)
    report(results)
    print("\n✅ Benchmark complete. Visualizations are saved as 'aes_performance_analysis.png'")
