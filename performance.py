import time, json, os
from aes_handler import AESHandler
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

conf = yaml.safe_load(open('config.yaml'))
dataset = pd.read_csv(conf['dataset']['file']).sample(conf['dataset']['sample_size'])

def benchmark(mode, key_size):
    aes = AESHandler(key_size=key_size, mode=mode)
    enc_times, dec_times, size_overheads = [], [], []
    
    for _, row in dataset.iterrows():
        features = row[conf['dataset']['features']].to_dict()
        plaintext = json.dumps(features)

        # Measure encryption time
        start = time.time()
        encrypted = aes.encrypt(plaintext)
        enc_time = (time.time() - start) * 1000
        
        # Calculate size overhead
        ciphertext_size = len(json.dumps(encrypted).encode('utf-8'))
        plaintext_size = len(plaintext.encode('utf-8'))
        size_overhead = ciphertext_size - plaintext_size
        
        enc_times.append(enc_time)
        size_overheads.append(size_overhead)

        # Measure decryption time
        start = time.time()
        decrypted, dec_time = aes.decrypt(encrypted)
        dec_times.append(dec_time)

    return {
        'mode': mode,
        'key_size': key_size,
        'avg_enc_ms': np.mean(enc_times),
        'std_enc_ms': np.std(enc_times),
        'avg_dec_ms': np.mean(dec_times),
        'std_dec_ms': np.std(dec_times),
        'avg_size_overhead_bytes': np.mean(size_overheads),
        'avg_plaintext_size': np.mean([len(json.dumps(row[conf['dataset']['features']].to_dict()).encode('utf-8')) for _, row in dataset.iterrows()]),
        'throughput_ops_per_sec': len(dataset) / (np.sum(enc_times) / 1000),
        'enc_times_raw': enc_times,
        'dec_times_raw': dec_times,
        'size_overheads_raw': size_overheads
    }

def create_enhanced_visualizations(results):
    """Create comprehensive visualization suite for AES performance analysis"""
    
    # Set up the figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    fig.suptitle('Comprehensive AES Performance Analysis for IoT MQTT Communication', fontsize=16, fontweight='bold')
    
    # 1. Latency Comparison (Line Plot)
    ax1 = plt.subplot(3, 3, 1)
    for mode in set(r['mode'] for r in results):
        subset = [r for r in results if r['mode'] == mode]
        x = [r['key_size'] for r in subset]
        y_enc = [r['avg_enc_ms'] for r in subset]
        y_dec = [r['avg_dec_ms'] for r in subset]
        
        ax1.plot(x, y_enc, 'o-', label=f'{mode} Encrypt', linewidth=2, markersize=8)
        ax1.plot(x, y_dec, 's--', label=f'{mode} Decrypt', linewidth=2, markersize=8)
    
    ax1.set_title('Encryption/Decryption Latency', fontweight='bold')
    ax1.set_xlabel('Key Size (bits)')
    ax1.set_ylabel('Time (milliseconds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Size Overhead Comparison (Bar Plot)
    ax2 = plt.subplot(3, 3, 2)
    x_pos = np.arange(len(results))
    labels = [f"AES-{r['key_size']}-{r['mode']}" for r in results]
    overheads = [r['avg_size_overhead_bytes'] for r in results]
    colors = ['#FF6B6B' if r['mode'] == 'CBC' else '#4ECDC4' for r in results]
    
    bars = ax2.bar(x_pos, overheads, color=colors, alpha=0.8)
    ax2.set_title('Payload Size Overhead', fontweight='bold')
    ax2.set_xlabel('AES Configuration')
    ax2.set_ylabel('Size Overhead (bytes)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45)
    
    # Add value labels on bars
    for bar, overhead in zip(bars, overheads):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{overhead:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Throughput Comparison (Bar Plot)
    ax3 = plt.subplot(3, 3, 3)
    throughputs = [r['throughput_ops_per_sec'] for r in results]
    bars = ax3.bar(x_pos, throughputs, color=colors, alpha=0.8)
    ax3.set_title('System Throughput', fontweight='bold')
    ax3.set_xlabel('AES Configuration')
    ax3.set_ylabel('Operations per Second')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45)
    
    # Add value labels on bars
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{throughput:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Latency Distribution (Box Plot)
    ax4 = plt.subplot(3, 3, 4)
    enc_data = [r['enc_times_raw'] for r in results]
    bp = ax4.boxplot(enc_data, labels=[f"AES-{r['key_size']}-{r['mode']}" for r in results], patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    ax4.set_title('Encryption Time Distribution', fontweight='bold')
    ax4.set_xlabel('AES Configuration')
    ax4.set_ylabel('Encryption Time (ms)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Size Efficiency (Scatter Plot)
    ax5 = plt.subplot(3, 3, 5)
    for mode in set(r['mode'] for r in results):
        subset = [r for r in results if r['mode'] == mode]
        x = [r['avg_size_overhead_bytes'] for r in subset]
        y = [r['avg_enc_ms'] for r in subset]
        sizes = [r['key_size']/4 for r in subset]  # Scale for visibility
        
        ax5.scatter(x, y, s=sizes, alpha=0.7, label=f'{mode}', 
                   c=['red' if mode == 'CBC' else 'blue' for _ in subset])
    
    ax5.set_title('Size vs Speed Trade-off', fontweight='bold')
    ax5.set_xlabel('Size Overhead (bytes)')
    ax5.set_ylabel('Encryption Time (ms)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis('tight')
    ax6.axis('off')
    
    # Create performance summary table
    table_data = []
    for r in results:
        table_data.append([
            f"AES-{r['key_size']}-{r['mode']}",
            f"{r['avg_enc_ms']:.2f}",
            f"{r['avg_dec_ms']:.2f}",
            f"{r['avg_size_overhead_bytes']:.0f}",
            f"{r['throughput_ops_per_sec']:.0f}"
        ])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Config', 'Enc (ms)', 'Dec (ms)', 'Overhead (B)', 'Throughput (ops/s)'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    ax6.set_title('Performance Summary', fontweight='bold', pad=20)
    
    # 7. Security vs Performance Trade-off
    ax7 = plt.subplot(3, 3, 7)
    security_scores = {'CBC-128': 3, 'CBC-256': 4, 'GCM-128': 4, 'GCM-256': 5}  # Arbitrary security scoring
    
    for r in results:
        config = f"{r['mode']}-{r['key_size']}"
        security_score = security_scores.get(config, 3)
        performance_score = 5 - (r['avg_enc_ms'] / max([res['avg_enc_ms'] for res in results]) * 4)  # Inverse performance scoring
        
        color = 'red' if r['mode'] == 'CBC' else 'blue'
        ax7.scatter(security_score, performance_score, s=r['key_size'], 
                   c=color, alpha=0.7, label=f"AES-{r['key_size']}-{r['mode']}")
    
    ax7.set_title('Security vs Performance Trade-off', fontweight='bold')
    ax7.set_xlabel('Security Level (1-5)')
    ax7.set_ylabel('Performance Score (1-5)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Memory Efficiency Analysis
    ax8 = plt.subplot(3, 3, 8)
    memory_usage = [r['avg_plaintext_size'] + r['avg_size_overhead_bytes'] for r in results]
    efficiency = [r['throughput_ops_per_sec'] / mem for r, mem in zip(results, memory_usage)]
    
    bars = ax8.bar(x_pos, efficiency, color=colors, alpha=0.8)
    ax8.set_title('Memory Efficiency (Throughput/Memory)', fontweight='bold')
    ax8.set_xlabel('AES Configuration')
    ax8.set_ylabel('Efficiency Score')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(labels, rotation=45)
    
    # 9. Overall Recommendation Chart
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate overall scores
    scores = []
    for r in results:
        # Normalize metrics (lower is better for latency, higher is better for throughput)
        latency_score = 5 - (r['avg_enc_ms'] / max([res['avg_enc_ms'] for res in results]) * 4)
        throughput_score = (r['throughput_ops_per_sec'] / max([res['throughput_ops_per_sec'] for res in results]) * 5)
        size_score = 5 - (r['avg_size_overhead_bytes'] / max([res['avg_size_overhead_bytes'] for res in results]) * 4)
        
        overall_score = (latency_score + throughput_score + size_score) / 3
        scores.append(overall_score)
    
    bars = ax9.bar(x_pos, scores, color=colors, alpha=0.8)
    ax9.set_title('Overall Performance Recommendation', fontweight='bold')
    ax9.set_xlabel('AES Configuration')
    ax9.set_ylabel('Overall Score (1-5)')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels(labels, rotation=45)
    ax9.set_ylim(0, 5)
    
    # Highlight the best performer
    best_idx = scores.index(max(scores))
    bars[best_idx].set_color('gold')
    bars[best_idx].set_edgecolor('orange')
    bars[best_idx].set_linewidth(3)
    
    # Add recommendation text
    best_config = labels[best_idx]
    ax9.text(0.5, 0.95, f'Recommended: {best_config}', 
             transform=ax9.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8),
             fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('aes_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_performance_report(results):
    """Generate a detailed performance report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE AES PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nDataset: {conf['dataset']['file']}")
    print(f"Sample size: {conf['dataset']['sample_size']} records")
    print(f"Features analyzed: {', '.join(conf['dataset']['features'])}")
    
    print("\n" + "-"*60)
    print("DETAILED RESULTS BY CONFIGURATION")
    print("-"*60)
    
    for r in results:
        print(f"\n🔐 AES-{r['key_size']}-{r['mode']}:")
        print(f"   Encryption Time: {r['avg_enc_ms']:.3f} ± {r['std_enc_ms']:.3f} ms")
        print(f"   Decryption Time: {r['avg_dec_ms']:.3f} ± {r['std_dec_ms']:.3f} ms")
        print(f"   Size Overhead:   {r['avg_size_overhead_bytes']:.0f} bytes ({(r['avg_size_overhead_bytes']/r['avg_plaintext_size']*100):.1f}% increase)")
        print(f"   Throughput:      {r['throughput_ops_per_sec']:.0f} operations/second")
    
    # Find best performers
    best_encryption = min(results, key=lambda x: x['avg_enc_ms'])
    best_throughput = max(results, key=lambda x: x['throughput_ops_per_sec'])
    smallest_overhead = min(results, key=lambda x: x['avg_size_overhead_bytes'])
    
    print("\n" + "-"*60)
    print("KEY FINDINGS & RECOMMENDATIONS")
    print("-"*60)
    print(f"🚀 Fastest Encryption:    AES-{best_encryption['key_size']}-{best_encryption['mode']} ({best_encryption['avg_enc_ms']:.3f} ms)")
    print(f"⚡ Highest Throughput:    AES-{best_throughput['key_size']}-{best_throughput['mode']} ({best_throughput['throughput_ops_per_sec']:.0f} ops/s)")
    print(f"💾 Smallest Overhead:     AES-{smallest_overhead['key_size']}-{smallest_overhead['mode']} ({smallest_overhead['avg_size_overhead_bytes']:.0f} bytes)")
    
    print(f"\n🎯 For IoT Applications:")
    print(f"   • Real-time systems: Use AES-{best_encryption['key_size']}-{best_encryption['mode']} for lowest latency")
    print(f"   • High-volume systems: Use AES-{best_throughput['key_size']}-{best_throughput['mode']} for maximum throughput")
    print(f"   • Bandwidth-limited: Use AES-{smallest_overhead['key_size']}-{smallest_overhead['mode']} for minimal overhead")

if __name__ == "__main__":
    print("Starting comprehensive AES performance benchmarking...")
    print("This may take a few moments...")
    
    results = []
    for mode in conf['encryption']['modes']:
        for key_size in conf['encryption']['key_sizes']:
            print(f"Benchmarking AES-{key_size}-{mode}...")
            r = benchmark(mode, key_size)
            results.append(r)
    
    # Generate visualizations
    create_enhanced_visualizations(results)
    
    # Generate detailed report
    generate_performance_report(results)
    
    print(f"\n✅ Analysis complete! Visualization saved as 'aes_performance_analysis.png'")
