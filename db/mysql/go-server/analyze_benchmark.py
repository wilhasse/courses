#!/usr/bin/env python3
"""
Analyze and visualize benchmark results from benchmark.sh
"""

import csv
import sys
from collections import defaultdict

def load_results(filename='benchmark_results.csv'):
    """Load benchmark results from CSV file"""
    results = defaultdict(dict)
    
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                backend = row['Backend']
                test = row['Test']
                results[backend][test] = {
                    'total_time': float(row['Total_Time']),
                    'avg_time': float(row['Avg_Time']),
                    'qps': float(row['QPS'])
                }
    except FileNotFoundError:
        print(f"Error: {filename} not found. Run benchmark.sh first.")
        sys.exit(1)
    
    return results

def print_comparison(results):
    """Print performance comparison between backends"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    # Get all test names
    all_tests = set()
    for backend in results:
        all_tests.update(results[backend].keys())
    
    # Print results for each test
    for test in sorted(all_tests):
        print(f"\n{test}:")
        print("-" * len(test))
        
        # Find best QPS for this test
        best_qps = 0
        best_backend = ""
        for backend in results:
            if test in results[backend]:
                qps = results[backend][test]['qps']
                if qps > best_qps:
                    best_qps = qps
                    best_backend = backend
        
        # Print results for each backend
        for backend in sorted(results.keys()):
            if test in results[backend]:
                data = results[backend][test]
                qps = data['qps']
                avg_time = data['avg_time'] * 1000  # Convert to ms
                
                # Calculate relative performance
                if best_qps > 0:
                    relative = (qps / best_qps) * 100
                else:
                    relative = 0
                
                # Mark the best performer
                marker = " ⭐" if backend == best_backend else ""
                
                print(f"  {backend:10} - QPS: {qps:8.2f} | Avg: {avg_time:6.2f}ms | "
                      f"Relative: {relative:5.1f}%{marker}")

def print_summary(results):
    """Print overall summary"""
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Calculate average QPS across all tests for each backend
    backend_avg = {}
    for backend in results:
        total_qps = sum(results[backend][test]['qps'] for test in results[backend])
        test_count = len(results[backend])
        backend_avg[backend] = total_qps / test_count if test_count > 0 else 0
    
    # Sort by average QPS
    sorted_backends = sorted(backend_avg.items(), key=lambda x: x[1], reverse=True)
    
    print("\nAverage QPS across all tests:")
    for backend, avg_qps in sorted_backends:
        print(f"  {backend:10} - {avg_qps:8.2f} queries/second")
    
    # Performance recommendations
    print("\n📊 Performance Insights:")
    if len(sorted_backends) > 0:
        best = sorted_backends[0][0]
        print(f"  • {best} shows the best overall performance")
        
        if len(sorted_backends) > 1:
            second = sorted_backends[1][0]
            improvement = ((sorted_backends[0][1] / sorted_backends[1][1]) - 1) * 100
            print(f"  • {best} is {improvement:.1f}% faster than {second} on average")

def main():
    """Main function"""
    print("Loading benchmark results...")
    results = load_results()
    
    if not results:
        print("No results found. Run benchmark.sh first.")
        return
    
    print_comparison(results)
    print_summary(results)
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()