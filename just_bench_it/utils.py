def print_results(results):
    print("Benchmark Results:")
    for env_name, score in results.items():
        print(f"{env_name}: {score:.2f}")
